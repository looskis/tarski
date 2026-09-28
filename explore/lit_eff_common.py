"""Shared helpers for the explore/lit_eff_*.py scripts (literature-scan efficiency ideas; see
research_notes/explore/lit_efficiency.md). Not a runnable experiment.

Contents: logging and JSON output, dataset loading with a CPU smoke subsample, split indices, mean-pooled
features from a FeatureCache, a train/calibrate/evaluate wrapper around tarski.train.train_branch (which
enforces >= 300 optimiser steps, no stopping before half the schedule, last epoch below 50 validation
rows), a batched logistic regression, shared-covariance Gaussian (LDA) statistics, cold-start selectors,
Learn-then-Test threshold selection and timing.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from tarski import data as tdata
from tarski.branches import BlockBranch, ProbeBranch, mean_pool
from tarski.train import FeatureCache, _targets, evaluate, fit_temperature, predict_logits, train_branch
from tarski.trunk import Trunk


# ---------------------------------------------------------------------------------------------------
# Logging / output
# ---------------------------------------------------------------------------------------------------

class Log:
    def __init__(self, out: Optional[str]):
        self.f = None
        if out:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            self.f = open(out.replace(".json", ".log"), "a")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n")
            self.f.flush()


def _default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def save(res: Dict, out: Optional[str]) -> None:
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=1, default=_default)


def threads(n: int = 8) -> None:
    torch.set_num_threads(max(1, min(n, os.cpu_count() or 1)))


# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------

def load_ds(name: str, smoke: bool = False, seed: int = 0, workflows: Optional[Sequence[str]] = None,
            n_smoke=(160, 60, 80)) -> tdata.Dataset:
    """A tarski dataset; `workflows` keeps only those typed-decisions workflows; `smoke` subsamples."""
    ds = tdata.load(name)
    if workflows and name == "typed-decisions":
        tasks = {t: v for t, v in ds.tasks.items() if t.split(".")[0] in workflows}
        keep = lambda xs: [e for e in xs if any(t in e.y for t in tasks)]
        ds = tdata.Dataset(ds.name, tasks, keep(ds.train), keep(ds.val), keep(ds.test), ds.max_len)
    if smoke:
        rng = np.random.default_rng(seed)
        pick = lambda xs, k: [xs[i] for i in sorted(rng.choice(len(xs), min(k, len(xs)), replace=False))]
        if name == "clinc150":
            ins = lambda xs: [e for e in xs if e.y["oos"] == 0]
            oos = lambda xs: [e for e in xs if e.y["oos"] == 1]
            ds.train = pick(ins(ds.train), 300) + pick(oos(ds.train), 20)
            ds.val = pick(ins(ds.val), 60) + pick(oos(ds.val), 20)
            ds.test = pick(ins(ds.test), 60) + pick(oos(ds.test), 20)
        else:
            ds.train, ds.val, ds.test = pick(ds.train, n_smoke[0]), pick(ds.val, n_smoke[1]), pick(ds.test, n_smoke[2])
        ds.max_len = min(ds.max_len, 128)
    return ds


def index(ds: tdata.Dataset):
    """allx = train + val + test; rng = positions of each split in allx."""
    allx = ds.train + ds.val + ds.test
    n_tr, n_va = len(ds.train), len(ds.val)
    return allx, {"train": list(range(0, n_tr)), "val": list(range(n_tr, n_tr + n_va)),
                  "test": list(range(n_tr + n_va, len(allx)))}


def task_rows(allx, rng: Dict[str, List[int]], task: str) -> Dict[str, List[int]]:
    return {s: [i for i in rng[s] if task in allx[i].y] for s in rng}


def ys(allx, idx: Sequence[int], task: str):
    """(hard labels, soft labels or None) as tensors."""
    return _targets([allx[i] for i in idx], task)


def texts(ds: tdata.Dataset) -> List[str]:
    return [e.text for e in ds.train + ds.val + ds.test]


def pooled(fc: FeatureCache, depth: int) -> torch.Tensor:
    """Mean-pooled (float32, CPU) states of every message in a FeatureCache."""
    return torch.stack([t.float().mean(0) for t in fc.h[depth]])


def cache_from_states(trunk: Trunk, states: Dict[int, List[torch.Tensor]]) -> FeatureCache:
    """A FeatureCache built from per-message token states {depth: [(L_i, D)]}."""
    fc = FeatureCache.__new__(FeatureCache)
    fc.trunk, fc.depths = trunk, sorted(states)
    fc.h = {d: [t.to("cpu", torch.float16) for t in v] for d, v in states.items()}
    fc.lengths = [t.shape[0] for t in states[fc.depths[0]]]
    fc.seconds = 0.0
    return fc


# ---------------------------------------------------------------------------------------------------
# Branch training (thin wrapper over tarski.train.train_branch)
# ---------------------------------------------------------------------------------------------------

def make(kind: str, split: int, labels: List[str], trunk: Trunk):
    """kind: probe | blocks:D"""
    if kind == "probe":
        return ProbeBranch(split, labels, trunk.hidden)
    return BlockBranch(split, labels, trunk.hidden, int(kind.split(":")[1]), trunk)


def train_eval(branch, cache, allx, sel: Dict[str, List[int]], task: str, kind: str = "probe",
               use_val: bool = True, min_steps: int = 300, seed: int = 0, epochs: Optional[int] = None,
               lr_layers: float = 1e-4, lr_head: Optional[float] = None, eval_caches: Optional[Dict] = None) -> Dict:
    """Train `branch` on sel['train'] of `cache`, temperature on validation (if used and >= 30 rows), test
    metrics. `use_val=False` is the few-label setting: no validation rows at all (last epoch, T = 1).
    `eval_caches` {name: cache} scores the same trained branch on other caches' test rows as well."""
    y_tr, soft_tr = ys(allx, sel["train"], task)
    val = sel["val"] if use_val else []
    y_va, soft_va = ys(allx, val, task) if val else (torch.zeros(0, dtype=torch.long), None)
    y_te, soft_te = ys(allx, sel["test"], task)
    ep = epochs if epochs is not None else (20 if kind == "probe" else 6)
    lh = lr_head if lr_head is not None else (3e-3 if kind == "probe" else 1e-3)
    t0 = time.time()
    info = train_branch(branch, cache, sel["train"], y_tr, soft_tr, val, y_va, epochs=ep, lr_layers=lr_layers,
                        lr_head=lh, seed=seed, min_steps=min_steps)
    T = fit_temperature(predict_logits(branch, cache, val), y_va, soft_va) if len(val) >= 30 else 1.0
    branch.temperature.fill_(T)
    out = {"train_s": round(time.time() - t0, 1), "T": T, "epochs_run": len(info["history"]),
           "steps": len(info["history"]) * -(-len(sel["train"]) // 32)}
    for name, c in {"": cache, **(eval_caches or {})}.items():
        p = torch.softmax(predict_logits(branch, c, sel["test"]) / T, -1).numpy()
        m = evaluate(p, y_te.numpy(), None if soft_te is None else soft_te.numpy())
        out[name or "test"] = m
        out[(name or "test") + "_probs"] = p
    out["y_test"] = y_te.numpy()
    return out


# ---------------------------------------------------------------------------------------------------
# Closed-form / batched heads on pooled features
# ---------------------------------------------------------------------------------------------------

def standardize(X: torch.Tensor, rows: Sequence[int]):
    mu = X[rows].mean(0, keepdim=True)
    sd = X[rows].std(0, keepdim=True).clamp_min(1e-4)
    return (X - mu) / sd


def fit_logreg(Xtr: torch.Tensor, Ttr: torch.Tensor, Xev: List[torch.Tensor], dev, l2s=(1e-4, 1e-3, 1e-2),
               steps: int = 300, lr: float = 1e-2) -> Dict:
    """Multinomial logistic regression, one independent model per L2 value, full-batch Adam, `steps` >= 300.
    Ttr: soft or one-hot targets (n, C). Returns per-grid-point logits (G, n, C) for every matrix in Xev."""
    Xtr, Ttr = Xtr.float().to(dev), Ttr.float().to(dev)
    G, D, C = len(l2s), Xtr.shape[1], Ttr.shape[1]
    W = torch.zeros(G, D, C, device=dev, requires_grad=True)
    b = torch.zeros(G, 1, C, device=dev, requires_grad=True)
    lam = torch.tensor(l2s, device=dev)[:, None, None]
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        z = torch.einsum("nd,gdc->gnc", Xtr, W) + b
        loss = (-(Ttr[None] * F.log_softmax(z, -1)).sum(-1).mean(-1)).sum() + (lam * W ** 2).sum() / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return {"logits": [(torch.einsum("nd,gdc->gnc", X.float().to(dev), W) + b).cpu() for X in Xev],
                "W": W.detach().cpu(), "b": b.detach().cpu()}


def onehot(y, C) -> torch.Tensor:
    return F.one_hot(torch.as_tensor(y, dtype=torch.long), C).float()


class GaussStats:
    """Additive sufficient statistics of a shared-covariance Gaussian classifier (LDA)."""

    def __init__(self, C: int, D: int):
        self.n = torch.zeros(C, dtype=torch.float64)
        self.s = torch.zeros(C, D, dtype=torch.float64)
        self.S = torch.zeros(D, D, dtype=torch.float64)

    def add(self, X: torch.Tensor, P: torch.Tensor, sign: float = 1.0) -> "GaussStats":
        X, P = X.double(), P.double()
        self.n += sign * P.sum(0)
        self.s += sign * P.T @ X
        self.S += sign * (X * P.sum(1, keepdim=True)).T @ X
        return self

    def copy(self) -> "GaussStats":
        o = GaussStats(len(self.n), self.s.shape[1])
        o.n, o.s, o.S = self.n.clone(), self.s.clone(), self.S.clone()
        return o

    def classifier(self, gamma: float):
        """(W, b): logits = X W + b; classes with no mass get -1e30."""
        C, D = self.s.shape
        act = self.n > 1e-9
        mu = self.s[act] / self.n[act, None]
        Sw = self.S - (self.s[act].T / self.n[act]) @ self.s[act]
        cov = Sw / max(1.0, float(self.n.sum()) - int(act.sum()))
        cov = (1 - gamma) * cov + gamma * max(float(cov.diagonal().mean()), 1e-6) * torch.eye(D, dtype=torch.float64)
        P = torch.linalg.inv(cov)
        W = torch.zeros(D, C, dtype=torch.float64)
        b = torch.full((C,), -1e30, dtype=torch.float64)
        W[:, act] = P @ mu.T
        b[act] = -0.5 * ((mu @ P) * mu).sum(-1)
        return W, b


def lda_logits(X: torch.Tensor, clf) -> torch.Tensor:
    return X.double() @ clf[0] + clf[1]


# ---------------------------------------------------------------------------------------------------
# Cold-start selectors (on L2-normalised features of the unlabelled pool)
# ---------------------------------------------------------------------------------------------------

def _kmeans(X: np.ndarray, k: int, seed: int):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    if len(X) > 5000:
        return MiniBatchKMeans(k, n_init=1, random_state=seed, batch_size=4096, max_iter=50).fit(X)
    return KMeans(k, n_init=1, random_state=seed).fit(X)


def select(how: str, X: np.ndarray, budget: int, seed: int = 0) -> np.ndarray:
    """Indices (into X) of `budget` messages to label first. how: random | typiclust | kmedoid | probcover."""
    n = len(X)
    budget = min(budget, n)
    rng = np.random.default_rng(seed)
    if how == "random":
        return rng.choice(n, budget, replace=False)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-9)
    if how in ("typiclust", "kmedoid"):
        km = _kmeans(Xn, budget, seed)
        if how == "typiclust":
            from sklearn.neighbors import NearestNeighbors
            k = min(20, n - 1)
            dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(Xn).kneighbors(Xn)
            score = 1.0 / (dist[:, 1:].mean(1) + 1e-9)
        out = []
        for c in range(budget):
            mem = np.where(km.labels_ == c)[0]
            if len(mem) == 0:
                continue
            if how == "typiclust":
                out.append(mem[np.argmax(score[mem])])
            else:
                out.append(mem[np.argmin(((Xn[mem] - km.cluster_centers_[c]) ** 2).sum(1))])
        rest = np.setdiff1d(np.arange(n), out)
        if len(out) < budget:
            out += list(rng.choice(rest, budget - len(out), replace=False))
        return np.array(out)
    if how == "probcover":
        # greedy maximum coverage of delta-balls (Yehuda et al. 2022). delta: twice the median
        # nearest-neighbour distance (a simple stand-in for the paper's purity-based choice).
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2).fit(Xn)
        delta = 2.0 * float(np.median(nn.kneighbors(Xn)[0][:, 1]))
        G = nn.radius_neighbors_graph(Xn, radius=delta, mode="connectivity").tocsr()   # symmetric, self incl.
        Gc = G.tocsc()
        gain = np.asarray(G.sum(1)).ravel().astype(np.float64)
        covered = np.zeros(n, bool)
        out: List[int] = []
        for _ in range(budget):
            gain_m = gain.copy()
            gain_m[out] = -np.inf
            j = int(np.argmax(gain_m))
            if gain_m[j] <= 0:
                j = int(rng.choice(np.setdiff1d(np.arange(n), out)))
            out.append(j)
            nb = G[j].indices
            new = nb[~covered[nb]]
            if len(new):
                covered[new] = True
                gain -= np.asarray(Gc[:, new].sum(1)).ravel()
        return np.array(out)
    raise ValueError(how)


# ---------------------------------------------------------------------------------------------------
# Learn-then-Test threshold selection (risk control)
# ---------------------------------------------------------------------------------------------------

def ltt_threshold(conf: np.ndarray, flip: np.ndarray, delta: float, eps: float = 0.05,
                  grid: Optional[Sequence[float]] = None) -> Optional[float]:
    """Smallest confidence threshold tau (from strict to loose, fixed-sequence testing) such that the
    risk P(conf >= tau and flip) <= delta is certified at level eps with a binomial tail p-value
    (Angelopoulos et al., Learn then Test, 2021; used for early exits by Jazbec et al., NeurIPS 2024).
    Returns None if even the strictest threshold is not certified."""
    from scipy.stats import binom
    n = len(conf)
    grid = sorted(set(grid if grid is not None else np.linspace(0.999, 0.3, 140)), reverse=True)
    best = None
    for tau in grid:
        k = int(((conf >= tau) & flip).sum())
        p = float(binom.cdf(k, n, delta))
        if p <= eps:
            best = tau
        else:
            break
    return best


# ---------------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------------

def sync(dev) -> None:
    dev = torch.device(dev)
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn: Callable, dev, reps: int = 20, warm: int = 3) -> float:
    for _ in range(warm):
        fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def mean_std(xs) -> Dict[str, float]:
    xs = [float(x) for x in xs]
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)) if len(xs) > 1 else 0.0, "n": len(xs)}
