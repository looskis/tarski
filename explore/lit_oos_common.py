"""Shared helpers for the lit_oos_* scripts (not a runnable experiment).

The lit_oos scripts implement the ranked ideas in research_notes/explore/lit_incremental_oos.md. They all
work on mean-pooled frozen-trunk states, so this module provides:
  - datasets with a small, stratified CPU smoke subsample (CLINC150 keeps its out-of-scope rates);
  - mean-pooled trunk states at several depths from one pass, cached on disk per (texts, depth) under
    ~/.cache/tarski/lit_oos (or $LIT_OOS_CACHE) so that several scripts on one machine share a trunk pass;
  - a batched full-batch logistic-regression probe (an L2 grid fitted in one run, 300 Adam steps, the
    project's minimum), temperature calibration, and Saerens EM prior correction;
  - out-of-scope scores: Mahalanobis (plain and L2-normalised, "Mahalanobis++"), relative Mahalanobis,
    kNN, kernel-PCA reconstruction error, and random Fourier features;
  - OOD metrics (AUROC, AUPR, FPR@95) and a tiny JSON/log writer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from tarski import data as tdata
from tarski.autosplit import pooled_by_depth
from tarski.train import fit_temperature
from tarski.trunk import Trunk

EPS = 1e-12
# pooled-feature cache shared by the lit_oos scripts on one machine (outside results/, so it is never synced
# back with the results); override with LIT_OOS_CACHE, disable per call with use_cache=False
CACHE_DIR = os.environ.get("LIT_OOS_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "tarski", "lit_oos"))


# ---------------------------------------------------------------------------------------------------
# logging / devices
# ---------------------------------------------------------------------------------------------------

class Logger:
    def __init__(self, out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.out_path = out_path
        self.f = open(out_path.replace(".json", ".log"), "a")

    def __call__(self, msg: str):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

    def dump(self, obj):
        with open(self.out_path, "w") as f:
            json.dump(obj, f, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


def pick_device(smoke: bool) -> torch.device:
    if smoke:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_trunk(smoke: bool) -> Trunk:
    return Trunk(device="cpu" if smoke else None)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------

def _stratified(rows: List, per: int, key, rng: random.Random) -> List:
    groups: Dict = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = []
    for g in sorted(groups):
        xs = list(groups[g])
        rng.shuffle(xs)
        out.extend(xs[:per])
    rng.shuffle(out)
    return out


def load_ds(name: str, smoke: bool, seed: int = 0) -> tdata.Dataset:
    """A tarski dataset; in smoke mode a small stratified subsample that keeps every class (and CLINC's
    out-of-scope proportions: about 2% train, 6% val, 18% test)."""
    ds = tdata.load(name)
    if not smoke:
        return ds
    rng = random.Random(seed)
    if name == "clinc150":
        key = lambda e: e.y["intent"]
        ins = lambda xs: [e for e in xs if e.y["oos"] == 0]
        oos = lambda xs: [e for e in xs if e.y["oos"] == 1]
        ds.train = _stratified(ins(ds.train), 6, key, rng) + oos(ds.train)[:20]
        ds.val = _stratified(ins(ds.val), 2, key, rng) + oos(ds.val)[:20]
        ds.test = _stratified(ins(ds.test), 3, key, rng) + oos(ds.test)[:100]
    else:
        key = lambda e: e.y["intent"]
        ds.train = _stratified(ds.train, 8, key, rng)
        ds.val = _stratified(ds.val, 3, key, rng)
        ds.test = _stratified(ds.test, 4, key, rng)
    return ds


def split_index(ds: tdata.Dataset) -> Tuple[List, Dict[str, np.ndarray]]:
    allx = ds.train + ds.val + ds.test
    n_tr, n_va = len(ds.train), len(ds.val)
    idx = {"train": np.arange(0, n_tr), "val": np.arange(n_tr, n_tr + n_va),
           "test": np.arange(n_tr + n_va, len(allx))}
    return allx, idx


def humanize(label: str) -> str:
    return label.replace("_", " ")


# ---------------------------------------------------------------------------------------------------
# trunk features (mean-pooled, disk-cached)
# ---------------------------------------------------------------------------------------------------

def _texts_key(trunk: Trunk, texts: Sequence[str], max_len: int) -> str:
    h = hashlib.sha1()
    h.update(trunk.base.encode())
    h.update(str(max_len).encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def pooled(trunk: Trunk, texts: Sequence[str], depths: Sequence[int], max_len: int,
           use_cache: bool = True, log=print) -> Dict[int, torch.Tensor]:
    """Mean-pooled trunk states at each depth (float32, CPU). Cached per (texts, depth) in fp16 under
    CACHE_DIR so the lit_oos scripts share one trunk pass per machine."""
    depths = sorted(set(int(d) for d in depths))
    key = _texts_key(trunk, texts, max_len)
    out, missing = {}, []
    for d in depths:
        p = os.path.join(CACHE_DIR, f"{key}_d{d}.pt")
        if use_cache and os.path.exists(p):
            out[d] = torch.load(p).float()
        else:
            missing.append(d)
    if missing:
        t0 = time.time()
        feats = pooled_by_depth(trunk, list(texts), missing, max_len)
        log(f"  trunk pass: {len(texts)} messages at depths {missing} in {time.time() - t0:.1f}s")
        for d in missing:
            out[d] = feats[d].float()
            if use_cache:
                os.makedirs(CACHE_DIR, exist_ok=True)
                torch.save(feats[d].half(), os.path.join(CACHE_DIR, f"{key}_d{d}.pt"))
    return out


def zscore(X: torch.Tensor, rows) -> torch.Tensor:
    mu = X[rows].mean(0, keepdim=True)
    sd = X[rows].std(0, keepdim=True).clamp_min(1e-4)
    return (X - mu) / sd


def l2n(X: torch.Tensor) -> torch.Tensor:
    return F.normalize(X.float(), dim=-1)


# ---------------------------------------------------------------------------------------------------
# probes and calibration
# ---------------------------------------------------------------------------------------------------

def fit_logreg(Xtr: torch.Tensor, ytr: np.ndarray, C: int, Xev: List[torch.Tensor], dev,
               l2s=(1e-4, 1e-3, 1e-2), steps: int = 300, lr: float = 1e-2, seed: int = 0,
               weight: Optional[np.ndarray] = None) -> Dict:
    """Batched multinomial logistic regression: one independent model per L2 value, full-batch Adam,
    `steps` >= 300 optimiser steps. Returns per-grid-point logits for every matrix in `Xev` ((G, n, C))."""
    steps = max(300, steps)
    torch.manual_seed(seed)
    X = Xtr.float().to(dev)
    y = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    w = None if weight is None else torch.as_tensor(weight, dtype=torch.float32, device=dev)
    G, D = len(l2s), X.shape[1]
    W = torch.zeros(G, D, C, device=dev, requires_grad=True)
    b = torch.zeros(G, 1, C, device=dev, requires_grad=True)
    lam = torch.tensor(l2s, device=dev, dtype=torch.float32)[:, None, None]
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        z = torch.einsum("nd,gdc->gnc", X, W) + b
        nll = F.cross_entropy(z.reshape(-1, C), y.repeat(G), reduction="none").reshape(G, -1)
        nll = (nll * w).sum(-1) / w.sum() if w is not None else nll.mean(-1)
        loss = nll.sum() + (lam * W ** 2).sum() / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        outs = [(torch.einsum("nd,gdc->gnc", Z.float().to(dev), W) + b).cpu() for Z in Xev]
    return {"W": W.detach().cpu(), "b": b.detach().cpu(), "logits": outs, "l2s": list(l2s)}


class Probe:
    """A calibrated linear probe: L2 chosen on validation accuracy, one temperature on validation NLL."""

    def __init__(self, X: torch.Tensor, y: np.ndarray, tr, va, C: int, dev, steps: int = 300, seed: int = 0,
                 va_T=None, weight=None):
        r = fit_logreg(X[tr], y[tr], C, [X[va]], dev, steps=steps, seed=seed, weight=weight)
        zva = r["logits"][0]
        accs = (zva.argmax(-1).numpy() == y[va][None]).mean(-1)
        k = int(np.argmax(accs))
        self.W, self.b, self.l2 = r["W"][k], r["b"][k], r["l2s"][k]
        self.C = C
        rows_T = va if va_T is None else va_T
        zt = self.logits(X[rows_T])
        self.T = fit_temperature(zt, torch.as_tensor(y[rows_T])) if len(rows_T) >= 30 else 1.0
        self.val_acc = float(accs[k])

    def logits(self, X: torch.Tensor) -> torch.Tensor:
        return X.float() @ self.W + self.b[0]

    def probs(self, X: torch.Tensor) -> np.ndarray:
        return torch.softmax(self.logits(X) / self.T, -1).numpy().astype(np.float64)


def saerens_em(probs: np.ndarray, prior_src: np.ndarray, iters: int = 200, tol: float = 1e-8):
    """Saerens, Latinne & Decaestecker (2002): EM estimate of the target prior from calibrated posteriors,
    and the prior-corrected posteriors. No target labels."""
    ps = np.clip(prior_src, 1e-12, None)
    pt = ps.copy()
    for _ in range(iters):
        adj = probs * (pt / ps)
        adj /= np.clip(adj.sum(-1, keepdims=True), 1e-300, None)
        new = adj.mean(0)
        done = np.abs(new - pt).max() < tol
        pt = new
        if done:
            break
    adj = probs * (pt / ps)
    adj /= np.clip(adj.sum(-1, keepdims=True), 1e-300, None)
    return adj, pt


def class_prior(y: np.ndarray, C: int) -> np.ndarray:
    c = np.bincount(y, minlength=C).astype(np.float64)
    return c / max(1.0, c.sum())


# ---------------------------------------------------------------------------------------------------
# OOD scores (higher = more out-of-scope)
# ---------------------------------------------------------------------------------------------------

def ood_metrics(score: np.ndarray, is_oos: np.ndarray) -> Dict[str, float]:
    auroc = float(roc_auc_score(is_oos, score))
    aupr = float(average_precision_score(is_oos, score))
    fpr, tpr, _ = roc_curve(is_oos, score)
    fpr95 = float(fpr[np.searchsorted(tpr, 0.95, side="left")]) if (tpr >= 0.95).any() else 1.0
    return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}


class Gauss:
    """Class means and one shared (shrunk) covariance, float64 on the CPU. Gives LDA logits and the
    minimum Mahalanobis distance. `shrink` mixes the covariance with a scaled identity."""

    def __init__(self, X: torch.Tensor, y: np.ndarray, C: int, shrink: float = 0.1):
        X = X.double().cpu()
        y = torch.as_tensor(y, dtype=torch.long)
        n = torch.bincount(y, minlength=C).double()
        s = torch.zeros(C, X.shape[1], dtype=torch.float64).index_add_(0, y, X)
        self.active = n > 0
        self.mu = s / n.clamp_min(1)[:, None]
        Xc = X - self.mu[y]
        self.cov = Xc.T @ Xc / max(1, len(X) - int(self.active.sum()))
        self._set_shrink(shrink)

    def _set_shrink(self, shrink: float):
        D = self.cov.shape[0]
        cov = (1 - shrink) * self.cov + shrink * self.cov.diagonal().mean() * torch.eye(D, dtype=torch.float64)
        self.shrink = shrink
        self.P = torch.linalg.inv(cov)
        self.PmuT = self.P @ self.mu.T
        self.mPm = (self.mu * self.PmuT.T).sum(-1)

    def reshrink(self, shrink: float) -> "Gauss":
        """The same statistics with another shrinkage (no new pass over the data)."""
        out = Gauss.__new__(Gauss)
        out.active, out.mu, out.cov = self.active, self.mu, self.cov
        out._set_shrink(shrink)
        return out

    def d2(self, X: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        out = []
        for s in range(0, len(X), chunk):
            x = X[s:s + chunk].double().cpu()
            xPx = ((x @ self.P) * x).sum(-1, keepdim=True)
            d = xPx - 2 * x @ self.PmuT + self.mPm[None]
            d[:, ~self.active] = float("inf")
            out.append(d)
        return torch.cat(out)

    def min_dist(self, X: torch.Tensor) -> np.ndarray:
        return self.d2(X).min(-1).values.clamp_min(0).numpy()

    def lda_logits(self, X: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.d2(X)


def maha_background(Xtr: torch.Tensor, shrink: float = 0.1):
    """One Gaussian over all in-scope training points (for relative Mahalanobis, Ren et al. 2021)."""
    return Gauss(Xtr, np.zeros(len(Xtr), dtype=int), 1, shrink)


@torch.no_grad()
def knn_dist(train: torch.Tensor, test: torch.Tensor, ks: Sequence[int], dev, chunk: int = 2048) -> Dict[int, np.ndarray]:
    """Distance to the k-th nearest training point on L2-normalised features (Sun et al., ICML 2022)."""
    tr = l2n(train).to(dev)
    out = {k: np.zeros(len(test)) for k in ks}
    for s in range(0, len(test), chunk):
        te = l2n(test[s:s + chunk]).to(dev)
        top = torch.topk(te @ tr.T, max(ks), dim=-1).values
        for k in ks:
            out[k][s:s + chunk] = torch.sqrt((2 - 2 * top[:, k - 1]).clamp_min(0)).cpu().numpy()
    return out


class RFF:
    """Random Fourier features of the RBF kernel exp(-gamma ||x - x'||^2) (Rahimi & Recht 2007). Fixed and
    seeded: it is part of the trunk, not of any user's route, so statistics built on it stay additive."""

    def __init__(self, d_in: int, D: int, gamma: float, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(d_in, D, generator=g, dtype=torch.float32) * math.sqrt(2 * gamma)
        self.b = torch.rand(D, generator=g, dtype=torch.float32) * 2 * math.pi
        self.D = D

    def __call__(self, X: torch.Tensor, dev=None, chunk: int = 8192) -> torch.Tensor:
        dev = dev or torch.device("cpu")
        W, b = self.W.to(dev), self.b.to(dev)
        out = []
        for s in range(0, len(X), chunk):
            out.append((math.sqrt(2.0 / self.D) * torch.cos(X[s:s + chunk].float().to(dev) @ W + b)).cpu())
        return torch.cat(out)


def median_sqdist(X: torch.Tensor, n: int = 2000, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)[:n]
    Z = X[idx].float()
    d = torch.cdist(Z, Z).pow(2)
    return float(d[torch.triu(torch.ones_like(d, dtype=torch.bool), 1)].median())


def pca_recon_error(Xtr: torch.Tensor, Xev: torch.Tensor, ratios: Sequence[float]) -> Dict[float, np.ndarray]:
    """Kernel-PCA-style reconstruction error (Fang et al., NeurIPS 2024): PCA on (explicitly mapped)
    training features, error = squared norm of the residual outside the top components that explain
    `ratio` of the variance."""
    mu = Xtr.mean(0, keepdim=True)
    A = (Xtr - mu).double()
    cov = A.T @ A / len(A)
    evals, evecs = torch.linalg.eigh(cov)
    evals, evecs = evals.flip(0), evecs.flip(1)
    cum = torch.cumsum(evals.clamp_min(0), 0) / evals.clamp_min(0).sum()
    Z = (Xev - mu).double()
    tot = (Z ** 2).sum(-1)
    out = {}
    for r in ratios:
        q = int(torch.searchsorted(cum, torch.tensor(r, dtype=torch.float64))) + 1
        proj = Z @ evecs[:, :q]
        out[r] = (tot - (proj ** 2).sum(-1)).clamp_min(0).numpy()
    return out


# ---------------------------------------------------------------------------------------------------
# thresholds and binary calibration of a score
# ---------------------------------------------------------------------------------------------------

def f1_at(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    tp = int(((pred == 1) & (y == 1)).sum())
    p = tp / max(1, int(pred.sum()))
    r = tp / max(1, int(y.sum()))
    return {"precision": p, "recall": r, "f1": 0.0 if p + r == 0 else 2 * p * r / (p + r)}


def score_to_prob(score_val: np.ndarray, oos_val: np.ndarray):
    """1-D logistic calibration of an OOS score on validation (uses the validation OOS labels)."""
    from sklearn.linear_model import LogisticRegression
    mu, sd = score_val.mean(), score_val.std() + 1e-9
    lr = LogisticRegression(C=10.0).fit(((score_val - mu) / sd)[:, None], oos_val)
    return lambda s: lr.predict_proba(((s - mu) / sd)[:, None])[:, 1]


def binary_em_f1(p_oos_test: np.ndarray, prior_src_oos: float, y_test: np.ndarray) -> Dict[str, float]:
    """Saerens EM on a binary OOS posterior, threshold 0.5, F1 on the test pool."""
    P = np.stack([1 - p_oos_test, p_oos_test], 1)
    adj, pt = saerens_em(P, np.array([1 - prior_src_oos, prior_src_oos]))
    out = f1_at((adj[:, 1] > 0.5).astype(int), y_test)
    out["est_oos_rate"] = float(pt[1])
    return out


def typo_noise(text: str, p: float, rng: random.Random) -> str:
    """Keyboard-style character noise: each letter is deleted, doubled, substituted or swapped with
    probability p (one operation per affected position)."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < p:
            op = rng.randrange(4)
            if op == 0:
                pass                                   # delete
            elif op == 1:
                out.extend([c, c])                     # double
            elif op == 2:
                out.append(rng.choice(letters))        # substitute
            elif i + 1 < len(chars):
                out.extend([chars[i + 1], c])          # swap with next
                i += 1
            else:
                out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)
