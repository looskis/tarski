"""Shared helpers for the geometry-lens scripts (not a runnable experiment).

Datasets with a CPU smoke subsample, mean-pooled trunk states at several depths from one pass, and a
GPU-friendly multinomial logistic regression whose L2 grid is fitted in one batched run (every grid point
is an independent model; at least 300 optimiser steps, the tarski rule).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from tarski import data as tdata
from tarski.autosplit import pooled_by_depth
from tarski.trunk import Trunk


def load_dataset(name: str, smoke: bool, seed: int = 0) -> tdata.Dataset:
    ds = tdata.load(name)
    if smoke:
        rng = np.random.default_rng(seed)
        pick = lambda xs, k: [xs[i] for i in sorted(rng.choice(len(xs), min(k, len(xs)), replace=False))]
        if name == "clinc150":
            ins = lambda xs: [e for e in xs if e.y["oos"] == 0]
            oos = lambda xs: [e for e in xs if e.y["oos"] == 1]
            ds.train = pick(ins(ds.train), 300) + pick(oos(ds.train), 20)
            ds.val = pick(ins(ds.val), 80) + pick(oos(ds.val), 20)
            ds.test = pick(ins(ds.test), 80) + pick(oos(ds.test), 20)
        else:
            ds.train, ds.val, ds.test = pick(ds.train, 160), pick(ds.val, 60), pick(ds.test, 80)
        ds.max_len = min(ds.max_len, 128)
    return ds


def splits(ds: tdata.Dataset):
    allx = ds.train + ds.val + ds.test
    n_tr, n_va = len(ds.train), len(ds.val)
    return allx, n_tr, n_va


def task_split(allx, task: str, n_tr: int, n_va: int):
    idx = [i for i, e in enumerate(allx) if task in e.y]
    return ([i for i in idx if i < n_tr], [i for i in idx if n_tr <= i < n_tr + n_va],
            [i for i in idx if i >= n_tr + n_va])


def targets(allx, idx: Sequence[int], task: str, C: int) -> torch.Tensor:
    """Soft targets when every example has them (typed-decisions), else one-hot."""
    if all(task in allx[i].soft for i in idx):
        return torch.tensor(np.stack([allx[i].soft[task] for i in idx]), dtype=torch.float32)
    return F.one_hot(torch.tensor([allx[i].y[task] for i in idx]), C).float()


def labels(allx, idx: Sequence[int], task: str) -> np.ndarray:
    return np.array([allx[i].y[task] for i in idx])


def pooled(trunk: Trunk, ds: tdata.Dataset, depths: Sequence[int]) -> Dict[int, torch.Tensor]:
    allx, _, _ = splits(ds)
    return pooled_by_depth(trunk, [e.text for e in allx], list(depths), ds.max_len)


def standardize(F_: torch.Tensor, tr: Sequence[int]) -> torch.Tensor:
    mu = F_[tr].mean(0, keepdim=True)
    sd = F_[tr].std(0, keepdim=True).clamp_min(1e-4)
    return (F_ - mu) / sd


def fit_logreg(Xtr: torch.Tensor, Ttr: torch.Tensor, Xev: List[torch.Tensor], dev, l2s=(1e-4, 1e-3, 1e-2),
               steps: int = 300, lr: float = 1e-2, prox=None, penalty=None) -> Dict:
    """Batched multinomial logistic regression: one independent model per L2 value, full-batch Adam.
    `penalty(W)` (optional) adds a scalar to the loss (W is (G, D, C); it must be a sum of per-grid-point
    terms so the grid points stay independent); `prox(W, lr)` (optional) is applied after every step.
    Returns per-grid-point logits for every matrix in `Xev` (each (G, n, C))."""
    Xtr, Ttr = Xtr.float().to(dev), Ttr.float().to(dev)
    G, D, C = len(l2s), Xtr.shape[1], Ttr.shape[1]
    W = torch.zeros(G, D, C, device=dev, requires_grad=True)
    b = torch.zeros(G, 1, C, device=dev, requires_grad=True)
    lam = torch.tensor(l2s, device=dev)[:, None, None]
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        z = torch.einsum("nd,gdc->gnc", Xtr, W) + b
        loss = (-(Ttr[None] * F.log_softmax(z, -1)).sum(-1).mean(-1)).sum() + (lam * W ** 2).sum() / 2
        if penalty is not None:
            loss = loss + penalty(W)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if prox is not None:
            with torch.no_grad():
                prox(W, lr)
    with torch.no_grad():
        outs = [(torch.einsum("nd,gdc->gnc", X.float().to(dev), W) + b).cpu() for X in Xev]
    return {"W": W.detach().cpu(), "b": b.detach().cpu(), "logits": outs}


def pick_by_val(logits_val: torch.Tensor, yva: np.ndarray) -> int:
    accs = (logits_val.argmax(-1).numpy() == yva[None]).mean(-1)
    return int(np.argmax(accs))


def acc(z: torch.Tensor, y: np.ndarray) -> float:
    return float((z.argmax(-1).numpy() == y).mean())


def device(smoke: bool) -> torch.device:
    if smoke:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Logger:
    def __init__(self, out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.f = open(out_path.replace(".json", ".log"), "a")

    def __call__(self, msg: str):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def warmup_cosine(opt, total: int, warm_frac: float = 0.1):
    """Linear warm-up then cosine decay to 0 (a LambdaLR, safe for any number of steps)."""
    import math
    warm = max(1, int(warm_frac * total))
    f = lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def message_groups(allx, tasks) -> Dict[str, str]:
    """task -> group id; tasks labelled on exactly the same messages share a group (typed-decisions:
    workflows; CLINC150: one group)."""
    sig = {}
    for t in tasks:
        sig.setdefault(tuple(i for i, e in enumerate(allx) if t in e.y), []).append(t)
    return {t: f"g{k}:{ts[0].split('.')[0]}" for k, ts in enumerate(sig.values()) for t in ts}
