"""Shared helpers for the far-field exploration scripts (explore/farfield_*.py).

Nothing here touches `tarski/`; these are conveniences on top of it: repo-root import path, offline HF
defaults, logging to stdout + file, full-batch linear probes on pooled trunk states, OOS metrics, and
deterministic subsampling for `--smoke` modes.
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

EPS = 1e-12
LAMBDA_DIR = os.path.join(REPO, "results", "tarski", "lambda")


class Logger:
    def __init__(self, path: Optional[str]):
        self.f = open(path, "a") if path else None

    def __call__(self, msg: str):
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n")
            self.f.flush()


def out_paths(args, name: str):
    """Default output path (smoke runs get a `_smoke` suffix) and its log twin; creates the directory."""
    if not args.out:
        args.out = f"results/tarski/explore_{name}{'_smoke' if args.smoke else ''}.json"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    return args.out, args.out.replace(".json", ".log")


def dump(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


# ---------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------

def subsample(rows: List, n: int, seed: int, key=None) -> List:
    """Deterministic subsample; with `key`, roughly n rows spread evenly over key(row) groups."""
    rng = random.Random(seed)
    rows = list(rows)
    if key is None:
        rng.shuffle(rows)
        return rows[:n]
    groups: Dict = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    per = max(1, n // len(groups))
    out = []
    for g in groups.values():
        rng.shuffle(g)
        out.extend(g[:per])
    rng.shuffle(out)
    return out


def split_indices(n_train: int, n_val: int, n_all: int) -> Dict[str, np.ndarray]:
    return {"train": np.arange(0, n_train), "val": np.arange(n_train, n_train + n_val), "test": np.arange(n_train + n_val, n_all)}


def task_rows(allx, task: str, rows: Iterable[int]) -> List[int]:
    return [i for i in rows if task in allx[i].y]


def labels_of(allx, task: str, rows: Sequence[int]) -> torch.Tensor:
    return torch.tensor([allx[i].y[task] for i in rows])


def soft_of(allx, task: str, rows: Sequence[int]) -> Optional[torch.Tensor]:
    if all(task in allx[i].soft for i in rows) and len(rows):
        return torch.tensor(np.stack([allx[i].soft[task] for i in rows])).float()
    return None


# ---------------------------------------------------------------------------------------------------
# probes on pooled features
# ---------------------------------------------------------------------------------------------------

class Standardiser:
    def __init__(self, x: torch.Tensor):
        self.mu, self.sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-4)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sd


def fit_linear(x: torch.Tensor, y: torch.Tensor, n_labels: int, device, steps: int = 300, lr: float = 1e-2,
               wd: float = 1e-4, seed: int = 0, soft: Optional[torch.Tensor] = None) -> torch.nn.Linear:
    """Full-batch linear probe (the autosplit recipe); `steps` optimiser steps, so never under-trained."""
    torch.manual_seed(seed)
    lin = torch.nn.Linear(x.shape[1], n_labels).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    x, y = x.to(device), y.to(device)
    soft = soft.to(device) if soft is not None else None
    for _ in range(steps):
        z = lin(x)
        loss = F.cross_entropy(z, y) if soft is None else -(soft * F.log_softmax(z, -1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lin.eval()


@torch.no_grad()
def logits_of(lin: torch.nn.Module, x: torch.Tensor, device) -> torch.Tensor:
    return lin(x.to(device)).float().cpu()


def softmax_np(z: torch.Tensor, t: float = 1.0) -> np.ndarray:
    return torch.softmax(z / t, -1).numpy().astype(np.float64)


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(p + EPS)).sum(-1)


def js_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * ((p * (np.log(p + EPS) - np.log(m + EPS))).sum(-1) + (q * (np.log(q + EPS) - np.log(m + EPS))).sum(-1))


# ---------------------------------------------------------------------------------------------------
# OOS metrics
# ---------------------------------------------------------------------------------------------------

def ood_metrics(score: np.ndarray, is_oos: np.ndarray) -> Dict[str, float]:
    """Higher score = more out-of-scope; OOS is the positive class."""
    fpr, tpr, _ = roc_curve(is_oos, score)
    fpr95 = float(fpr[np.searchsorted(tpr, 0.95, side="left")]) if (tpr >= 0.95).any() else 1.0
    return {"auroc": float(roc_auc_score(is_oos, score)), "aupr": float(average_precision_score(is_oos, score)), "fpr95": fpr95}


def best_threshold_acc(score_val, oos_val, score_test, oos_test) -> Dict[str, float]:
    cands = np.unique(np.quantile(score_val, np.linspace(0, 1, 401)))
    accs = [((score_val > c).astype(int) == oos_val).mean() for c in cands]
    thr = float(cands[int(np.argmax(accs))])
    pred = (score_test > thr).astype(int)
    tp = int(((pred == 1) & (oos_test == 1)).sum())
    return {"threshold": thr, "acc": float((pred == oos_test).mean()),
            "oos_recall": tp / max(1, int(oos_test.sum())), "oos_precision": tp / max(1, int(pred.sum()))}


@torch.no_grad()
def knn_distance(train: torch.Tensor, test: torch.Tensor, k: int, device, chunk: int = 1024) -> np.ndarray:
    """Euclidean distance to the k-th nearest training row after L2 normalisation."""
    tr = F.normalize(train.to(device), dim=-1)
    out = np.zeros(len(test))
    for s in range(0, len(test), chunk):
        te = F.normalize(test[s:s + chunk].to(device), dim=-1)
        top = torch.topk(te @ tr.T, k, dim=-1).values[:, k - 1]
        out[s:s + chunk] = torch.sqrt((2 - 2 * top).clamp_min(0)).cpu().numpy()
    return out


def load_sweeps(dataset_prefix: str) -> Dict[str, Dict]:
    """All results/tarski/lambda/sweep_<prefix>*.json files, keyed by file stem (one per seed)."""
    out = {}
    if os.path.isdir(LAMBDA_DIR):
        for name in sorted(os.listdir(LAMBDA_DIR)):
            if name.startswith(f"sweep_{dataset_prefix}") and name.endswith(".json"):
                out[name[:-5]] = json.load(open(os.path.join(LAMBDA_DIR, name)))
    return out


# ---------------------------------------------------------------------------------------------------
# blocks/probe branches on a FeatureCache: the per-task body of tarski.train.fit, for a prebuilt branch
# ---------------------------------------------------------------------------------------------------

def train_and_eval(branch, cache, allx, rows: Dict[str, Iterable[int]], task: str, epochs: int = 8, min_steps: int = 300,
                   seed: int = 0, log=lambda s: None, lr_layers: float = 1e-4, lr_head: float = 1e-3, bs: int = 32,
                   min_val: int = 50) -> Dict:
    """Train `branch` (any tarski Branch whose `split` is cached) on the task's labelled rows, calibrate a
    temperature on validation (>= 30 rows, the harness rule) and return test metrics plus test logits."""
    import time as _time
    from tarski import train as T

    sel = {s: task_rows(allx, task, r) for s, r in rows.items()}
    y = {s: labels_of(allx, task, sel[s]) for s in sel}
    soft = {s: soft_of(allx, task, sel[s]) for s in sel}
    t0 = _time.time()
    info = T.train_branch(branch, cache, sel["train"], y["train"], soft["train"], sel["val"], y["val"], epochs=epochs, bs=bs,
                          lr_layers=lr_layers, lr_head=lr_head, seed=seed, log=log, min_val=min_val, min_steps=min_steps)
    temp = T.fit_temperature(T.predict_logits(branch, cache, sel["val"]), y["val"], soft["val"]) if len(sel["val"]) >= 30 else 1.0
    branch.temperature.fill_(temp)
    z = T.predict_logits(branch, cache, sel["test"])
    probs = torch.softmax(z / temp, -1).numpy()
    m = T.evaluate(probs, y["test"].numpy(), None if soft["test"] is None else soft["test"].numpy())
    m.update({"val_acc": info["val_acc"], "temperature": temp, "train_s": round(_time.time() - t0, 1),
              "epochs_run": len(info["history"]), "steps_run": len(info["history"]) * ((len(sel["train"]) + bs - 1) // bs),
              "params": sum(p.numel() for p in branch.parameters()),
              "trainable_params": sum(p.numel() for p in branch.parameters() if p.requires_grad)})
    return {"metrics": m, "test_logits": z, "test_rows": sel["test"], "val_rows": sel["val"], "train_rows": sel["train"]}


def smoke_subset(ds, n_train: int, n_val: int, n_test: int, seed: int, key=None):
    """Replace a Dataset's splits by deterministic subsamples (for --smoke)."""
    import copy as _copy
    d = _copy.copy(ds)
    d.train = subsample(ds.train, n_train, seed, key)
    d.val = subsample(ds.val, n_val, seed + 1, key)
    d.test = subsample(ds.test, n_test, seed + 2, key)
    return d
