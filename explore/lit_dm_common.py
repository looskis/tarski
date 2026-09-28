"""Shared helpers for the lit_dm_* experiments (research_notes/explore/lit_decision_models.md).

Scripts that download new models or datasets must set HF_HUB_OFFLINE / HF_DATASETS_OFFLINE to "0" *before*
importing this module (it imports tarski, which imports transformers), because the GPU job runner exports
both as "1".
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tarski.branches import BlockBranch, ProbeBranch, mean_pool  # noqa: E402
from tarski.data import Dataset, Example  # noqa: E402
from tarski.train import FeatureCache, ece_score, evaluate, fit_temperature, predict_logits, train_branch  # noqa: E402
from tarski.trunk import Trunk  # noqa: E402

SMOKE_BASE = "jhu-clsp/ettin-encoder-17m"   # 7 layers, hidden 256, ModernBERT architecture: CPU smoke tests


# ---------------------------------------------------------------------------------------------------
# Logging and output
# ---------------------------------------------------------------------------------------------------

class Log:
    def __init__(self, out: Optional[str]):
        self.f = None
        if out:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            self.f = open(out.replace(".json", ".log"), "a")
        self.t0 = time.time()

    def __call__(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, torch.Tensor):
        return _jsonable(x.detach().cpu().tolist())
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def dump(obj, path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def load_json(path: Optional[str]) -> Dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_trunk(base: Optional[str] = None, device: Optional[str] = None, cls=None) -> Trunk:
    """tarski Trunk, cast to float32: some checkpoints (gte-modernbert-base) are stored in fp16 and transformers
    5 loads them in that dtype, which breaks fp32 branch layers copied from them. The trunk pass itself still
    runs under tarski's autocast on GPU."""
    cls = cls or Trunk
    t = cls(base, device=device) if base else cls(device=device)
    if next(t.model.parameters()).dtype != torch.float32:
        t.model.float()
    return t


# ---------------------------------------------------------------------------------------------------
# Label text
# ---------------------------------------------------------------------------------------------------

_ABBREV = {
    "pto": "paid time off", "w2": "W-2 tax form", "apr": "annual percentage rate", "mpg": "miles per gallon",
    "gas": "gas", "oil": "oil", "ingredients": "ingredients", "calories": "calories", "insurance": "insurance",
    "todo": "to-do", "cancel": "cancel", "min": "minimum", "fun": "fun", "rollover": "rollover",
    "401k": "401k retirement", "atm": "ATM", "pin": "PIN", "id": "ID", "uk": "UK", "eu": "EU",
    "sms": "SMS", "whatsapp": "WhatsApp", "visa": "Visa", "mastercard": "Mastercard",
}


def humanize(label: str) -> str:
    """'card_arrival' -> 'card arrival'; 'pto_request' -> 'paid time off request'."""
    words = label.replace("-", " ").replace("_", " ").split()
    out = [_ABBREV.get(w.lower(), w.lower()) for w in words]
    return " ".join(out).strip() or label


def label_texts(labels: Sequence[str], template: str = "{name}") -> List[str]:
    return [template.format(name=humanize(l)) for l in labels]


# ---------------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------------

def load_dataset_by_name(name: str) -> Dataset:
    from tarski import data
    if name == "typed-cs":
        ds = data.load("typed-decisions")
        return restrict_tasks(ds, [t for t in ds.tasks if t.startswith("customer_service.")], "typed-cs")
    return data.load(name)


def restrict_tasks(ds: Dataset, tasks: Sequence[str], name: Optional[str] = None) -> Dataset:
    keep = set(tasks)

    def f(xs):
        out = []
        for e in xs:
            y = {t: v for t, v in e.y.items() if t in keep}
            if y:
                out.append(Example(e.text, y, {t: v for t, v in e.soft.items() if t in keep}))
        return out

    return Dataset(name or ds.name, {t: ds.tasks[t] for t in tasks}, f(ds.train), f(ds.val), f(ds.test), ds.max_len)


def subsample(ds: Dataset, n_train: int, n_val: int, n_test: int, seed: int = 0,
              classes: Optional[Dict[str, Sequence[int]]] = None) -> Dataset:
    """Small random subset for smoke tests; with `classes`, only messages whose label for each named task is
    in the given set."""
    rng = random.Random(seed)

    def f(xs, n):
        xs = [e for e in xs if not classes or all(e.y.get(t) in set(c) for t, c in classes.items())]
        xs = list(xs)
        rng.shuffle(xs)
        return xs[:n]

    return Dataset(ds.name, ds.tasks, f(ds.train, n_train), f(ds.val, n_val), f(ds.test, n_test), ds.max_len)


def kshot(y: np.ndarray, pool: Sequence[int], k: Optional[int], seed: int) -> List[int]:
    """k examples per class from `pool` (indices into y); k None = all."""
    pool = list(pool)
    if k is None:
        return pool
    rng = np.random.default_rng(seed)
    by = {}
    for i in pool:
        by.setdefault(int(y[i]), []).append(i)
    out = []
    for c in sorted(by):
        idx = by[c]
        take = rng.choice(len(idx), size=min(k, len(idx)), replace=False)
        out += [idx[j] for j in sorted(take)]
    return out


# ---------------------------------------------------------------------------------------------------
# Pooling and scoring on cached states
# ---------------------------------------------------------------------------------------------------

@torch.no_grad()
def pooled(cache: FeatureCache, depth: int, idx: Sequence[int], mode: str = "mean", bs: int = 256) -> torch.Tensor:
    """Pooled trunk states [n, hidden] (float32, CPU) for cached texts `idx` at `depth`."""
    out = []
    for s in range(0, len(idx), bs):
        sel = list(idx[s:s + bs])
        if mode == "mean":
            v = torch.stack([cache.h[depth][i].float().mean(0) for i in sel])
        elif mode == "cls":
            v = torch.stack([cache.h[depth][i][0].float() for i in sel])
        elif mode == "last":
            v = torch.stack([cache.h[depth][i][-1].float() for i in sel])
        else:
            raise ValueError(mode)
        out.append(v)
    return torch.cat(out) if out else torch.zeros(0, cache.trunk.hidden)


def cosine_logits(q: torch.Tensor, c: torch.Tensor, center: Optional[torch.Tensor] = None) -> torch.Tensor:
    if center is not None:
        q, c = q - center, c - center
    return F.normalize(q, dim=-1) @ F.normalize(c, dim=-1).T


def oos_metrics(conf_in: np.ndarray, conf_oos: np.ndarray, thr: Optional[float] = None) -> Dict[str, float]:
    """In-scope vs out-of-scope separation from a confidence (higher = more in-scope).

    auroc: in-scope as positive. fpr95: share of oos kept when the threshold keeps 95% of in-scope (test).
    With `thr` (e.g. chosen on in-scope validation data): oos recall and in-scope retention at that threshold."""
    from sklearn.metrics import roc_auc_score
    conf_in, conf_oos = np.asarray(conf_in, float), np.asarray(conf_oos, float)
    if len(conf_in) == 0 or len(conf_oos) == 0:
        return {}
    # a diverged arm can emit NaN scores; count them and treat them as least in-scope instead of crashing
    n_nan = int(np.isnan(conf_in).sum() + np.isnan(conf_oos).sum())
    conf_in, conf_oos = np.nan_to_num(conf_in, nan=-1e9), np.nan_to_num(conf_oos, nan=-1e9)
    y = np.r_[np.ones(len(conf_in)), np.zeros(len(conf_oos))]
    s = np.r_[conf_in, conf_oos]
    t95 = np.quantile(conf_in, 0.05)
    out = {"auroc": float(roc_auc_score(y, s)), "fpr95": float((conf_oos >= t95).mean()),
           "n_in": int(len(conf_in)), "n_oos": int(len(conf_oos)), "n_nan_scores": n_nan}
    if thr is not None:
        rec = float((conf_oos < thr).mean())
        keep = float((conf_in >= thr).mean())
        tp, fp = (conf_oos < thr).sum(), (conf_in < thr).sum()
        prec = float(tp / max(tp + fp, 1))
        out.update({"thr": float(thr), "oos_recall": rec, "inscope_kept": keep, "oos_precision": prec,
                    "oos_f1": float(2 * prec * rec / max(prec + rec, 1e-12))})
    return out


def selective(conf: np.ndarray, correct: np.ndarray, coverages=(1.0, 0.8, 0.6, 0.5)) -> Dict[str, float]:
    """Accuracy when only the most confident fraction is answered, plus AURC (area under risk-coverage)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable")
    c = correct[order]
    n = len(c)
    out = {}
    for cov in coverages:
        m = max(1, int(round(cov * n)))
        out[f"acc@{int(round(cov * 100))}"] = float(c[:m].mean())
    risk = 1 - np.cumsum(c) / np.arange(1, n + 1)
    out["aurc"] = float(risk.mean())
    return out


# ---------------------------------------------------------------------------------------------------
# Branch training wrapper
# ---------------------------------------------------------------------------------------------------

LR = {"probe": (1e-4, 3e-3), "blocks": (1e-4, 1e-3), "full": (5e-5, 1e-3)}
EPOCHS = {"probe": 20, "blocks": 6, "full": 5}


def make(kind: str, split: int, depth: int, labels: Sequence[str], trunk: Trunk):
    if kind == "probe":
        return ProbeBranch(split, list(labels), trunk.hidden)
    if kind == "blocks":
        return BlockBranch(split, list(labels), trunk.hidden, depth, trunk)
    if kind == "full":
        return BlockBranch(0, list(labels), trunk.hidden, trunk.n_layers, trunk)
    raise ValueError(kind)


def run_branch(branch, cache: FeatureCache, tr: Sequence[int], y_tr: torch.Tensor, va: Sequence[int],
               y_va: torch.Tensor, kind: str, seed: int = 0, soft_tr: Optional[torch.Tensor] = None,
               select: bool = True, epochs: Optional[int] = None, log: Callable = lambda s: None,
               min_temp_rows: int = 30) -> Dict:
    """Train with tarski.train.train_branch (>= 300 optimiser steps; best-epoch selection only with >= 50
    validation rows and `select`), then fit one temperature on the validation rows (hard labels).
    Returns the training info and the fitted temperature."""
    lr_l, lr_h = LR[kind]
    t0 = time.time()
    info = train_branch(branch, cache, list(tr), y_tr, soft_tr, list(va), y_va, epochs=epochs or EPOCHS[kind],
                        lr_layers=lr_l, lr_head=lr_h, seed=seed, log=lambda s: None,
                        min_val=50 if select else 10 ** 9)
    steps = len(info["history"]) * ((len(tr) + 31) // 32)
    t = 1.0
    if len(va) >= min_temp_rows:
        t = fit_temperature(predict_logits(branch, cache, list(va)), y_va)
    branch.temperature.fill_(t)
    out = {"train_s": round(time.time() - t0, 1), "epochs_run": len(info["history"]), "opt_steps": steps,
           "selection": info["selection"], "val_acc": info["val_acc"], "temperature": t,
           "n_train": len(tr), "n_val": len(va)}
    log(f"      {kind}: {out['epochs_run']} epochs, {steps} steps, {out['train_s']}s, val acc {info['val_acc']:.3f}, T={t:.2f}")
    return out


def probs_of(branch, cache: FeatureCache, idx: Sequence[int]) -> np.ndarray:
    z = predict_logits(branch, cache, list(idx))
    return torch.softmax(z / float(branch.temperature), -1).numpy()


def metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    m = evaluate(probs, y)
    return {k: m[k] for k in ("acc", "macro_f1", "ece", "nll", "n")}


def gpu_gc() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
