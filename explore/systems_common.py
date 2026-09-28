"""Shared helpers for the explore/systems_*.py scripts (logging, dataset subsets, per-task branch
training that mirrors tarski.train.fit but returns the branches, JSON structure masks, timing)."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch

from tarski import data
from tarski.branches import BlockBranch, ProbeBranch
from tarski.train import FeatureCache, _targets, evaluate, fit_temperature, predict_logits, train_branch
from tarski.trunk import Trunk


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


def save(res: Dict, out: Optional[str]) -> None:
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def threads():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def subset(ds: data.Dataset, tasks: Sequence[str], n_tr: int, n_va: int, n_te: int,
           max_len: Optional[int] = None, stride: int = 1) -> data.Dataset:
    """Messages that carry every task in `tasks`, the first n of each split (every `stride`-th)."""
    keep = lambda xs, n: [e for e in xs if all(t in e.y for t in tasks)][::stride][:n]
    return data.Dataset(ds.name, {t: ds.tasks[t] for t in tasks}, keep(ds.train, n_tr), keep(ds.val, n_va),
                        keep(ds.test, n_te), max_len or ds.max_len)


def split_index(ds: data.Dataset):
    """Positions of train/val/test messages in the concatenation train + val + test."""
    n_tr, n_va = len(ds.train), len(ds.val)
    allx = ds.train + ds.val + ds.test
    return allx, {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va), "test": range(n_tr + n_va, len(allx))}


def all_texts(ds: data.Dataset) -> List[str]:
    return [e.text for e in ds.train + ds.val + ds.test]


def make(kind: str, split: int, labels: List[str], trunk: Trunk):
    """kind: probe | blocks:D"""
    if kind == "probe":
        return ProbeBranch(split, labels, trunk.hidden)
    return BlockBranch(split, labels, trunk.hidden, int(kind.split(":")[1]), trunk)


def train_tasks(trunk: Trunk, ds: data.Dataset, cache, tasks: Sequence[str], kind: str, split: int, seed: int = 0,
                epochs: Optional[int] = None, min_steps: int = 300, lr_layers: float = 1e-4,
                lr_head: Optional[float] = None, build: Optional[Callable] = None,
                log: Callable = print) -> Dict[str, Dict]:
    """One branch per task on `cache` (FeatureCache interface over train+val+test), temperature on val,
    test metrics. Returns {task: {"branch", "T", "metrics", "sel", "y", "test_probs"}}."""
    allx, rng = split_index(ds)
    out = {}
    for task in tasks:
        sel = {s: [i for i in rng[s] if task in allx[i].y] for s in rng}
        y, soft = {}, {}
        for s in sel:
            y[s], soft[s] = _targets([allx[i] for i in sel[s]], task)
        labels = ds.tasks[task].labels
        br = build(task, labels) if build else make(kind, split, labels, trunk)
        ep = epochs if epochs is not None else (20 if kind == "probe" else 6)
        lh = lr_head if lr_head is not None else (3e-3 if kind == "probe" else 1e-3)
        t0 = time.time()
        info = train_branch(br, cache, sel["train"], y["train"], soft["train"], sel["val"], y["val"], epochs=ep,
                            lr_layers=lr_layers, lr_head=lh, seed=seed, min_steps=min_steps)
        T = fit_temperature(predict_logits(br, cache, sel["val"]), y["val"], soft["val"]) if len(sel["val"]) >= 30 else 1.0
        br.temperature.fill_(T)
        p = torch.softmax(predict_logits(br, cache, sel["test"]) / T, -1).numpy()
        m = evaluate(p, y["test"].numpy(), None if soft["test"] is None else soft["test"].numpy())
        m.update({"train_s": round(time.time() - t0, 1), "T": T, "epochs_run": len(info["history"])})
        out[task] = {"branch": br, "T": T, "metrics": m, "sel": sel, "y": y, "soft": soft, "test_probs": p}
        log(f"   [{kind}@{split}] {task}: test acc {m['acc']:.4f} ({m['train_s']:.0f}s)")
    return out


def cache_from_states(trunk: Trunk, states: Dict[int, List[torch.Tensor]]) -> FeatureCache:
    """A FeatureCache built from precomputed per-message states {depth: [ (L_i, D) ]}."""
    fc = FeatureCache.__new__(FeatureCache)
    fc.trunk, fc.depths = trunk, sorted(states)
    fc.h = {d: [t.to("cpu", torch.float16) for t in v] for d, v in states.items()}
    fc.lengths = [t.shape[0] for t in states[fc.depths[0]]]
    fc.seconds = 0.0
    return fc


# ---------------------------------------------------------------------------------------------------
# JSON structure (for typed-decisions states)
# ---------------------------------------------------------------------------------------------------

def _walk_keys(o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            out.append(k)
            _walk_keys(v, out)
    elif isinstance(o, list):
        for v in o:
            _walk_keys(v, out)


def json_structure(tok, text: str, max_len: int, workflow: str):
    """Per token of `tok(text)` (with special tokens, truncated like Trunk.token_ids): whether it is pure
    JSON structure (a key, or punctuation) and its schema signature (workflow, current key, token id,
    occurrence since that key). Special tokens and value tokens get signature None."""
    enc = tok(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    struct = np.zeros(len(text) + 1, bool)
    for i, ch in enumerate(text):
        if ch in '{}[]":,':
            struct[i] = True
    key_at = np.full(len(text) + 1, -1)
    keys: List[str] = []
    try:
        _walk_keys(json.loads(text), keys)
    except Exception:
        keys = []
    pos, key_names = 0, []
    for k in keys:
        pat = json.dumps(k, ensure_ascii=False) + ":"
        j = text.find(pat, pos)
        if j >= 0:
            struct[j:j + len(pat)] = True
            key_at[j:] = len(key_names)
            key_names.append(k)
            pos = j + len(pat)
    is_struct, sigs, seen = [], [], {}
    for t, (a, b) in zip(ids, offs):
        if b <= a:                                   # special token
            is_struct.append(False)
            sigs.append(None)
            continue
        s = bool(struct[a:b].all())
        is_struct.append(s)
        if s:
            kname = key_names[key_at[a]] if key_at[a] >= 0 else "<root>"
            n = seen.get((kname, t, int(key_at[a])), 0)
            seen[(kname, t, int(key_at[a]))] = n + 1
            sigs.append((workflow, kname, int(t), n))
        else:
            sigs.append(None)
    return ids, is_struct, sigs


# ---------------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------------

def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, dev, reps: int = 20, warm: int = 3) -> float:
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


def layer_flops(D: int, I: int, L: int, n_query: Optional[int] = None) -> float:
    """FLOPs of one encoder layer: linear ops for n_query tokens + attention of n_query queries over L keys."""
    q = L if n_query is None else n_query
    return 2 * q * (4 * D * D + 3 * D * I) + 4 * q * L * D
