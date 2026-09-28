"""Runtime: one resident trunk, branches loaded from disk on demand.

`Engine.decide(texts, tasks)` runs the trunk once per batch, stopping at the deepest split any requested
branch needs, and hands each branch the hidden states at its own split. Branches are kept in an LRU
cache: switching to a task that is not resident costs a load of its branch file (MB), never a reload of
the base model.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from tarski.branches import Branch
from tarski.fused import FusedBlocks, group_branches
from tarski.train import autocast
from tarski.trunk import DEFAULT_BASE, Trunk


def confidence(p: np.ndarray) -> float:
    """1 - H(p)/ln K, as Jev reports it: 1 when certain, 0 when uniform."""
    k = len(p)
    if k < 2:
        return 1.0
    h = -float((p * np.log(np.clip(p, 1e-12, 1.0))).sum())
    return max(0.0, 1.0 - h / math.log(k))


def scan_store(store: str) -> Dict[str, Dict]:
    out = {}
    if not os.path.isdir(store):
        return out
    for name in sorted(os.listdir(store)):
        meta_path = os.path.join(store, name, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["bytes"] = os.path.getsize(os.path.join(store, name, "branch.safetensors"))
            out[meta.get("task", name)] = {**meta, "dir": os.path.join(store, name)}
    return out


class Engine:
    def __init__(self, store: str, base: Optional[str] = None, device: Optional[str] = None, max_loaded: int = 64,
                 fuse: bool = True):
        self.store = store
        self.fuse = fuse
        self._fused: Dict[tuple, FusedBlocks] = {}
        metas = scan_store(store)
        bases = {m["base"] for m in metas.values()}
        if base is None:
            if len(bases) > 1:
                raise ValueError(f"store {store!r} mixes branches for several bases {sorted(bases)}; pass base=")
            base = bases.pop() if bases else DEFAULT_BASE
        self.trunk = Trunk(base, device)
        self.max_loaded = max(1, int(max_loaded))
        self._loaded: "OrderedDict[str, Branch]" = OrderedDict()
        self._lock = threading.RLock()
        self.stats = {"loads": 0, "evictions": 0, "load_ms": []}

    # -- task registry ------------------------------------------------------------------------------

    def tasks(self) -> Dict[str, Dict]:
        metas = scan_store(self.store)
        for name, m in metas.items():
            m["loaded"] = name in self._loaded
        return metas

    def branch(self, task: str) -> Branch:
        with self._lock:
            if task in self._loaded:
                self._loaded.move_to_end(task)
                return self._loaded[task]
            metas = scan_store(self.store)
            if task not in metas:
                raise KeyError(task)
            t0 = time.perf_counter()
            b = Branch.load(metas[task]["dir"], self.trunk)
            self.stats["loads"] += 1
            self.stats["load_ms"].append((time.perf_counter() - t0) * 1000)
            self._loaded[task] = b
            while len(self._loaded) > self.max_loaded:
                self._loaded.popitem(last=False)
                self.stats["evictions"] += 1
            return b

    def unload(self, task: Optional[str] = None) -> None:
        with self._lock:
            if task is None:
                self._loaded.clear()
            else:
                self._loaded.pop(task, None)

    # -- inference ----------------------------------------------------------------------------------

    @torch.no_grad()
    def decide(self, texts: Sequence[str], tasks: Sequence[str]) -> Dict:
        """Probabilities per text per task, plus where the time went."""
        if not tasks:
            raise ValueError("no tasks requested")
        with self._lock:
            t0 = time.perf_counter()
            branches = {t: self.branch(t) for t in tasks}
            t_load = time.perf_counter()
            max_len = max(int(b.meta.get("max_len", self.trunk.max_len)) for b in branches.values())
            batch = self.trunk.tokenize(texts, max_len)
            n_tokens = int(batch["attention_mask"].sum())
            dev = self.trunk.device
            with autocast(dev):
                taps, ctx = self.trunk.taps(batch["input_ids"], batch["attention_mask"],
                                            {b.split for b in branches.values()})
                self._sync()
                t_trunk = time.perf_counter()
                probs, branch_ms, done = {}, {}, set()
                if self.fuse:
                    for key, names in group_branches(branches).items():
                        if len(names) < 2:
                            continue
                        s = time.perf_counter()
                        fk = tuple((n, id(branches[n])) for n in names)
                        if fk not in self._fused:
                            if len(self._fused) > 32:
                                self._fused.clear()
                            self._fused[fk] = FusedBlocks([branches[n] for n in names])
                        p = self._fused[fk].probs(taps[key[0]], ctx).float().cpu().numpy()
                        for i, n in enumerate(names):
                            probs[n] = p[i]
                        branch_ms["+".join(names)] = (time.perf_counter() - s) * 1000
                        done.update(names)
                for t, b in branches.items():
                    if t in done:
                        continue
                    s = time.perf_counter()
                    probs[t] = b.probs(taps[b.split], ctx).float().cpu().numpy()
                    branch_ms[t] = (time.perf_counter() - s) * 1000
            results = [{t: probs[t][i] for t in tasks} for i in range(len(texts))]
            return {"results": results, "input_tokens": n_tokens,
                    "timing_ms": {"load": (t_load - t0) * 1000, "trunk": (t_trunk - t_load) * 1000,
                                  "branches": branch_ms, "total": (time.perf_counter() - t0) * 1000},
                    "depth": max(b.split for b in branches.values())}

    def _sync(self):
        if self.trunk.device.type == "mps":
            torch.mps.synchronize()
        elif self.trunk.device.type == "cuda":
            torch.cuda.synchronize()

    def labels(self, task: str) -> List[str]:
        return self.branch(task).labels
