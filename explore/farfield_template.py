"""Far-field idea 11: template cancellation (efference copy) on typed-decisions.

The motor system sends a copy of each command to sensory cortex so that self-generated input is subtracted
before perception. A workflow's JSON schema is self-generated input: keys, punctuation and boilerplate are
identical across its messages. Here the per-field mean trunk state (over the workflow's training tokens) is
subtracted from every token of that field before the branch reads it, so the branch spends its capacity on
what varies between messages.

Conditions on the cached trunk states at the split:
  none     the cache as is
  global   subtract one mean token vector per workflow (mean removal, All-but-the-Top style)
  field    subtract the per-field mean (fields = JSON key paths to depth 2, from farfield_coarsegrain)

Kinds: `probe` (LayerNorm per token, mean pool, linear: the per-token subtraction is not absorbed by the
head because the LayerNorm is nonlinear) and `blocks` (copied layers, which expect the raw residual stream).

Testable prediction: `field` helps probes on field-local decisions by a few points and hurts or does nothing
for blocks (whose copied attention/MLP weights were trained on the raw stream and re-learn the offset).

Usage:
  .venv/bin/python explore/farfield_template.py --smoke
  .venv/bin/python explore/farfield_template.py --out results/tarski/explore_template.json
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import numpy as np
import torch

from explore.farfield_coarsegrain import SYNTAX, token_fields
from explore.farfield_common import Logger, dump, out_paths, split_indices, subsample, train_and_eval
from tarski import data
from tarski.branches import BlockBranch, ProbeBranch
from tarski.train import FeatureCache
from tarski.trunk import Trunk


def cancelled_cache(cache: FeatureCache, depth: int, groups: List[List[int]], tok_fields: List[List[str]], train_rows: List[int],
                    mode: str) -> FeatureCache:
    """A shallow copy of the cache whose token states have the workflow template subtracted.
    `groups`: message indices per workflow; `mode`: 'global' (one mean per workflow) or 'field'."""
    c = copy.copy(cache)
    c.h = {depth: list(cache.h[depth])}
    tr = set(train_rows)
    for ids in groups:
        sums: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        for i in ids:
            if i not in tr:
                continue
            h = cache.h[depth][i].float()
            names = ["_all"] * len(h) if mode == "global" else tok_fields[i]
            for f in set(names):
                m = torch.tensor([n == f for n in names])
                sums[f] = sums.get(f, 0) + h[m].sum(0)
                counts[f] = counts.get(f, 0) + int(m.sum())
        means = {f: sums[f] / counts[f] for f in sums}
        fallback = sum(sums.values()) / sum(counts.values())
        for i in ids:
            h = cache.h[depth][i].float()
            names = ["_all"] * len(h) if mode == "global" else tok_fields[i]
            t = torch.stack([means.get(f, fallback) for f in names])
            c.h[depth][i] = (h - t).to(torch.float16)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", type=int, default=14)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--kinds", nargs="*", default=["probe", "blocks"])
    ap.add_argument("--conditions", nargs="*", default=["none", "global", "field"])
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--field-depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.device = args.device or "cpu"
        args.epochs, args.min_steps = 1, 6
        args.tasks = args.tasks or ["customer_service.category", "invoice_processing.duplicate"]
    out, logp = out_paths(args, "template")
    L = Logger(logp)
    t_start = time.time()
    trunk = Trunk(device=args.device)
    ds = data.load_typed_decisions()
    wf_of = lambda e: next(t.split(".")[0] for t in e.y)
    if args.smoke:
        ds = copy.copy(ds)
        ds.train, ds.val, ds.test = (subsample(ds.train, 160, args.seed, wf_of), subsample(ds.val, 32, args.seed + 1, wf_of),
                                     subsample(ds.test, 80, args.seed + 2, wf_of))
    tasks = args.tasks or list(ds.tasks)
    allx = ds.train + ds.val + ds.test
    idx = split_indices(len(ds.train), len(ds.val), len(allx))
    workflows = sorted({wf_of(e) for e in allx})
    groups = [[i for i, e in enumerate(allx) if wf_of(e) == wf] for wf in workflows]
    L(f"== template cancellation on {ds.summary()} | split {args.split}+{args.depth}, kinds {args.kinds}, conditions {args.conditions}, "
      f"{len(tasks)} tasks, device {trunk.device}, smoke={args.smoke}")
    texts = [e.text for e in allx]
    enc = trunk.tok(texts, truncation=True, max_length=ds.max_len, return_offsets_mapping=True)
    tok_fields = [token_fields(t, enc["offset_mapping"][i], args.field_depth) for i, t in enumerate(texts)]
    base_cache = FeatureCache(trunk, texts, [args.split], ds.max_len)
    assert all(base_cache.lengths[i] == len(tok_fields[i]) for i in range(len(allx)))
    L(f"  cached depth {args.split}: {base_cache.seconds:.1f}s ({base_cache.bytes() / 1e6:.0f} MB)")
    caches = {"none": base_cache}
    for cond in args.conditions:
        if cond != "none":
            t0 = time.time()
            caches[cond] = cancelled_cache(base_cache, args.split, groups, tok_fields, list(idx["train"]), cond)
            L(f"  built '{cond}' cache in {time.time() - t0:.1f}s")

    results = {"config": vars(args), "device": str(trunk.device), "tasks": {}, "mean": {}}
    for task in tasks:
        labels = ds.tasks[task].labels
        results["tasks"][task] = {}
        for kind in args.kinds:
            for cond in args.conditions:
                if kind == "probe":
                    branch = ProbeBranch(args.split, labels, trunk.hidden)
                else:
                    branch = BlockBranch(args.split, labels, trunk.hidden, args.depth, trunk, init="next")
                r = train_and_eval(branch, caches[cond], allx, idx, task, epochs=args.epochs, min_steps=args.min_steps, seed=args.seed)
                m = r["metrics"]
                results["tasks"][task][f"{kind}/{cond}"] = m
                L(f"  [{task}] {kind:>6}/{cond:<6}: acc {m['acc']:.3f} macro-F1 {m['macro_f1']:.3f} NLL {m['nll']:.3f} "
                  f"Brier {m.get('brier_soft', float('nan')):.3f} ({m['steps_run']} steps, {m['train_s']}s)")
                del branch
        dump(results, out)
    for kind in args.kinds:
        for cond in args.conditions:
            key = f"{kind}/{cond}"
            ms = [results["tasks"][t][key] for t in tasks]
            results["mean"][key] = {k: float(np.mean([m[k] for m in ms])) for k in ("acc", "macro_f1", "nll", "brier_soft")}
    results["wall_seconds"] = round(time.time() - t_start, 1)
    dump(results, out)
    L(f"== done in {results['wall_seconds']}s; wrote {out}")
    for key, m in results["mean"].items():
        L(f"   {key:>13}: mean acc {m['acc']:.4f}  macro-F1 {m['macro_f1']:.4f}  NLL {m['nll']:.4f}  Brier {m['brier_soft']:.4f}")


if __name__ == "__main__":
    main()
