"""Far-field idea 8: alternative splicing of branch layers (molecular genetics).

One gene yields many proteins by splicing different, not necessarily contiguous, exons together. A blocks
branch at split k copies base layers [k, k+d) ("next"); `experiments/ablation_init.py` also tries the base's
last d layers ("top") and random weights. Here the branch is spliced from any two base layers at or above
the split, chosen by a cheap score or fixed patterns:

  next          [k, k+1]                      the harness default
  next_reversed [k+1, k]                      same layers, wrong order (is order load-bearing?)
  mid           [~(k+22)/2, +1]               two middle layers
  top           [20, 21]                      the base's last two layers
  stride        [k, ~(k+22)/2]                one next layer and one middle layer
  probe_best2   the two base layers j >= k whose frozen application to h_k gives the best probe validation
                accuracy, kept in base order (the cheap splice selector)
  random_pair   two random layers >= k, base order

Testable prediction: `next` is within noise of `probe_best2` and beats `random_pair` and `top` by 1-2
points; `next_reversed` costs little (layers are painters: mostly order-robust). A null result (all within
noise) is itself informative for H5, since it says the copied layers act as a warm start, not as code.

Usage:
  .venv/bin/python explore/farfield_splice.py --smoke
  .venv/bin/python explore/farfield_splice.py --out results/tarski/explore_splice.json
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import numpy as np
import torch
from torch import nn

from explore.farfield_common import (Logger, Standardiser, dump, fit_linear, labels_of, logits_of, out_paths, smoke_subset,
                                     split_indices, task_rows, train_and_eval)
from tarski import data
from tarski.branches import BlockBranch, Branch, mean_pool
from tarski.train import FeatureCache, autocast
from tarski.trunk import Trunk


class SplicedBranch(BlockBranch):
    """A blocks branch whose layers are copies of arbitrary base layers, in the given order."""

    def __init__(self, split: int, labels: List[str], hidden: int, layer_ids: List[int], trunk: Trunk, dropout: float = 0.1):
        Branch.__init__(self, split, labels, hidden)
        self.depth, self.init, self.layer_ids = len(layer_ids), "splice", list(layer_ids)
        self.layers = nn.ModuleList(copy.deepcopy(trunk.model.layers[j]) for j in layer_ids)
        for p in self.layers.parameters():
            p.requires_grad_(True)
        self.norm = trunk.copy_final_norm()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, len(labels))

    def config(self):
        return {**Branch.config(self), "depth": self.depth, "init": "splice", "layer_ids": self.layer_ids}


def layer_probe_curve(trunk: Trunk, cache: FeatureCache, split: int, rows: Dict[str, List[int]], y: Dict[str, torch.Tensor],
                      n_labels: int, steps: int, seed: int, bs: int = 128) -> Dict[int, float]:
    """Validation accuracy of a probe on mean_pool(frozen base layer j applied to h_split), for each j >= split."""
    out = {}
    all_rows = list(rows["train"]) + list(rows["val"])
    order = sorted(range(len(all_rows)), key=lambda i: cache.lengths[all_rows[i]])
    for j in range(split, trunk.n_layers):
        feats = torch.zeros(len(all_rows), trunk.hidden)
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            with torch.no_grad():
                h, ctx = cache.batch(split, [all_rows[i] for i in sel])
                with autocast(trunk.device):
                    h2 = ctx.run([trunk.model.layers[j]], h)
                feats[sel] = mean_pool(h2.float(), ctx.attention_mask).cpu()
        n_tr = len(rows["train"])
        st = Standardiser(feats[:n_tr])
        lin = fit_linear(st(feats[:n_tr]), y["train"], n_labels, trunk.device, steps, seed=seed)
        out[j] = float((logits_of(lin, st(feats[n_tr:]), trunk.device).argmax(-1) == y["val"]).float().mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150"])
    ap.add_argument("--clinc-tasks", nargs="*", default=["domain"])
    ap.add_argument("--split", type=int, default=8)
    ap.add_argument("--configs", nargs="*", default=["next", "next_reversed", "mid", "top", "stride", "probe_best2", "random_pair"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.device = args.device or "cpu"
        args.datasets = ["banking77"] if args.datasets == ["banking77", "clinc150"] else args.datasets
        args.epochs, args.min_steps, args.probe_steps = 1, 8, 50
    out, logp = out_paths(args, "splice")
    L = Logger(logp)
    t_start = time.time()
    trunk = Trunk(device=args.device)
    n_layers, k = trunk.n_layers, args.split
    mid = (k + n_layers) // 2
    results = {"config": vars(args), "device": str(trunk.device), "datasets": {}}

    for name in args.datasets:
        ds = data.load(name)
        if args.smoke:
            ds = smoke_subset(ds, 500, 150, 300, args.seed, key=lambda e: e.y["intent"])
        tasks = args.clinc_tasks if name == "clinc150" else list(ds.tasks)
        allx = ds.train + ds.val + ds.test
        idx = split_indices(len(ds.train), len(ds.val), len(allx))
        L(f"== {ds.summary()} | split {k}, tasks {tasks}")
        t0 = time.time()
        cache = FeatureCache(trunk, [e.text for e in allx], [k], ds.max_len)
        L(f"  cached depth {k}: {cache.seconds:.1f}s ({cache.bytes() / 1e6:.0f} MB)")
        dres = {}
        for task in tasks:
            labels = ds.tasks[task].labels
            rows = {s: task_rows(allx, task, idx[s]) for s in idx}
            y = {s: labels_of(allx, task, rows[s]) for s in rows}
            t1 = time.time()
            curve = layer_probe_curve(trunk, cache, k, rows, y, len(labels), args.probe_steps, args.seed)
            best2 = sorted(sorted(curve, key=lambda j: (-curve[j], j))[:2])
            L(f"  [{task}] probe val acc of frozen layer j on h_{k}: " + " ".join(f"{j}:{a:.3f}" for j, a in curve.items()) +
              f"  -> probe_best2 {best2} ({time.time() - t1:.0f}s)")
            rng = random.Random(args.seed)
            splices = {"next": [k, k + 1], "next_reversed": [k + 1, k], "mid": [mid, mid + 1], "top": [n_layers - 2, n_layers - 1],
                       "stride": [k, mid], "probe_best2": best2, "random_pair": sorted(rng.sample(range(k, n_layers), 2))}
            tres = {"layer_probe_curve": curve, "configs": {}}
            for cfg in args.configs:
                ids = splices[cfg]
                branch = SplicedBranch(k, labels, trunk.hidden, ids, trunk)
                r = train_and_eval(branch, cache, allx, idx, task, epochs=args.epochs, min_steps=args.min_steps, seed=args.seed)
                m = r["metrics"]
                tres["configs"][cfg] = {"layer_ids": ids, **{kk: v for kk, v in m.items()}}
                L(f"  [{task}] {cfg:>14} layers {ids}: test acc {m['acc']:.4f} macro-F1 {m['macro_f1']:.4f} NLL {m['nll']:.3f} "
                  f"(val {m['val_acc']:.4f}, {m['steps_run']} steps, {m['train_s']}s)")
                del branch
            dres[task] = tres
        results["datasets"][name] = dres
        dump(results, out)
        del cache
    # summary: accuracy relative to `next`
    summary = {}
    for name, dres in results["datasets"].items():
        for task, tres in dres.items():
            base = tres["configs"].get("next", {}).get("acc")
            summary[f"{name}/{task}"] = {cfg: round(v["acc"] - base, 4) if base is not None else v["acc"] for cfg, v in tres["configs"].items()}
    results["acc_minus_next"] = summary
    results["wall_seconds"] = round(time.time() - t_start, 1)
    dump(results, out)
    L(f"== done in {results['wall_seconds']}s; wrote {out}")
    L("   acc - next: " + str(summary))


if __name__ == "__main__":
    main()
