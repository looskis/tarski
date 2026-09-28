"""Idea 5c: global-only forks. Build a branch from the base's next *global* layers, skipping the local ones.

ModernBERT-base mixes across the whole message only in layers 0, 3, 6, ..., 21; the others use a
128-token window. A `blocks@k+d` branch copies base layers [k, k+d). Here a branch copies an arbitrary list
of base layers, so that one can compare, at the same split and the same number of layers (same cost):

  next      [k, k+1]                    the stock blocks@k+2
  global    the next two global layers  (split 11: [12, 15]; split 16: [18, 21])
  local     the next two local layers   (split 11: [11, 13]; split 16: [16, 17])
  next1     [k]                         the stock blocks@k+1
  global1   the next global layer       (split 11: [12]; split 16: [18])

Branches are trained with tarski.train.train_branch (min 300 steps, the same epoch-selection rule), on the
trunk cache at the split. Windows only bind on long inputs: typed-decisions states (median 240 tokens,
72% over 128) should show the effect if it exists. Banking77 (<= 64 tokens) is the control where local
layers see everything anyway. The depth scan (no sawtooth in probe accuracy) and the blocks@15..20+1
ablation on Banking77 (the global layers 15 and 18 were not special) are negative so far; this is the
architectural variant.

Prediction (geometry.md idea 5c): on typed-decisions, global >= next > local at equal cost; no difference on
Banking77.

Usage:
  .venv/bin/python explore/geometry_globalfork.py --smoke
  .venv/bin/python explore/geometry_globalfork.py --dataset typed-decisions --splits 11 --variants next global local \
      --out results/tarski/explore_globalfork_typed.json                                  # ~15 min on an A10
  .venv/bin/python explore/geometry_globalfork.py --dataset banking77 --splits 11 16 \
      --out results/tarski/explore_globalfork_banking77.json                              # ~10 min on an A10
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_common import Logger, load_dataset, splits

import numpy as np
import torch
from torch import nn

from tarski import train as ttrain
from tarski.branches import Branch, mean_pool
from tarski.trunk import Trunk


class LayerListBranch(Branch):
    """Copies of an arbitrary list of base layers (in the given order) on top of the trunk at `split`."""
    kind = "layerlist"

    def __init__(self, split, labels, trunk: Trunk, layer_ids, dropout=0.1):
        super().__init__(split, labels, trunk.hidden)
        self.layer_ids = list(layer_ids)
        self.layers = nn.ModuleList(copy.deepcopy(trunk.model.layers[i]) for i in self.layer_ids)
        for p in self.layers.parameters():
            p.requires_grad_(True)
        self.norm = trunk.copy_final_norm()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(trunk.hidden, len(labels))

    def logits(self, h, ctx):
        h = self.norm(ctx.run(self.layers, h))
        return self.out(self.drop(mean_pool(h, ctx.attention_mask)))


def variants_for(split, types):
    glob = [i for i in range(split, len(types)) if types[i] == "full_attention"]
    loc = [i for i in range(split, len(types)) if types[i] == "sliding_attention"]
    return {"next": [split, split + 1], "global": glob[:2], "local": loc[:2], "next1": [split], "global1": glob[:1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dataset", default="typed-decisions")
    ap.add_argument("--splits", type=int, nargs="*", default=[11])
    ap.add_argument("--variants", nargs="*", default=["next", "global", "local", "next1", "global1"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.min_steps, args.epochs = 12, 1
    out_path = args.out or ("results/tarski/explore_globalfork_smoke.json" if args.smoke
                            else f"results/tarski/explore_globalfork_{args.dataset}.json")
    log = Logger(out_path)
    trunk = Trunk(device="cpu" if args.smoke else None)
    ds = load_dataset(args.dataset, args.smoke)
    tasks = list(ds.tasks)
    if args.smoke and args.dataset == "typed-decisions":
        tasks = tasks[:2]
    allx, n_tr, n_va = splits(ds)
    types = trunk.cfg.layer_types
    log(f"== global-only forks | {trunk.device} | {ds.summary()[:100]} | layer types "
        + "".join("G" if t == "full_attention" else "l" for t in types))
    res = {"args": vars(args), "splits": {}}
    split_idx = {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va), "test": range(n_tr + n_va, len(allx))}
    for k in args.splits:
        fc = ttrain.FeatureCache(trunk, [e.text for e in allx], [k], ds.max_len)
        vs = variants_for(k, types)
        res["splits"][k] = {}
        seen = set()
        for v in args.variants:
            ids = vs[v]
            if tuple(ids) in seen:
                continue
            seen.add(tuple(ids))
            t0, per = time.time(), {}
            for t in tasks:
                sel = {s: [i for i in split_idx[s] if t in allx[i].y] for s in split_idx}
                y = {s: torch.tensor([allx[i].y[t] for i in sel[s]]) for s in sel}
                soft = torch.tensor(np.stack([allx[i].soft[t] for i in sel["train"]])) \
                    if all(t in allx[i].soft for i in sel["train"]) else None
                br = LayerListBranch(k, ds.tasks[t].labels, trunk, ids)
                info = ttrain.train_branch(br, fc, sel["train"], y["train"], soft, sel["val"], y["val"], epochs=args.epochs,
                                           lr_layers=1e-4, lr_head=1e-3, min_steps=args.min_steps)
                z = ttrain.predict_logits(br, fc, sel["test"])
                per[t] = {"acc": float((z.argmax(-1) == y["test"]).float().mean()), "epochs": len(info["history"])}
                del br
            mean = float(np.mean([p["acc"] for p in per.values()]))
            res["splits"][k][v] = {"layers": ids, "layer_types": ["G" if types[i] == "full_attention" else "l" for i in ids],
                                   "mean_acc": mean, "tasks": per, "s": round(time.time() - t0, 1)}
            log(f"   split {k} {v:>7} layers {ids} ({''.join(res['splits'][k][v]['layer_types'])}): mean acc {mean:.4f} "
                f"({time.time() - t0:.0f}s)")
            json.dump(res, open(out_path, "w"), indent=1)
        del fc
    log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
