"""Exploration K: soft-label bit-value curves, and branches vs. a full fine-tune, as N grows.

Two questions the brief and the coordinator's follow-up both ask for directly:

  1. "How many bits is a soft label worth?" -- at a fixed N, does training on the dataset's own
     soft (teacher) labels beat training on the dataset's hard (argmax) labels, and if so, how many
     EXTRA hard-labelled examples would close that gap? We report a leverage ratio: the smallest hard-
     label N that reaches the accuracy a soft-label run reaches at n.
  2. "A label-efficiency curve for branches vs full fine-tune" (the coordinator's own phrasing) -- as N
     grows from a few dozen to the whole per-workflow pool (~270 messages, the same order of magnitude
     as "a few hundred labelled Slack messages"), how close does a cheap branch get to a full fine-tune
     of the same frozen base, and at what N does the gap stop mattering in practice?

This directly follows up on the coordinator's finding that the earlier "typed-decisions collapse" was
mostly under-training (`tarski.train.train_branch` now forces >=300 optimiser steps and only trusts
best-epoch selection with >=50 validation rows): with that fixed, `blocks@14+2` reaches 78.6 vs a 79.4
full fine-tune on the customer-service decisions. This script reuses `tarski.train.train_branch`
UNCHANGED (it already carries that fix by default) rather than any custom loop, on exactly that
`blocks@14+2` branch shape (`BlockBranch(split=14, depth=2)`) against `BlockBranch(split=0,
depth=trunk.n_layers)` -- which is this codebase's own definition of "full fine-tune" (see
`experiments/sweep.py`'s docstring: "full is the reference: a branch holding copies of all layers
(split 0)").

Novelty note (see research_notes/explore/learning.md, idea K): the *qualitative* claim that soft labels
reduce sample complexity is established (Menon et al., "A Statistical Perspective on Distillation", ICML
2021). Turning it into a specific leverage number for a routing/typed-decision branch, and pairing it
with a branch-vs-full-fine-tune curve at the same label counts, is the part we did not find written up
elsewhere -- it is a quantitative planning tool ("expect to need about Nx as many labels if you only have
hard ones, and a branch instead of a full fine-tune costs you about Y accuracy points at your label
budget"), not a new mechanism.

Usage:
  .venv/bin/python explore/learning_label_efficiency.py --smoke
  .venv/bin/python explore/learning_label_efficiency.py --out results/tarski/explore_label_efficiency.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `tarski`

from tarski.branches import BlockBranch, ProbeBranch
from tarski.data import Example, load_typed_decisions
from tarski.train import FeatureCache, evaluate, predict_logits, train_branch
from tarski.trunk import Trunk

DEFAULT_TASK_FULL = "customer_service.action"
DEFAULT_TASK_SMOKE = "customer_service.action"


def leverage_lookup(hard_curve: List[Dict], target_acc: float) -> Optional[int]:
    """Smallest hard-label N (among tested points) whose accuracy reaches `target_acc`."""
    for pt in sorted(hard_curve, key=lambda p: p["n"]):
        if pt["acc"] >= target_acc:
            return pt["n"]
    return None


def _cap(rows: List[Example], cap: Optional[int], rng: np.random.Generator) -> List[Example]:
    if cap and len(rows) > cap:
        idx = sorted(rng.choice(len(rows), cap, replace=False).tolist())
        rows = [rows[i] for i in idx]
    return rows


def run_task(task: str, trunk: Trunk, ds, light_split: int, light_depth: int, full_depth: int,
            ns: List[int], full_ft_ns: List[int], epochs: int, seed: int, run_full_finetune: bool,
            min_steps_light: int, min_steps_full: int, cap_pool: Optional[int] = None,
            cap_val: Optional[int] = None, cap_test: Optional[int] = None, log=print) -> Dict:
    cap_rng = np.random.default_rng(seed)
    pool = _cap([e for e in ds.train if task in e.y], cap_pool, cap_rng)
    val = _cap([e for e in ds.val if task in e.y], cap_val, cap_rng)
    test = _cap([e for e in ds.test if task in e.y], cap_test, cap_rng)
    labels = ds.tasks[task].labels
    rng = np.random.default_rng(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    log(f"[{task}] pool={len(pool)} val={len(val)} test={len(test)} labels={labels}")

    texts = [e.text for e in pool] + [e.text for e in val] + [e.text for e in test]
    depths = sorted({0, light_split}) if run_full_finetune else [light_split]
    t0 = time.time()
    cache = FeatureCache(trunk, texts, depths, ds.max_len)
    log(f"  cached trunk depths {depths} for {len(texts)} messages in {cache.seconds:.1f}s")
    n_pool, n_val = len(pool), len(val)
    val_idx = list(range(n_pool, n_pool + n_val))
    test_idx = list(range(n_pool + n_val, len(texts)))
    y_val = torch.tensor([e.y[task] for e in val])
    y_test = torch.tensor([e.y[task] for e in test])
    soft_test = torch.tensor(np.stack([e.soft[task] for e in test]))

    def eval_branch(branch) -> Dict:
        z = predict_logits(branch, cache, test_idx)
        probs = torch.softmax(z, -1).numpy()
        return evaluate(probs, y_test.numpy(), soft_test.numpy())

    curves: Dict[str, List[Dict]] = {"light_hard": [], "light_soft": [], "full_finetune_soft": []}
    for n in [n for n in ns if n <= n_pool]:
        idx = order[:n]
        y = torch.tensor([pool[i].y[task] for i in idx])
        soft = torch.tensor(np.stack([pool[i].soft[task] for i in idx]))

        bh = BlockBranch(light_split, labels, trunk.hidden, light_depth, trunk)
        train_branch(bh, cache, idx, y, None, val_idx, y_val, epochs=epochs, seed=seed,
                    min_steps=min_steps_light)
        curves["light_hard"].append({"n": n, "params": sum(p.numel() for p in bh.parameters()),
                                    **eval_branch(bh)})

        bs_ = BlockBranch(light_split, labels, trunk.hidden, light_depth, trunk)
        train_branch(bs_, cache, idx, y, soft, val_idx, y_val, epochs=epochs, seed=seed,
                    min_steps=min_steps_light)
        curves["light_soft"].append({"n": n, "params": sum(p.numel() for p in bs_.parameters()),
                                    **eval_branch(bs_)})
        log(f"  n={n:4d}  light_hard acc={curves['light_hard'][-1]['acc']:.3f}   "
            f"light_soft acc={curves['light_soft'][-1]['acc']:.3f}")

    if run_full_finetune:
        for n in [n for n in full_ft_ns if n <= n_pool]:
            idx = order[:n]
            y = torch.tensor([pool[i].y[task] for i in idx])
            soft = torch.tensor(np.stack([pool[i].soft[task] for i in idx]))
            bf = BlockBranch(0, labels, trunk.hidden, full_depth, trunk)
            t1 = time.time()
            train_branch(bf, cache, idx, y, soft, val_idx, y_val, epochs=epochs, seed=seed,
                        min_steps=min_steps_full)
            ev = eval_branch(bf)
            curves["full_finetune_soft"].append({"n": n, "params": sum(p.numel() for p in bf.parameters()),
                                               "train_s": round(time.time() - t1, 1), **ev})
            log(f"  n={n:4d}  full_finetune_soft acc={ev['acc']:.3f} ({time.time() - t1:.0f}s)")

    leverage = []
    for pt in curves["light_soft"]:
        need = leverage_lookup(curves["light_hard"], pt["acc"])
        leverage.append({"soft_n": pt["n"], "soft_acc": pt["acc"], "hard_n_for_same_acc": need,
                         "leverage_ratio": (need / pt["n"]) if need else None})
    for pt in curves["full_finetune_soft"]:
        need_light = leverage_lookup(curves["light_soft"], pt["acc"])
        pt["light_soft_n_for_same_acc"] = need_light

    return {"labels": labels, "n_pool": n_pool, "n_test": len(test), "curves": curves,
           "soft_vs_hard_leverage": leverage}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--light-split", type=int, default=14)
    ap.add_argument("--light-depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--no-full-finetune", action="store_true",
                    help="skip the expensive split=0/depth=n_layers 'full fine-tune' curve")
    args = ap.parse_args()

    if args.smoke:
        # BlockBranch runs REAL transformer layers per step (light: 2 layers, full: all of them), not a
        # cached linear head, so train_branch's default min_steps=300 (correct for the full/GPU run) is
        # far too slow on a CPU smoke test -- cap both explicitly, just enough to exercise every code
        # path (data pooling, caching at two depths, both branch shapes, leverage lookup, JSON output).
        device = "cpu"
        tasks = [args.tasks] if args.tasks else [DEFAULT_TASK_SMOKE]
        ns, full_ft_ns, epochs = [8, 16], [8], 1
        min_steps_light, min_steps_full = 4, 2
        cap_pool, cap_val, cap_test = 16, 8, 8
        run_full_finetune = not args.no_full_finetune
    else:
        device = args.device
        tasks = args.tasks.split(",") if args.tasks else [DEFAULT_TASK_FULL]
        ns = [20, 50, 100, 160, 270]
        full_ft_ns = [50, 270]                # sparser: a full fine-tune point costs a few GPU-minutes
        epochs = args.epochs
        min_steps_light, min_steps_full = 300, 300
        cap_pool, cap_val, cap_test = None, None, None
        run_full_finetune = not args.no_full_finetune

    t_start = time.time()
    trunk = Trunk(device=device)
    ds = load_typed_decisions(seed=args.seed)
    print(f"loaded {ds.summary()}")

    results = {}
    for task in tasks:
        results[task] = run_task(task, trunk, ds, args.light_split, args.light_depth, trunk.n_layers,
                                 ns, full_ft_ns, epochs, args.seed, run_full_finetune,
                                 min_steps_light, min_steps_full, cap_pool, cap_val, cap_test)

    print("\n=== label-efficiency summary ===")
    for task, r in results.items():
        print(f"[{task}] n_pool={r['n_pool']}")
        for row in r["soft_vs_hard_leverage"]:
            lev = f"{row['leverage_ratio']:.2f}x" if row["leverage_ratio"] else ">max tested"
            print(f"  soft n={row['soft_n']:4d} (acc={row['soft_acc']:.3f}) needs hard n="
                  f"{row['hard_n_for_same_acc']} to match  -> leverage {lev}")
        for row in r["curves"]["full_finetune_soft"]:
            print(f"  full_finetune n={row['n']:4d} acc={row['acc']:.3f}  "
                  f"(light branch would need n={row['light_soft_n_for_same_acc']} for the same acc)")

    out_data = {"config": vars(args), "results": results, "elapsed_s": round(time.time() - t_start, 1)}
    out = args.out or (None if args.smoke else "results/tarski/explore_label_efficiency.json")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(out_data, f, indent=1)
        print(f"wrote {out}")
    print(f"total elapsed: {out_data['elapsed_s']}s")


if __name__ == "__main__":
    main()
