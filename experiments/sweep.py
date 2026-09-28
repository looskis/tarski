"""Accuracy and calibration by branch type, split depth and branch depth, per task.

Each config trains one branch per task on a trunk cache shared by every config at the same split.
`full` is the reference: a branch holding copies of all layers (split 0), i.e. a full fine-tune with
the embedding table frozen. Results are written after every config, and configs already present in the
output file are skipped, so an interrupted run resumes.

Usage:
  python -m experiments.sweep --dataset banking77 --out results/tarski/sweep_banking77.json
  python -m experiments.sweep --dataset clinc150 --probe-splits 6 11 16 22 --block-splits 6 11 16 20
"""

import argparse
import json
import os
import time

import numpy as np

from tarski import data, train
from tarski.trunk import Trunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--probe-splits", type=int, nargs="*", default=[4, 8, 11, 14, 18, 22])
    ap.add_argument("--block-splits", type=int, nargs="*", default=[4, 8, 11, 14, 18, 20])
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--full", action="store_true", help="also run the full fine-tune reference")
    ap.add_argument("--full-tasks", nargs="*", help="tasks for the full reference (default: the first task only, "
                                                     "since a full fine-tune per task is the slow part)")
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--block-epochs", type=int, default=6)
    ap.add_argument("--full-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logf = open(args.out.replace(".json", ".log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    trunk = Trunk()
    ds = data.load(args.dataset)
    tasks = args.tasks or list(ds.tasks)
    log(f"== {ds.summary()} | base {trunk.base} on {trunk.device} | seed {args.seed}")
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}

    configs = [("probe", k, 0, args.probe_epochs, 1e-4, 3e-3) for k in args.probe_splits]
    configs += [("blocks", k, d, args.block_epochs, 1e-4, 1e-3) for k in args.block_splits for d in args.depths
                if k + d <= trunk.n_layers]
    if args.full:
        configs.append(("blocks", 0, trunk.n_layers, args.full_epochs, 5e-5, 1e-3))

    by_split = {}
    for kind, k, d, epochs, lr_l, lr_h in configs:
        name = "full" if (kind == "blocks" and k == 0 and d == trunk.n_layers) else \
            (f"probe@{k}" if kind == "probe" else f"blocks@{k}+{d}")
        if name in results:
            log(f"-- {name}: already done, skipping")
            continue
        log(f"-- {name}")
        # keep only the trunk cache for the current split to bound memory
        cache = by_split.setdefault(k, {})
        for other in [s for s in by_split if s != k]:
            del by_split[other]
        t0 = time.time()
        run_tasks = (args.full_tasks or tasks[:1]) if name == "full" else tasks
        r = train.fit(trunk, ds, run_tasks, kind=kind, split=k, depth=d, epochs=epochs, lr_layers=lr_l,
                      lr_head=lr_h, seed=args.seed, log=log, cache=cache)
        mean = {m: float(np.mean([v[m] for v in r.values()])) for m in ("acc", "macro_f1", "ece", "nll")}
        results[name] = {"kind": kind, "split": k, "depth": d, "mean": mean, "tasks": r,
                         "wall_s": round(time.time() - t0, 1)}
        log(f"   {name} mean: " + json.dumps({m: round(v, 4) for m, v in mean.items()}))
        json.dump(results, open(args.out, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
