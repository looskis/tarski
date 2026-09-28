"""Far-field idea 6: free-rider depth (public-goods economics), post hoc from the sweep results.

In a request bundle the trunk runs to the deepest requested split, so every shallower decision can read a
deeper trunk state for free. If each task stores a branch at several depths, the engine can serve each
decision from the deepest branch whose split the bundle has already paid for. This script measures what
that policy would gain, from results/tarski/lambda/sweep_*.json (three seeds where present):

  cheapest        every task at the shallowest swept blocks split (what a cost-minimising user picks)
  ride_to_D       every task at the deepest split D requested by any task in the bundle
  ride_best<=D    every task at the split <= D with the best validation accuracy (choice needs a val set)
  own_best        every task at its best split by validation accuracy, ignoring cost (the upper bound)

Testable prediction: gains are positive but small (under one point on CLINC), and can be negative when a
task's accuracy is not monotone in depth (typed-decisions action), which is why `ride_best<=D` exists.
No GPU is needed; `--smoke` runs the same computation.

Usage:
  .venv/bin/python explore/farfield_freerider.py --smoke
  .venv/bin/python explore/farfield_freerider.py --out results/tarski/explore_freerider.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import numpy as np

from explore.farfield_common import LAMBDA_DIR, Logger, REPO, dump, load_sweeps, out_paths


def blocks_configs(sweep: Dict, depth: int) -> Dict[int, Dict]:
    """split -> per-task metrics for blocks configs of the given branch depth."""
    out = {}
    for name, cfg in sweep.items():
        if cfg.get("kind") == "blocks" and cfg.get("depth") == depth and cfg.get("split", 0) > 0:
            out[int(cfg["split"])] = cfg["tasks"]
    return dict(sorted(out.items()))


def analyse(sweep: Dict, depth: int) -> Dict:
    by_split = blocks_configs(sweep, depth)
    splits = list(by_split)
    if len(splits) < 2:
        return {}
    tasks = sorted(set.intersection(*(set(t) for t in by_split.values())))
    acc = {t: {s: by_split[s][t]["acc"] for s in splits} for t in tasks}
    val = {t: {s: by_split[s][t].get("val_acc", by_split[s][t]["acc"]) for s in splits} for t in tasks}
    cheapest = splits[0]
    own_best = {t: max(splits, key=lambda s: (val[t][s], -s)) for t in tasks}
    policies = {}
    for D in splits:
        ride_best = {t: max([s for s in splits if s <= D], key=lambda s: (val[t][s], -s)) for t in tasks}
        policies[str(D)] = {
            "cheapest": float(np.mean([acc[t][cheapest] for t in tasks])),
            "ride_to_D": float(np.mean([acc[t][D] for t in tasks])),
            "ride_best_leq_D": float(np.mean([acc[t][ride_best[t]] for t in tasks])),
            "per_task_gain_ride_to_D": {t: acc[t][D] - acc[t][cheapest] for t in tasks},
            "per_task_gain_ride_best": {t: acc[t][ride_best[t]] - acc[t][cheapest] for t in tasks},
        }
    return {"splits": splits, "tasks": tasks, "acc": acc, "val_acc": val, "own_best_split": own_best,
            "own_best_mean_acc": float(np.mean([acc[t][own_best[t]] for t in tasks])), "policies": policies}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=["clinc150", "typed_cs", "banking77"])
    ap.add_argument("--depth", type=int, default=2, help="branch depth of the blocks configs to compare")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out, logp = out_paths(args, "freerider")
    L = Logger(logp)
    t0 = time.time()
    results = {"config": vars(args), "source": LAMBDA_DIR, "datasets": {}}
    for name in args.datasets:
        sweeps = load_sweeps(name)
        if not sweeps:                                           # fall back to the laptop sweeps
            p = os.path.join(REPO, "results", "tarski", f"sweep_{name}.json")
            if os.path.exists(p):
                sweeps = {f"sweep_{name}": json.load(open(p))}
        per_seed = {k: analyse(v, args.depth) for k, v in sweeps.items()}
        per_seed = {k: v for k, v in per_seed.items() if v}
        if not per_seed:
            L(f"-- {name}: no sweep with >= 2 blocks splits at depth {args.depth}; skipped")
            continue
        # aggregate policies over seeds where the same splits exist
        splits = sorted(set.intersection(*(set(v["splits"]) for v in per_seed.values())))
        agg = {}
        for D in splits:
            rows = [v["policies"][str(D)] for v in per_seed.values()]
            agg[str(D)] = {pol: {"mean": float(np.mean([r[pol] for r in rows])), "std": float(np.std([r[pol] for r in rows]))}
                           for pol in ("cheapest", "ride_to_D", "ride_best_leq_D")}
        results["datasets"][name] = {"seeds": list(per_seed), "splits": splits, "aggregate": agg, "per_seed": per_seed}
        L(f"-- {name} ({len(per_seed)} seed(s), blocks depth {args.depth}, splits {splits}); tasks: {per_seed[list(per_seed)[0]]['tasks']}")
        for D in splits:
            a = agg[str(D)]
            L(f"   deepest requested split {D:>2}: cheapest {a['cheapest']['mean']:.4f} | ride to D {a['ride_to_D']['mean']:.4f} "
              f"({a['ride_to_D']['mean'] - a['cheapest']['mean']:+.4f}) | ride best<=D {a['ride_best_leq_D']['mean']:.4f} "
              f"({a['ride_best_leq_D']['mean'] - a['cheapest']['mean']:+.4f})  [std over seeds {a['ride_to_D']['std']:.4f}]")
        v0 = per_seed[list(per_seed)[0]]
        worst = min(((t, g) for D in v0["policies"] for t, g in v0["policies"][D]["per_task_gain_ride_to_D"].items()), key=lambda x: x[1])
        L(f"   own-best mean acc (ignoring cost) {v0['own_best_mean_acc']:.4f}; most negative single-task ride gain: {worst[0]} {worst[1]:+.3f}")
    results["wall_seconds"] = round(time.time() - t0, 1)
    dump(results, out)
    L(f"== done in {results['wall_seconds']}s; wrote {out}")


if __name__ == "__main__":
    main()
