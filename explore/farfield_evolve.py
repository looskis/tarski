"""Far-field idea 12: population search over branch topology (evolution), kept small.

A population of branch genomes (split depth, branch depth, layer source) competes per task under a compute
price: fitness = -validation NLL - lambda * (split + depth) / n_layers. Tournament selection, uniform
crossover and one-gene mutation for a few generations; every genome is trained once (blocks branch on cached
trunk states, >= 300 optimiser steps) and cached. Compared on the test set with (a) the autosplit choice
(probe-curve split, depth 2, next layers), (b) the sweep's fixed blocks@14+2, and (c) the evolved genome.
The number of branch trainings is the cost of each selector.

Testable prediction: evolution finds the cheapest genome within noise of the best (it prefers shallower
splits than the sweep's 14 when lambda prices layers), and beats autosplit by under one point at 3-5x its
training cost; with 26-31 validation rows per task the fitness is noisy, which is the honest limit.

Usage:
  .venv/bin/python explore/farfield_evolve.py --smoke
  .venv/bin/python explore/farfield_evolve.py --out results/tarski/explore_evolve.json
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import numpy as np
import torch
import torch.nn.functional as F

from explore.farfield_common import Logger, dump, labels_of, out_paths, soft_of, split_indices, subsample, task_rows, train_and_eval
from tarski import data
from tarski.autosplit import choose_split, layer_curves
from tarski.branches import BlockBranch
from tarski.train import FeatureCache, predict_logits
from tarski.trunk import Trunk

Genome = Tuple[int, int, str]          # (split, depth, init)


class CacheLRU:
    def __init__(self, trunk: Trunk, texts: List[str], max_len: int, keep: int, log):
        self.trunk, self.texts, self.max_len, self.keep, self.log = trunk, texts, max_len, keep, log
        self.d: "OrderedDict[int, FeatureCache]" = OrderedDict()

    def get(self, split: int) -> FeatureCache:
        if split in self.d:
            self.d.move_to_end(split)
            return self.d[split]
        c = FeatureCache(self.trunk, self.texts, [split], self.max_len)
        self.log(f"    cached depth {split} in {c.seconds:.1f}s")
        self.d[split] = c
        while len(self.d) > self.keep:
            self.d.popitem(last=False)
        return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tasks", nargs="*", default=["customer_service.category", "customer_service.churn_risk", "customer_service.urgency"])
    ap.add_argument("--splits", type=int, nargs="*", default=[4, 6, 8, 11, 14, 16, 18, 20])
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--inits", nargs="*", default=["next", "top"])
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.3, help="NLL price of running the whole trunk (per-layer price = lam / 22)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.device = args.device or "cpu"
        args.tasks = ["customer_service.category"]
        args.splits, args.pop, args.generations, args.epochs, args.min_steps = [6, 14], 3, 2, 1, 6
    out, logp = out_paths(args, "evolve")
    L = Logger(logp)
    t_start = time.time()
    trunk = Trunk(device=args.device)
    ds = data.load_typed_decisions()
    wfs = {t.split(".")[0] for t in args.tasks}
    keep = lambda e: any(t.split(".")[0] in wfs for t in e.y)
    ds = copy.copy(ds)
    ds.train, ds.val, ds.test = [e for e in ds.train if keep(e)], [e for e in ds.val if keep(e)], [e for e in ds.test if keep(e)]
    if args.smoke:
        ds.train, ds.val, ds.test = subsample(ds.train, 100, args.seed), subsample(ds.val, 26, args.seed + 1), subsample(ds.test, 50, args.seed + 2)
    allx = ds.train + ds.val + ds.test
    idx = split_indices(len(ds.train), len(ds.val), len(allx))
    L(f"== evolution on {ds.summary()} | tasks {args.tasks} | splits {args.splits} depths {args.depths} inits {args.inits} | "
      f"pop {args.pop} x {args.generations} generations, lambda {args.lam} | device {trunk.device}")
    caches = CacheLRU(trunk, [e.text for e in allx], ds.max_len, keep=3, log=L)
    rng = random.Random(args.seed)
    results = {"config": vars(args), "device": str(trunk.device), "tasks": {}}

    # autosplit's probe curves (one trunk pass over train+val) for the baseline selector
    t0 = time.time()
    curves = layer_curves(trunk, ds, args.tasks, depths=args.splits)
    L(f"  autosplit probe curves in {time.time() - t0:.0f}s")

    for task in args.tasks:
        labels = ds.tasks[task].labels
        val_rows = task_rows(allx, task, idx["val"])
        y_val, s_val = labels_of(allx, task, val_rows), soft_of(allx, task, val_rows)
        evaluated: Dict[Genome, Dict] = {}
        n_train = [0]

        def evaluate_genome(g: Genome) -> Dict:
            if g in evaluated:
                return evaluated[g]
            split, depth, init = g
            cache = caches.get(split)
            br = BlockBranch(split, labels, trunk.hidden, depth, trunk, init=init)
            r = train_and_eval(br, cache, allx, idx, task, epochs=args.epochs, min_steps=args.min_steps, seed=args.seed)
            zv = predict_logits(br, cache, val_rows) / br.temperature.float().cpu()
            tgt = s_val if s_val is not None else F.one_hot(y_val, len(labels)).float()
            nll_val = float(-(tgt * F.log_softmax(zv, -1)).sum(-1).mean())
            cost = (split + depth) / trunk.n_layers
            rec = {"genome": list(g), "val_nll": nll_val, "val_acc": r["metrics"]["val_acc"], "cost": cost,
                   "fitness": -nll_val - args.lam * cost, "test": r["metrics"]}
            evaluated[g] = rec
            n_train[0] += 1
            L(f"    [{task}] {g}: val NLL {nll_val:.3f} val acc {rec['val_acc']:.3f} fitness {rec['fitness']:.3f} "
              f"test acc {r['metrics']['acc']:.3f} ({r['metrics']['train_s']}s)")
            del br
            return rec

        def random_genome() -> Genome:
            return (rng.choice(args.splits), rng.choice(args.depths), rng.choice(args.inits))

        def mutate(g: Genome) -> Genome:
            s, d, i = g
            which = rng.randrange(3)
            if which == 0:
                j = args.splits.index(s) + rng.choice([-1, 1])
                s = args.splits[min(max(j, 0), len(args.splits) - 1)]
            elif which == 1:
                d = rng.choice(args.depths)
            else:
                i = rng.choice(args.inits)
            return (s, d, i)

        def crossover(a: Genome, b: Genome) -> Genome:
            return tuple(a[k] if rng.random() < 0.5 else b[k] for k in range(3))

        pop = list({random_genome() for _ in range(args.pop)})
        while len(pop) < args.pop:
            pop.append(random_genome())
        history = []
        for gen in range(args.generations):
            scored = sorted((evaluate_genome(g) for g in pop), key=lambda r: -r["fitness"])
            history.append({"generation": gen, "best": scored[0]["genome"], "best_fitness": scored[0]["fitness"],
                            "population": [r["genome"] for r in scored]})
            L(f"  [{task}] generation {gen}: best {scored[0]['genome']} fitness {scored[0]['fitness']:.3f}")
            if gen == args.generations - 1:
                break
            elite = [tuple(r["genome"]) for r in scored[:2]]
            children = []
            while len(elite) + len(children) < args.pop:
                a = min(rng.sample(scored, 2), key=lambda r: -r["fitness"])["genome"]
                b = min(rng.sample(scored, 2), key=lambda r: -r["fitness"])["genome"]
                child = crossover(tuple(a), tuple(b))
                if rng.random() < 0.7:
                    child = mutate(child)
                children.append(child)
            pop = elite + children
        best = max(evaluated.values(), key=lambda r: r["fitness"])
        # baselines: autosplit choice and the sweep's fixed config, trained the same way
        auto_split = choose_split(curves[task])
        auto = evaluate_genome((auto_split, 2, "next"))
        fixed = evaluate_genome((14, 2, "next")) if 14 in args.splits else None
        cheapest_within = min((r for r in evaluated.values() if r["val_nll"] <= best["val_nll"] + 0.02), key=lambda r: r["cost"])
        results["tasks"][task] = {
            "evaluated": [r for r in evaluated.values()], "n_branch_trainings": n_train[0], "history": history,
            "evolved": best, "autosplit": {"probe_curve": curves[task], "split": auto_split, **auto},
            "fixed_14_2_next": fixed, "cheapest_within_0.02_nll_of_best": cheapest_within}
        L(f"  [{task}] evolved {best['genome']} test acc {best['test']['acc']:.3f} (cost {best['cost']:.2f}) | autosplit split {auto_split} "
          f"test acc {auto['test']['acc']:.3f} | fixed 14+2 test acc {fixed['test']['acc'] if fixed else float('nan'):.3f} | "
          f"{n_train[0]} branch trainings")
        dump(results, out)

    summary = {t: {"evolved": (r["evolved"]["genome"], round(r["evolved"]["test"]["acc"], 3)),
                   "autosplit": (r["autosplit"]["split"], round(r["autosplit"]["test"]["acc"], 3)),
                   "fixed_14_2": round(r["fixed_14_2_next"]["test"]["acc"], 3) if r["fixed_14_2_next"] else None,
                   "trainings": r["n_branch_trainings"]} for t, r in results["tasks"].items()}
    results["summary"] = summary
    results["wall_seconds"] = round(time.time() - t_start, 1)
    dump(results, out)
    L(f"== done in {results['wall_seconds']}s; wrote {out}")
    L(f"   {summary}")


if __name__ == "__main__":
    main()
