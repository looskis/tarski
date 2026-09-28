"""Does a one-pass probe curve pick a good split depth per task, and what does it cost to pick?

For each task: the probe curve over depths 1..(L - branch depth) from one trunk pass, the split it
chooses at several tolerances, and the test accuracy of a probe and (optionally) a 2-layer branch
trained at that split. Compare with the sweep grid in results/tarski/sweep_*.json (the oracle) and with
the sweep's training time (what a Wei et al. style fine-tuning sweep would cost).

Usage: python -m experiments.autosplit_eval --dataset clinc150 --blocks --out results/tarski/autosplit_clinc150.json
"""

import argparse
import json
import os
import time

from tarski import autosplit, data, train
from tarski.trunk import Trunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--tols", type=float, nargs="*", default=[0.005, 0.01, 0.02])
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--blocks", action="store_true", help="also train a blocks branch at each chosen split")
    ap.add_argument("--method", choices=["probe", "branch"], default="probe",
                    help="rank depths by linear probes (one pass) or by short 1-layer branch runs")
    ap.add_argument("--proxy-steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    trunk = Trunk()
    ds = data.load(a.dataset)
    tasks = a.tasks or list(ds.tasks)
    t0 = time.time()
    if a.method == "probe":
        curves = autosplit.layer_curves(trunk, ds, tasks, depths=range(1, trunk.n_layers - a.depth + 1))
        choose = autosplit.choose_split
    else:
        curves = autosplit.branch_curves(trunk, ds, tasks, steps=a.proxy_steps, seed=a.seed)
        choose = autosplit.choose_split_branch
    curve_s = time.time() - t0
    print(f"{a.method} curves for {len(tasks)} task(s) in {curve_s:.0f}s", flush=True)
    res = {"dataset": a.dataset, "method": a.method, "curve_seconds": curve_s, "curves": curves, "chosen": {},
           "trained": {}}
    caches = {}
    for tol in a.tols:
        chosen = {t: choose(c, tol) for t, c in curves.items()}
        res["chosen"][str(tol)] = chosen
        print(f"tol {tol}: " + ", ".join(f"{t}->{k}" for t, k in chosen.items()), flush=True)
        for k in sorted(set(chosen.values())):
            group = [t for t in tasks if chosen[t] == k]
            cache = caches.setdefault(k, {})
            kinds = [("probe", 20, 3e-3)] + ([("blocks", 6, 1e-3)] if a.blocks else [])
            for kind, epochs, lr_h in kinds:
                todo = [t for t in group if f"{kind}@{k}|{t}" not in res["trained"]]
                if not todo:
                    continue
                r = train.fit(trunk, ds, todo, kind=kind, split=k, depth=a.depth, epochs=epochs, lr_head=lr_h,
                              cache=cache, log=lambda s: None, seed=a.seed)
                for t, m in r.items():
                    res["trained"][f"{kind}@{k}|{t}"] = m
                    print(f"  {kind}@{k} {t}: acc {m['acc']:.4f} ({m['train_s']:.0f}s)", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
