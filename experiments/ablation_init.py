"""Where should a branch's layers come from? Copies of the layers the trunk would run next ("next"),
copies of the base's last layers ("top"), or fresh random layers ("random"), with the trunk frozen below
the split and every base layer above the branch dropped.

Usage: python -m experiments.ablation_init --out results/tarski/ablation_init.json
"""

import argparse
import json
import os

from tarski import data, train
from tarski.trunk import Trunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150"])
    ap.add_argument("--tasks", nargs="*", help="tasks to ablate (default: intent where present, else all)")
    ap.add_argument("--splits", type=int, nargs="*", default=[4, 8, 14])
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--inits", nargs="*", default=["next", "top", "random"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--seed1-splits", type=int, nargs="*", default=[8], help="splits that also run seeds > 0")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    trunk = Trunk()
    for name in a.datasets:
        ds = data.load(name)
        tasks = [t for t in (a.tasks or ["intent"]) if t in ds.tasks] or list(ds.tasks)
        for k in a.splits:
            cache = {}
            for seed in a.seeds:
                if seed > 0 and k not in a.seed1_splits:
                    continue
                for init in a.inits:
                    todo = [t for t in tasks if f"{name}|{t}|{k}+{a.depth}|{init}|seed{seed}" not in res]
                    if not todo:
                        continue
                    log(f"-- {name} {k}+{a.depth} {init} seed{seed}: {len(todo)} task(s)")
                    r = train.fit(trunk, ds, todo, kind="blocks", split=k, depth=a.depth, epochs=a.epochs,
                                  seed=seed, init=init, log=log, cache=cache)
                    for t, m in r.items():
                        res[f"{name}|{t}|{k}+{a.depth}|{init}|seed{seed}"] = m
                    json.dump(res, open(a.out, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
