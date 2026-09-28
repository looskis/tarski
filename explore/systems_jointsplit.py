"""Idea 8: workload-aware joint split planning ("free depth promotion").

The trunk runs to the deepest split any requested decision needs, so choosing each task's split in
isolation (tarski.autosplit: shallowest depth within tol of the best probe) optimises the wrong thing.
Given which decisions are requested together:

  autosplit          per-task shallowest split within tol; request depth = max over its tasks
  +promote           same request depth, but each task then reads its best split <= that depth (free)
  best               per-task best split (no tolerance); request depth = max
  joint              choose request depths for the workload directly: greedily lower tasks' planned
                     splits while the workload's promoted validation accuracy stays within tol of `best`
  all@22             everything at the top

Accuracy comes from one-pass linear probes on mean-pooled trunk states at every depth (as autosplit),
fitted full-batch, with validation used for every choice and test for reporting. Workloads: typed-decisions
(requests = a workflow's 5 decisions, or random subsets of them) and CLINC150 (all subsets of
intent/domain/oos).

Tested prediction: joint/promote planning reaches `best`-level accuracy at a lower expected trunk depth
than per-task autosplit, or higher accuracy at the same depth.

Usage:
  python explore/systems_jointsplit.py --smoke
  python explore/systems_jointsplit.py --out results/tarski/explore_jointsplit.json
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explore.systems_common import Log, all_texts, save, split_index, subset, threads

import numpy as np
import torch
import torch.nn.functional as F

from tarski import data
from tarski.autosplit import pooled_by_depth
from tarski.trunk import Trunk


def probe_curve(feats: Dict[int, torch.Tensor], rows: Dict[str, List[int]], y: Dict[str, torch.Tensor],
                n_labels: int, device, steps: int = 300, lr: float = 1e-2, wd: float = 1e-4):
    """Validation and test accuracy of a full-batch linear probe at every depth."""
    val, test = {}, {}
    for d, X in feats.items():
        xtr = X[rows["train"]]
        mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-4)
        xs = {s: ((X[rows[s]] - mu) / sd).to(device) for s in rows}
        ys = {s: y[s].to(device) for s in rows}
        torch.manual_seed(0)
        lin = torch.nn.Linear(X.shape[1], n_labels).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
        for _ in range(steps):
            loss = F.cross_entropy(lin(xs["train"]), ys["train"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val[d] = float((lin(xs["val"]).argmax(-1) == ys["val"]).float().mean())
            test[d] = float((lin(xs["test"]).argmax(-1) == ys["test"]).float().mean())
    return val, test


def evaluate_plan(requests: List[Tuple[Tuple[str, ...], float]], planned: Dict[str, int], val, test, depths,
                  promote: bool):
    """Expected trunk depth and mean val/test accuracy over the workload."""
    E_depth, acc_v, acc_t = 0.0, 0.0, 0.0
    for tasks, w in requests:
        D = max(planned[t] for t in tasks)
        av, at = [], []
        for t in tasks:
            d = max((x for x in depths if x <= D), key=lambda x: (val[t][x], -x)) if promote else planned[t]
            av.append(val[t][d])
            at.append(test[t][d])
        E_depth += w * D
        acc_v += w * np.mean(av)
        acc_t += w * np.mean(at)
    return E_depth, acc_v, acc_t


def plan_joint(requests, val, test, depths, start: Dict[str, int], target: float):
    """Greedy: repeatedly lower one task's planned split (to its next candidate) if the promoted workload
    validation accuracy stays >= target; take the move that cuts expected depth most."""
    planned = dict(start)
    while True:
        E0, _, _ = evaluate_plan(requests, planned, val, test, depths, True)
        best = None
        for t in planned:
            lower = [d for d in depths if d < planned[t]]
            if not lower:
                continue
            cand = dict(planned)
            cand[t] = max(lower)
            E, av, _ = evaluate_plan(requests, cand, val, test, depths, True)
            if av >= target and E < E0 and (best is None or E < best[0]):
                best = (E, cand)
        if best is None:
            return planned
        planned = best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["typed-decisions", "clinc150"])
    ap.add_argument("--depths", type=int, nargs="*", default=[2, 4, 6, 8, 10, 11, 12, 14, 16, 18, 20, 22])
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.depths, a.steps, a.datasets = "cpu", [2, 4, 6], 30, ["clinc150"]
    if not a.smoke and not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    trunk = Trunk(device=a.device)
    res = {"depths": a.depths, "tol": a.tol, "sets": {}}
    for name in a.datasets:
        ds = data.load(name)
        if a.smoke:
            ds = subset(ds, list(ds.tasks), 300, 100, 100, stride=13)
        feats = pooled_by_depth(trunk, all_texts(ds), a.depths, ds.max_len)
        allx, rng = split_index(ds)
        val, test = {}, {}
        for t in ds.tasks:
            rows = {s: [i for i in rng[s] if t in allx[i].y] for s in rng}
            y = {s: torch.tensor([allx[i].y[t] for i in rows[s]]) for s in rows}
            val[t], test[t] = probe_curve(feats, rows, y, len(ds.tasks[t].labels), trunk.device, a.steps)
        # workloads
        if name == "typed-decisions":
            groups = {}
            for t in ds.tasks:
                groups.setdefault(t.split(".")[0], []).append(t)
            wl = {"workflow": [(tuple(ts), 1.0 / len(groups)) for ts in groups.values()]}
            subs = [(c, 1.0) for ts in groups.values() for r in range(1, len(ts) + 1) for c in itertools.combinations(ts, r)]
            wl["subsets"] = [(c, w / len(subs)) for c, w in subs]
        else:
            ts = list(ds.tasks)
            subs = [c for r in range(1, len(ts) + 1) for c in itertools.combinations(ts, r)]
            wl = {"all": [(tuple(ts), 1.0)], "subsets": [(c, 1.0 / len(subs)) for c in subs]}
        best_d = {t: max(a.depths, key=lambda d: (val[t][d], -d)) for t in ds.tasks}
        auto_d = {t: min(d for d in a.depths if val[t][d] >= max(val[t].values()) - a.tol) for t in ds.tasks}
        top = {t: max(a.depths) for t in ds.tasks}
        entry = {"curves_val": val, "curves_test": test, "best_split": best_d, "autosplit": auto_d, "workloads": {}}
        log(f"== {ds.name}: per-task best split {best_d} | autosplit {auto_d}")
        for wname, reqs in wl.items():
            out = {}
            for pname, planned, promote in (("autosplit", auto_d, False), ("autosplit+promote", auto_d, True),
                                            ("best", best_d, False), ("best+promote", best_d, True),
                                            ("all@top", top, True)):
                out[pname] = dict(zip(("E_depth", "val_acc", "test_acc"), evaluate_plan(reqs, planned, val, test,
                                                                                        a.depths, promote)))
            target = out["best+promote"]["val_acc"] - a.tol
            joint = plan_joint(reqs, val, test, a.depths, best_d, target)
            out["joint"] = dict(zip(("E_depth", "val_acc", "test_acc"),
                                    evaluate_plan(reqs, joint, val, test, a.depths, True)))
            out["joint"]["planned"] = joint
            entry["workloads"][wname] = out
            for pname, r in out.items():
                log(f"   [{wname}] {pname:18s} E[trunk depth] {r['E_depth']:5.2f}  val {r['val_acc']:.4f}  test {r['test_acc']:.4f}")
        res["sets"][name] = entry
        save(res, a.out)
    save(res, a.out)
    log("done")


if __name__ == "__main__":
    main()
