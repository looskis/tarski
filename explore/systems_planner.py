"""Idea 5: semantic query optimisation for decisions.

Three plans over the same trained branches, simulated on the test set from cached calibrated
probabilities (costs are counted in layer-equivalents: trunk layers run + 2 per blocks:2 branch run):

  all-blocks   trunk to 11, every requested decision through its blocks@11+2 branch (the baseline).
  fd           trunk to 11, run the root decision (intent); derive the decisions it functionally
               determines from a table learned on co-labelled training data, when the root is confident
               (p >= tau) and the table entry is pure (>= 0.99); otherwise run their branches.
  exit         conjunctive early exit: trunk to 4, probes@4 for every requested decision; if every one is
               confident (p >= tau), answer there; otherwise resume the trunk to 11 and run all-blocks.
  exit+fd      trunk to 4, root probe; if confident, answer the root and derive the rest; otherwise resume to
               11 and run fd.

Also reports the functional-dependency structure among typed-decisions' co-labelled tasks (from labels
only, no training): how much a planner could skip there in principle.

Tested prediction: 30-50% fewer layer-equivalents at < 0.5 points accuracy loss on CLINC150.

Usage:
  python explore/systems_planner.py --smoke
  python explore/systems_planner.py --out results/tarski/explore_planner.json
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explore.systems_common import Log, all_texts, save, subset, threads, train_tasks

import numpy as np

from tarski import data
from tarski.train import FeatureCache
from tarski.trunk import Trunk


def fd_table(examples, a: str, b: str):
    """For each value of task a: the most common value of task b and its purity, from co-labelled rows."""
    c = defaultdict(Counter)
    for e in examples:
        if a in e.y and b in e.y:
            c[e.y[a]][e.y[b]] += 1
    table, purity, n = {}, {}, {}
    for va, cnt in c.items():
        vb, m = cnt.most_common(1)[0]
        table[va], purity[va], n[va] = vb, m / sum(cnt.values()), sum(cnt.values())
    total = sum(n.values())
    weighted = sum(purity[v] * n[v] for v in n) / max(1, total)
    return table, purity, weighted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="clinc150")
    ap.add_argument("--root", default="intent")
    ap.add_argument("--shallow", type=int, default=4)
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--taus", type=float, nargs="*", default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995])
    ap.add_argument("--purity", type=float, default=0.99)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--probe-epochs", type=int, default=10)
    ap.add_argument("--block-epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.shallow, a.split, a.min_steps, a.probe_epochs, a.block_epochs = "cpu", 2, 4, 5, 1, 1
    if not a.smoke and not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    trunk = Trunk(device=a.device)
    ds = data.load(a.dataset)
    if a.smoke:
        ds = subset(ds, list(ds.tasks), 150, 40, 60, stride=29)
    tasks = list(ds.tasks)
    others = [t for t in tasks if t != a.root]
    cache = FeatureCache(trunk, all_texts(ds), [a.shallow, a.split], ds.max_len)
    log(f"== {ds.summary()} | probes@{a.shallow}, blocks@{a.split}+2 | cached in {cache.seconds:.1f}s")
    shallow = train_tasks(trunk, ds, cache, tasks, "probe", a.shallow, seed=a.seed, epochs=a.probe_epochs,
                          min_steps=a.min_steps, log=log)
    deep = train_tasks(trunk, ds, cache, tasks, "blocks:2", a.split, seed=a.seed, epochs=a.block_epochs,
                       min_steps=a.min_steps, log=log)
    y = {t: deep[t]["y"]["test"].numpy() for t in tasks}
    P = {("s", t): shallow[t]["test_probs"] for t in tasks}
    P.update({("d", t): deep[t]["test_probs"] for t in tasks})
    n = len(y[a.root])
    fds = {b: fd_table(ds.train, a.root, b) for b in others}
    res = {"dataset": ds.name, "root": a.root, "shallow": a.shallow, "split": a.split,
           "fd_weighted_purity": {b: fds[b][2] for b in others},
           "branch_acc": {f"probe@{a.shallow}": {t: shallow[t]["metrics"]["acc"] for t in tasks},
                          f"blocks@{a.split}+2": {t: deep[t]["metrics"]["acc"] for t in tasks}}, "plans": {}}
    log(f"  FD purity {a.root} -> " + ", ".join(f"{b}: {fds[b][2]:.4f}" for b in others))

    def derive(root_pred, b):
        table, purity, _ = fds[b]
        vb = np.array([table.get(int(v), -1) for v in root_pred])
        pure = np.array([purity.get(int(v), 0.0) >= a.purity for v in root_pred])
        return vb, pure

    def score(pred: Dict[str, np.ndarray], cost: np.ndarray, extra: Dict):
        acc = {t: float((pred[t] == y[t]).mean()) for t in tasks}
        return {"mean_acc": float(np.mean(list(acc.values()))), "acc": acc, "layer_equiv": float(cost.mean()), **extra}

    trunk_cost = {a.shallow: a.shallow, a.split: a.split}
    # baseline: all blocks
    base_pred = {t: P[("d", t)].argmax(-1) for t in tasks}
    base_cost = np.full(n, a.split + 2.0 * len(tasks))
    res["plans"]["all-blocks"] = score(base_pred, base_cost, {})
    log(f"-- all-blocks: acc {res['plans']['all-blocks']['mean_acc']:.4f}, layer-equiv {base_cost.mean():.1f}")

    for tau in a.taus:
        # fd at depth split
        pr = P[("d", a.root)]
        root = pr.argmax(-1)
        conf = pr.max(-1) >= tau
        pred, cost = {a.root: root}, np.full(n, a.split + 2.0)
        for b in others:
            vb, pure = derive(root, b)
            use = conf & pure & (vb >= 0)
            pred[b] = np.where(use, vb, P[("d", b)].argmax(-1))
            cost = cost + np.where(use, 0.0, 2.0)
        res["plans"][f"fd@tau={tau}"] = score(pred, cost, {})
        # conjunctive exit
        conf_all = np.all([P[("s", t)].max(-1) >= tau for t in tasks], 0)
        pred = {t: np.where(conf_all, P[("s", t)].argmax(-1), base_pred[t]) for t in tasks}
        cost = np.where(conf_all, float(a.shallow), a.split + 2.0 * len(tasks))
        res["plans"][f"exit@tau={tau}"] = score(pred, cost, {"exit_rate": float(conf_all.mean())})
        # exit + fd: an early message pays the shallow trunk; any decision it cannot derive forces the rest
        # of the trunk (once) plus that decision's deep branch. A late message runs fd at depth `split`.
        ps = P[("s", a.root)]
        root_s = ps.argmax(-1)
        early = ps.max(-1) >= tau
        pred = {a.root: np.where(early, root_s, root)}
        cost = np.where(early, float(a.shallow), a.split + 2.0)
        resume = np.zeros(n, bool)
        for b in others:
            vb_s, pure_s = derive(root_s, b)
            vb_d, pure_d = derive(root, b)
            ok_s = early & pure_s & (vb_s >= 0)
            ok_d = ~early & conf & pure_d & (vb_d >= 0)
            pred[b] = np.where(ok_s, vb_s, np.where(ok_d, vb_d, P[("d", b)].argmax(-1)))
            cost = cost + np.where(ok_s | ok_d, 0.0, 2.0)
            resume |= early & ~ok_s
        cost = cost + np.where(resume, float(a.split - a.shallow), 0.0)
        res["plans"][f"exit+fd@tau={tau}"] = score(pred, cost, {"exit_rate": float(early.mean())})
        log(f"   tau {tau}: fd {res['plans'][f'fd@tau={tau}']['mean_acc']:.4f}/{res['plans'][f'fd@tau={tau}']['layer_equiv']:.1f}"
            f" | exit {res['plans'][f'exit@tau={tau}']['mean_acc']:.4f}/{res['plans'][f'exit@tau={tau}']['layer_equiv']:.1f}"
            f" (exit {conf_all.mean():.2f}) | exit+fd {res['plans'][f'exit+fd@tau={tau}']['mean_acc']:.4f}/"
            f"{res['plans'][f'exit+fd@tau={tau}']['layer_equiv']:.1f} (exit {early.mean():.2f})   [acc/layer-equiv]")
    save(res, a.out)

    # functional dependencies among typed-decisions tasks, from labels only
    try:
        td = data.load("typed-decisions")
        fd = {}
        for wf in sorted({t.split(".")[0] for t in td.tasks}):
            ts = [t for t in td.tasks if t.startswith(wf + ".")]
            for x, z in itertools.permutations(ts, 2):
                _, _, w = fd_table(td.train, x, z)
                base = Counter(e.y[z] for e in td.train if z in e.y).most_common(1)[0][1] / \
                    sum(1 for e in td.train if z in e.y)
                fd[f"{x} -> {z}"] = {"purity": w, "majority_baseline": base}
        top = sorted(fd.items(), key=lambda kv: -(kv[1]["purity"] - kv[1]["majority_baseline"]))[:12]
        res["typed_fd"] = fd
        log("  typed-decisions strongest dependencies (purity vs majority baseline):")
        for kname, v in top:
            log(f"    {kname}: {v['purity']:.3f} vs {v['majority_baseline']:.3f}")
        res["typed_fd_ge_0.95"] = sum(1 for v in fd.values() if v["purity"] >= 0.95)
    except Exception as e:  # typed-decisions not cached: the CLINC part still stands
        log(f"  typed-decisions FD analysis skipped: {e}")
    save(res, a.out)
    log("== summary (best plan within 0.5 points of all-blocks, by layer-equivalents)")
    b0 = res["plans"]["all-blocks"]
    ok = [(k_, v) for k_, v in res["plans"].items() if v["mean_acc"] >= b0["mean_acc"] - 0.005]
    for k_, v in sorted(ok, key=lambda kv: kv[1]["layer_equiv"])[:5]:
        log(f"  {k_:22s} acc {v['mean_acc']:.4f} layer-equiv {v['layer_equiv']:.2f} "
            f"({1 - v['layer_equiv'] / b0['layer_equiv']:.0%} saved)")
    log("done")


if __name__ == "__main__":
    main()
