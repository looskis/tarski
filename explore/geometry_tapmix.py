"""Idea 8: cost-aware tap mixing. Let each decision learn how deep the trunk must run.

The depth scan found that a probe on the concatenation of taps 6, 11, 16 and 22 beats the best single-depth
probe (Banking77 89.7 vs ~85). Reading several taps is free in tarski (one trunk pass), but the deepest
tap any requested decision reads sets how far the trunk runs for that message. So the useful quantity
is the accuracy you can buy at a given trunk cut, and whether a branch can *learn* its own cut.

Per task, on mean-pooled trunk states at taps T = {2, 4, ..., 22} (standardized per tap):
  single      logistic regression on one tap (the tarski probe family), best tap chosen on validation
  prefix@D    logistic regression on the concatenation of all taps <= D (the best a mixer can do with the
              trunk cut at D), for every D
  glasso(l)   logistic regression on all taps with a cost-weighted group-lasso penalty
              l * sum_d (d/22) * ||W_d||_F. Deep taps pay more, so their weight blocks shrink unless they
              earn their cost. Taps whose block norm falls below 10% of the largest are dropped, the
              survivors are refitted without the penalty, and the deepest survivor is the learned trunk cut
  select(l)   cost-aware selection on the prefix curve: argmin_D (1 - val_acc(D)) + l * D/22

Serving view: a message's trunk cut is the max over the decisions asked about it (typed-decisions: the five
of its workflow; CLINC: intent + domain + oos). Reported per method: mean test accuracy over tasks and mean
cut over messages (22 = full trunk).

Prediction (docs/research/notes/explore/geometry.md, idea 8): on Banking77/CLINC a prefix mixer cut at <= 12
keeps >= 80% of the full multi-tap gain over the best single tap, and the group lasso finds cuts <= 14
within 1 point of the full multi-tap.

Usage:
  .venv/bin/python explore/geometry_tapmix.py --smoke
  .venv/bin/python explore/geometry_tapmix.py --out results/tarski/explore_tapmix.json    # ~10-15 min on an A10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_common import (Logger, acc, device, fit_logreg, labels, load_dataset, pick_by_val, pooled, splits,
                             standardize, targets, task_split)

import numpy as np
import torch

from tarski.trunk import Trunk


def groups_of(allx, tasks):
    sig = {}
    for t in tasks:
        sig.setdefault(tuple(i for i, e in enumerate(allx) if t in e.y), []).append(t)
    return [{"tasks": ts, "n_msgs": len(k)} for k, ts in sig.items()]


def run_task(Fs, taps, allx, task, C, n_tr, n_va, dev, lambdas, steps, sel_lambdas, thr=0.1):
    tr, va, te = task_split(allx, task, n_tr, n_va)
    T = targets(allx, tr, task, C)
    yva, yte = labels(allx, va, task), labels(allx, te, task)
    out = {"single": {}, "prefix": {}, "glasso": {}, "select": {}}
    for d in taps:
        r = fit_logreg(Fs[d][tr], T, [Fs[d][va], Fs[d][te]], dev, steps=steps)
        g = pick_by_val(r["logits"][0], yva)
        out["single"][d] = {"val": acc(r["logits"][0][g], yva), "test": acc(r["logits"][1][g], yte)}
    best_l2 = 1e-3
    for j, D in enumerate(taps):
        X = torch.cat([Fs[d] for d in taps[: j + 1]], -1)
        r = fit_logreg(X[tr], T, [X[va], X[te]], dev, steps=steps)
        g = pick_by_val(r["logits"][0], yva)
        out["prefix"][D] = {"val": acc(r["logits"][0][g], yva), "test": acc(r["logits"][1][g], yte)}
        if D == taps[-1]:
            best_l2 = (1e-4, 1e-3, 1e-2)[g]
    X = torch.cat([Fs[d] for d in taps], -1)
    H = Fs[taps[0]].shape[1]
    cost = torch.tensor([d / taps[-1] for d in taps], device=dev)
    lam = torch.tensor(lambdas, device=dev)

    def penalty(W):                        # W: (G, n_taps*H, C); group = one tap's block of rows
        G, _, Cc = W.shape
        norm = W.view(G, len(taps), H * Cc).pow(2).sum(-1).add(1e-12).sqrt()     # (G, n_taps)
        return (lam[:, None] * cost[None] * norm).sum()

    r = fit_logreg(X[tr], T, [X[va], X[te]], dev, l2s=[best_l2] * len(lambdas), steps=steps, penalty=penalty)
    Wn = r["W"].view(len(lambdas), len(taps), -1).norm(dim=-1)                  # (G, n_taps)
    for g, l in enumerate(lambdas):
        # taps whose weight block is below `thr` of the largest block are dropped; the survivors are refitted
        # without the penalty (debiasing), and the deepest survivor is the learned trunk cut
        active = [d for d, n in zip(taps, Wn[g].tolist()) if n > thr * float(Wn[g].max())]
        Xa = torch.cat([Fs[d] for d in active], -1)
        ra = fit_logreg(Xa[tr], T, [Xa[va], Xa[te]], dev, l2s=[best_l2], steps=steps)
        out["glasso"][l] = {"val": acc(ra["logits"][0][0], yva), "test": acc(ra["logits"][1][0], yte),
                            "penalized_test": acc(r["logits"][1][g], yte),
                            "cut": max(active), "active_taps": active,
                            "block_norms": [round(x, 4) for x in Wn[g].tolist()]}
    for l in sel_lambdas:
        D = min(taps, key=lambda D: (1 - out["prefix"][D]["val"]) + l * D / taps[-1])
        out["select"][l] = {"test": out["prefix"][D]["test"], "cut": D}
    ds_best = max(taps, key=lambda d: out["single"][d]["val"])
    out["single_best"] = {"test": out["single"][ds_best]["test"], "cut": ds_best}
    return out


def summarize(per, groups, taps, lambdas, sel_lambdas):
    """Mean test accuracy over tasks and message-weighted mean trunk cut (max over a message's tasks)."""
    def serve(get):
        accs = [get(t)[0] for t in per]
        n = sum(g["n_msgs"] for g in groups)
        cut = sum(g["n_msgs"] * max(get(t)[1] for t in g["tasks"]) for g in groups) / n
        return {"acc": float(np.mean(accs)), "mean_cut": float(cut)}
    s = {"single_best": serve(lambda t: (per[t]["single_best"]["test"], per[t]["single_best"]["cut"])),
         "single@22": serve(lambda t: (per[t]["single"][taps[-1]]["test"], taps[-1]))}
    for D in taps:
        s[f"prefix@{D}"] = serve(lambda t, D=D: (per[t]["prefix"][D]["test"], D))
    for l in lambdas:
        s[f"glasso({l})"] = serve(lambda t, l=l: (per[t]["glasso"][l]["test"], per[t]["glasso"][l]["cut"]))
    for l in sel_lambdas:
        s[f"select({l})"] = serve(lambda t, l=l: (per[t]["select"][l]["test"], per[t]["select"][l]["cut"]))
    full, single = s[f"prefix@{taps[-1]}"]["acc"], s["single_best"]["acc"]
    s["gain_kept_at_cut12"] = float((s[f"prefix@{max(d for d in taps if d <= 12)}"]["acc"] - single) / (full - single)) \
        if full > single else None
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150", "typed-decisions"])
    ap.add_argument("--taps", type=int, nargs="*", default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    ap.add_argument("--lambdas", type=float, nargs="*", default=[0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--select-lambdas", type=float, nargs="*", default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    taps = [4, 8, 12, 16, 22] if args.smoke else sorted(args.taps)
    steps = 40 if args.smoke else max(300, args.steps)
    out_path = args.out or ("results/tarski/explore_tapmix_smoke.json" if args.smoke else "results/tarski/explore_tapmix.json")
    log = Logger(out_path)
    dev = device(args.smoke)
    trunk = Trunk(device="cpu" if args.smoke else None)
    log(f"== tap mixing | {trunk.device} / fits on {dev} | taps {taps} | lambdas {args.lambdas} | steps {steps}")
    res = {"args": vars(args), "taps": taps, "datasets": {}}
    for name in args.datasets:
        t0 = time.time()
        ds = load_dataset(name, args.smoke)
        allx, n_tr, n_va = splits(ds)
        feats = pooled(trunk, ds, taps)
        Fs = {d: standardize(feats[d], list(range(n_tr))) for d in taps}
        tasks = list(ds.tasks)
        per = {}
        for t in tasks:
            per[t] = run_task(Fs, taps, allx, t, len(ds.tasks[t].labels), n_tr, n_va, dev, args.lambdas, steps,
                              args.select_lambdas)
        groups = groups_of(allx, tasks)
        s = summarize(per, groups, taps, args.lambdas, args.select_lambdas)
        res["datasets"][name] = {"tasks": per, "summary": s, "wall_s": round(time.time() - t0, 1)}
        log(f"-- {name} ({len(tasks)} tasks, {time.time() - t0:.0f}s)")
        for k, v in s.items():
            if isinstance(v, dict):
                log(f"   {k:>14}: acc {v['acc']:.4f}  mean trunk cut {v['mean_cut']:.1f}")
        log(f"   share of the full multi-tap gain kept with the trunk cut at 12: {s['gain_kept_at_cut12']}")
        json.dump(res, open(out_path, "w"), indent=1)
    log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
