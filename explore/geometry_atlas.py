"""Idea 7: a decision atlas. Do decisions with the same meaning share a trunk direction across schemas?

The depth scan tested this zero-shot for urgency: a ridge regression on one typed-decisions workflow's
expected urgency, applied to another workflow's states, gives Spearman 0.05-0.09 across workflows
(0.68 within). This script finishes the test in the setting where an atlas would actually be used, a new
workflow with a handful of labels, for three concept groups:

  urgency   the four `*.urgency` score questions (identical option texts, different schemas)
  severity  agent_trace.risk, invoice.discrepancy_severity, security.severity, customer_service.churn_risk
            (ordinal; expected level scaled to [0, 1])
  human     agent_trace.needs_review, customer_service.needs_human (P(true))

Per target task and n in {8, 16, 32, 64} labelled target messages (20 random draws each):
  target-only   ridge on the n target examples (L2 chosen on the target validation split)
  atlas prior   ridge shrunk towards a*w_src instead of 0, where w_src is fitted on the same concept in the
                other workflows (all training messages) and a is fitted on the n examples
  calibrated    y ~ a * (x . w_src) + b fitted on the n examples (the atlas direction alone)
  zero-shot     x . w_src (rank correlation needs no calibration)
Metric: Spearman correlation with the target's expected level on its test split (AUROC-equivalent ranking
for the binary group), at depths 11 and 22.

Prediction (geometry.md idea 7): if a shared direction exists, the atlas prior beats target-only at
n <= 16 by >= 0.05 Spearman; the zero-shot result (~0.07) says it probably does not.

Usage:
  .venv/bin/python explore/geometry_atlas.py --smoke
  .venv/bin/python explore/geometry_atlas.py --out results/tarski/explore_atlas.json    # ~2 min on an A10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_common import Logger, load_dataset, pooled, splits, standardize, task_split

import numpy as np
import torch
from scipy.stats import spearmanr

from tarski.trunk import Trunk

GROUPS = {
    "urgency": ["agent_trace_observability.urgency", "customer_service.urgency", "invoice_processing.urgency",
                "security_incidents.urgency"],
    "severity": ["agent_trace_observability.risk", "invoice_processing.discrepancy_severity",
                 "security_incidents.severity", "customer_service.churn_risk"],
    "human": ["agent_trace_observability.needs_review", "customer_service.needs_human"],
}
LAMBDAS = (1.0, 10.0, 100.0, 1000.0)


def level(allx, idx, t):
    """Expected level in [0, 1] from the soft labels (binary: P(true))."""
    out = []
    for i in idx:
        p = allx[i].soft[t]
        out.append(float(np.dot(p, np.arange(len(p)))) / (len(p) - 1))
    return np.array(out)


def ridge(X, y, lam, w0=None):
    """argmin ||Xw + b - y||^2 + lam ||w - w0||^2 in dual form (n << D); returns (w, b)."""
    X = X.double()
    y = torch.as_tensor(y, dtype=torch.float64)
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w0 = torch.zeros(X.shape[1], dtype=torch.float64) if w0 is None else w0
    alpha = torch.linalg.solve(Xc @ Xc.T + lam * torch.eye(len(X), dtype=torch.float64), yc - Xc @ w0)
    w = w0 + Xc.T @ alpha
    return w, float(ym - xm @ w)


def rho(a, b):
    r = spearmanr(a, b).correlation
    return 0.0 if r is None or np.isnan(r) else float(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--depths", type=int, nargs="*", default=[11, 22])
    ap.add_argument("--ns", type=int, nargs="*", default=[8, 16, 32, 64])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.depths, args.ns, args.draws = [11], [8], 3
    out_path = args.out or ("results/tarski/explore_atlas_smoke.json" if args.smoke else "results/tarski/explore_atlas.json")
    log = Logger(out_path)
    trunk = Trunk(device="cpu" if args.smoke else None)
    ds = load_dataset("typed-decisions", args.smoke)
    allx, n_tr, n_va = splits(ds)
    log(f"== decision atlas | {trunk.device} | {ds.summary()[:80]} | n {args.ns} x {args.draws} draws")
    feats = pooled(trunk, ds, args.depths)
    rng = np.random.default_rng(0)
    res = {"args": vars(args), "depths": {}}
    for d in args.depths:
        X = standardize(feats[d], list(range(n_tr))).double()
        res["depths"][d] = {}
        for gname, tasks in GROUPS.items():
            for tgt in tasks:
                srcs = [t for t in tasks if t != tgt]
                # source direction: the concept fitted on every other workflow's training messages
                Xs = torch.cat([X[task_split(allx, s, n_tr, n_va)[0]] for s in srcs])
                ys = np.concatenate([level(allx, task_split(allx, s, n_tr, n_va)[0], s) for s in srcs])
                lam_src = 100.0
                w_src, _ = ridge(Xs, ys, lam_src)
                tr, va, te = task_split(allx, tgt, n_tr, n_va)
                yte, yva = level(allx, te, tgt), level(allx, va, tgt)
                zs = rho((X[te] @ w_src).numpy(), yte)
                row = {"zero_shot": zs, "n": {}}
                for n in args.ns:
                    if n > len(tr):
                        continue
                    acc = {"target_only": [], "atlas_prior": [], "calibrated": []}
                    for _ in range(args.draws):
                        pick = rng.choice(len(tr), n, replace=False)
                        Xn, yn = X[[tr[i] for i in pick]], level(allx, [tr[i] for i in pick], tgt)
                        if np.std(yn) < 1e-9:
                            continue
                        best = max(LAMBDAS, key=lambda l: rho((X[va] @ ridge(Xn, yn, l)[0]).numpy(), yva))
                        acc["target_only"].append(rho((X[te] @ ridge(Xn, yn, best)[0]).numpy(), yte))
                        proj = (Xn @ w_src).numpy()
                        a = float(np.polyfit(proj, yn, 1)[0]) if np.std(proj) > 1e-9 else 0.0
                        best_p = max(LAMBDAS, key=lambda l: rho((X[va] @ ridge(Xn, yn, l, a * w_src)[0]).numpy(), yva))
                        acc["atlas_prior"].append(rho((X[te] @ ridge(Xn, yn, best_p, a * w_src)[0]).numpy(), yte))
                        acc["calibrated"].append(rho(np.sign(a) * (X[te] @ w_src).numpy(), yte))
                    row["n"][n] = {k: float(np.mean(v)) if v else None for k, v in acc.items()}
                res["depths"][d].setdefault(gname, {})[tgt] = row
                log(f"   d={d} {gname:>8} -> {tgt:<42} zero-shot {zs:+.3f} | " + " | ".join(
                    f"n={n}: target {v['target_only']:+.3f} atlas {v['atlas_prior']:+.3f} calib {v['calibrated']:+.3f}"
                    for n, v in row["n"].items() if v["target_only"] is not None))
        # summary: mean over targets
        summ = {}
        for gname in GROUPS:
            rows = res["depths"][d][gname].values()
            summ[gname] = {"zero_shot": float(np.mean([r["zero_shot"] for r in rows]))}
            for n in args.ns:
                vals = [r["n"][n] for r in rows if n in r["n"] and r["n"][n]["target_only"] is not None]
                if vals:
                    summ[gname][n] = {k: float(np.mean([v[k] for v in vals])) for k in ("target_only", "atlas_prior", "calibrated")}
        res["depths"][d]["summary"] = summ
        log(f"-- depth {d} summary: " + json.dumps(summ, default=lambda x: round(x, 3)))
        json.dump(res, open(out_path, "w"), indent=1, default=float)
    log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
