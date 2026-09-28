"""Entry 6 (lit_incremental_oos.md): conformal clarification sets for routing, with per-route coverage and an
open-set gate; the middle band goes to laya.

  - den Hengst et al., "Conformal Intent Classification and Clarification" (CICC, NAACL Findings 2024,
    arXiv 2403.18973): one route in the conformal set -> act; 2..th routes (th = 7) -> ask a clarifying
    question over the set; more -> escalate.
  - Ding, Angelopoulos, Bates, Jordan & Tibshirani, "Class-Conditional Conformal Prediction with Many
    Classes" (clustered conformal, NeurIPS 2023, arXiv 2306.09335): pool routes whose score distributions
    look alike, so per-route coverage holds with 10-20 calibration messages per route.
  - Open set. Xie et al., "Conformal Inference for Open-Set and Imbalanced Classification" (arXiv
    2510.13037, 2025) add an unseen-label "joker" via Good-Turing p-values. Good-Turing estimates the
    unseen-class probability from the number of labels seen exactly once; a routing dataset has no such
    singletons (every route has dozens of examples), so that estimator is degenerate here and is NOT
    implemented. Instead a conformal outlier gate (Bates et al., "Testing for outliers with conformal
    p-values", Ann. Stat. 2023) sends a message to NEW-ROUTE when its Mahalanobis++ p-value against
    known-route calibration messages is <= alpha_o; the guarantee is on known messages (wrongly gated
    <= alpha_o), not on recall of new routes.

Probe: a calibrated linear probe at --depth trained on 90% of train; its L2 and temperature come from the
other 10%, so the whole validation set stays a clean conformal calibration set.
Arms (alpha = 0.05 and 0.10; LAC and APS scores):
  marginal     split conformal
  classwise    one quantile per route
  clustered    Ding et al.: half of calibration clusters routes by score quantiles (k-means, M clusters),
               the other half gives one quantile per cluster (10 random splits)
Metrics on test (in-scope): marginal coverage, CovGap (mean |route coverage - (1 - alpha)|), worst-decile
route coverage, mean set size, share in each band (1 / 2..th / more or empty).
Route-or-clarify (LAC, alpha 0.05, marginal and clustered): accuracy of automated decisions when the middle
band is resolved by (a) the probe's top-1 (no clarification), (b) laya choosing among the set's routes,
(c) an oracle user who picks the right route if it is in the set (CICC's assumption); escalation share.
Open-set (CLINC): 30 intents held out of training; share of held-out-intent and out-of-scope messages gated
to NEW-ROUTE, and the share of known messages wrongly gated.

Usage:
  .venv/bin/python explore/lit_oos_cicc.py --smoke
  .venv/bin/python explore/lit_oos_cicc.py --out results/tarski/explore_lit_cicc.json   # ~10 min on an A10 (laya on ~1.5k messages)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_oos_common import (Gauss, Logger, Probe, get_trunk, humanize, l2n, load_ds, pick_device, pooled, seed_all,
                            split_index, zscore)

import numpy as np
from sklearn.cluster import KMeans

TH = 7


def lac_scores(P: np.ndarray) -> np.ndarray:
    return 1 - P                                           # score of every candidate label


def aps_scores(P: np.ndarray, rng) -> np.ndarray:
    """Randomised APS score of every candidate label: mass of labels ranked above it + U * its own mass."""
    order = np.argsort(-P, 1)
    Ps = np.take_along_axis(P, order, 1)
    cum = np.cumsum(Ps, 1) - Ps
    U = rng.random((len(P), 1))
    s_sorted = cum + U * Ps
    S = np.empty_like(P)
    np.put_along_axis(S, order, s_sorted, 1)
    return S


def qhat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return np.inf
    k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def evaluate_sets(sets: np.ndarray, y: np.ndarray, C: int, alpha: float) -> dict:
    cover = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    covs = np.array([cover[y == c].mean() for c in range(C) if (y == c).any()])
    worst = np.sort(covs)[: max(1, len(covs) // 10)].mean()
    return {"coverage": float(cover.mean()), "covgap": float(np.mean(np.abs(covs - (1 - alpha))) * 100),
            "worst_decile_cov": float(worst), "mean_size": float(size.mean()),
            "band_single": float((size == 1).mean()), "band_clarify": float(((size >= 2) & (size <= TH)).mean()),
            "band_escalate": float(((size > TH) | (size == 0)).mean())}


def clustered_thresholds(s_cal_true: np.ndarray, y_cal: np.ndarray, C: int, alpha: float, M: int, rng,
                         frac: float = 0.5, levels=(0.5, 0.7, 0.9), n_min: int = 3) -> np.ndarray:
    """Per-class thresholds from clustered conformal (Ding et al. 2023)."""
    perm = rng.permutation(len(y_cal))
    n1 = int(frac * len(perm))
    a, b = perm[:n1], perm[n1:]
    emb, cls = [], []
    for c in range(C):
        sc = s_cal_true[a][y_cal[a] == c]
        if len(sc) >= n_min:
            emb.append(np.quantile(sc, levels))
            cls.append(c)
    assign = np.full(C, -1)
    if len(cls) >= M:
        km = KMeans(M, n_init=5, random_state=int(rng.integers(1 << 30))).fit(np.array(emb))
        assign[cls] = km.labels_
    th = np.full(C, qhat(s_cal_true[b], alpha))          # null cluster: the marginal quantile
    for m in range(M):
        members = np.where(assign == m)[0]
        sc = s_cal_true[b][np.isin(y_cal[b], members)]
        if len(sc) > 0:
            th[members] = qhat(sc, alpha)
    return th


def run_laya(agent, texts, sets, labels, y, max_n, log):
    """laya chooses among the set's routes; returns (#asked, #correct) and per-message picks."""
    idx = np.where((sets.sum(1) >= 2) & (sets.sum(1) <= TH))[0][:max_n]
    picks = {}
    t0 = time.time()
    for i in idx:
        opts = [int(c) for c in np.where(sets[i])[0]]
        q = {"route": {"type": "choice", "instructions": "Which of these does the user's message ask for?",
                       "criteria": {humanize(labels[c]): None for c in opts}}}
        pr = agent.system_one(texts[i], q)["answers"]["route"]["probabilities"]
        names = [humanize(labels[c]) for c in opts]
        picks[int(i)] = opts[int(np.argmax([pr[nm] for nm in names]))]
    log(f"   laya answered {len(idx)} clarification questions in {time.time() - t0:.0f}s")
    return picks


def route_or_clarify(sets, P, y, picks) -> dict:
    size = sets.sum(1)
    single = size == 1
    mid = (size >= 2) & (size <= TH)
    esc = ~(single | mid)
    top1 = P.argmax(1)
    auto_single = (top1 == y)[single].sum()
    out = {"share_single": float(single.mean()), "share_clarify": float(mid.mean()), "share_escalate": float(esc.mean())}
    n_auto = single.sum() + mid.sum()
    out["acc_automated_top1"] = float((auto_single + (top1 == y)[mid].sum()) / max(1, n_auto))
    out["acc_automated_oracle_user"] = float((auto_single + sets[np.arange(len(y)), y][mid].sum()) / max(1, n_auto))
    if picks:
        mids = [i for i in np.where(mid)[0] if int(i) in picks]
        lay = sum(picks[int(i)] == y[i] for i in mids)
        top = sum(top1[i] == y[i] for i in mids)
        out["laya_questions"] = len(mids)
        out["clarify_band_acc_laya"] = float(lay / max(1, len(mids)))
        out["clarify_band_acc_top1_same_msgs"] = float(top / max(1, len(mids)))
        out["clarify_band_acc_oracle_same_msgs"] = float(np.mean([sets[i, y[i]] for i in mids])) if mids else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--depth", type=int, default=22)
    ap.add_argument("--gate-depth", type=int, default=4)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.05, 0.10])
    ap.add_argument("--route-alpha", type=float, default=0.05)
    ap.add_argument("--gate-alpha", type=float, default=0.05)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--laya-max", type=int, default=1500)
    ap.add_argument("--no-laya", action="store_true")
    ap.add_argument("--n-heldout", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.splits, args.laya_max, args.n_heldout = 2, 3, 10
    out = args.out or ("results/tarski/explore_lit_cicc_smoke.json" if args.smoke else "results/tarski/explore_lit_cicc.json")
    log = Logger(out)
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = pick_device(args.smoke)
    trunk = get_trunk(args.smoke)
    t0 = time.time()
    agent, laya_used = None, None
    res = {"args": vars(args), "datasets": {}}
    if not args.no_laya:
        import laya
        for sub in (None, "typed-decisions"):             # the English checkpoint, else the typed-decisions one
            try:
                agent = laya.load("convaiinnovations/laya", subfolder=sub,
                                  device="cpu" if args.smoke else str(trunk.device))
                laya_used = sub or "english"
                break
            except Exception as ex:                        # not cached on this machine and the Hub is offline
                log(f"   laya ({sub or 'english'}) unavailable: {type(ex).__name__}: {str(ex)[:120]}")
        if agent is None:
            log("   continuing without laya: the clarify band reports top-1 and oracle-user only")
    res["laya_checkpoint"] = laya_used
    for dname in ("clinc150", "banking77"):
        ds = load_ds(dname, args.smoke, args.seed)
        allx, idx = split_index(ds)
        labels_all = ds.tasks["intent"].labels
        depths = sorted({args.depth, args.gate_depth})
        feats = pooled(trunk, [e.text for e in allx], depths, ds.max_len, log=log)
        if dname == "clinc150":
            oos_i = labels_all.index("oos")
            ids = [i for i in range(len(labels_all)) if i != oos_i]
        else:
            oos_i, ids = None, list(range(len(labels_all)))
        imap = {i: j for j, i in enumerate(ids)}
        labels = [labels_all[i] for i in ids]
        C = len(ids)
        y = np.array([imap.get(e.y["intent"], -1) for e in allx])
        tr, va, te = idx["train"], idx["val"], idx["test"]
        tr_in, va_in, te_in = tr[y[tr] >= 0], va[y[va] >= 0], te[y[te] >= 0]
        perm = rng.permutation(tr_in)
        fit_hold, tr_fit = perm[: len(perm) // 10], perm[len(perm) // 10:]
        X = zscore(feats[args.depth], tr_fit)
        pr = Probe(X, y, tr_fit, fit_hold, C, dev, steps=args.steps, seed=args.seed)
        P_cal, P_te = pr.probs(X[va_in]), pr.probs(X[te_in])
        y_cal, y_te = y[va_in], y[te_in]
        log(f"== {dname}: {C} routes, calibration {len(va_in)} (~{len(va_in) / C:.0f}/route), test {len(te_in)}, "
            f"probe@{args.depth} test acc {float((P_te.argmax(1) == y_te).mean()):.4f}")
        dres = {"sets": {}}
        M = 10 if C >= 100 else 8
        if args.smoke:
            M = 4
        for score in ("lac", "aps"):
            for alpha in args.alphas:
                srng = np.random.default_rng(args.seed + 1)
                Sc = lac_scores(P_cal) if score == "lac" else aps_scores(P_cal, srng)
                St = lac_scores(P_te) if score == "lac" else aps_scores(P_te, srng)
                s_true = Sc[np.arange(len(y_cal)), y_cal]
                r = {}
                q = qhat(s_true, alpha)
                r["marginal"] = evaluate_sets(St <= q, y_te, C, alpha)
                thc = np.array([qhat(s_true[y_cal == c], alpha) for c in range(C)])
                r["classwise"] = evaluate_sets(St <= thc[None], y_te, C, alpha)
                runs = []
                for sp in range(args.splits):
                    th = clustered_thresholds(s_true, y_cal, C, alpha, M, np.random.default_rng(args.seed + 100 + sp))
                    runs.append(evaluate_sets(St <= th[None], y_te, C, alpha))
                r["clustered"] = {k: float(np.mean([x[k] for x in runs])) for k in runs[0]}
                dres["sets"][f"{score}@{alpha}"] = r
                log(f"-- {dname} {score} alpha {alpha}: " + " | ".join(
                    f"{m}: cov {v['coverage']:.3f} gap {v['covgap']:.1f} worst10 {v['worst_decile_cov']:.3f} "
                    f"size {v['mean_size']:.2f} bands {v['band_single']:.2f}/{v['band_clarify']:.2f}/{v['band_escalate']:.2f}"
                    for m, v in r.items()))
        # route-or-clarify with laya on the middle band
        s_true = lac_scores(P_cal)[np.arange(len(y_cal)), y_cal]
        St = lac_scores(P_te)
        texts_te = [allx[i].text for i in te_in]
        rc = {}
        th_cl = clustered_thresholds(s_true, y_cal, C, args.route_alpha, M, np.random.default_rng(args.seed + 100))
        for name, sets in (("marginal", St <= qhat(s_true, args.route_alpha)), ("clustered", St <= th_cl[None])):
            picks = run_laya(agent, texts_te, sets, labels, y_te, args.laya_max, log) if agent is not None else {}
            rc[name] = route_or_clarify(sets, P_te, y_te, picks)
            log(f"   route-or-clarify ({name}, alpha {args.route_alpha}): " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float)
                                                                                    else f"{k} {v}" for k, v in rc[name].items()))
        dres["route_or_clarify"] = rc
        # open-set gate: hold out routes, gate on a conformal p-value of Mahalanobis++
        if dname == "clinc150":
            held = rng.choice(C, min(args.n_heldout, C // 3), replace=False)
            known = np.setdiff1d(np.arange(C), held)
            kmap = {int(c): j for j, c in enumerate(known)}
            yk = np.array([kmap.get(int(v), -1) if v >= 0 else -1 for v in y])
            trk = tr_fit[yk[tr_fit] >= 0]
            hold_k = fit_hold[yk[fit_hold] >= 0]
            Xk = zscore(feats[args.depth], trk)
            prk = Probe(Xk, yk, trk, hold_k, len(known), dev, steps=args.steps, seed=args.seed)
            cen = feats[args.gate_depth][trk].mean(0, keepdim=True)
            gk = Gauss(l2n(feats[args.gate_depth][trk] - cen), yk[trk], len(known), 0.1)
            gate = lambda rows: gk.min_dist(l2n(feats[args.gate_depth][rows] - cen))
            cal_k = va[yk[va] >= 0]
            g_cal = np.sort(gate(cal_k))
            pval = lambda rows: (1 + len(g_cal) - np.searchsorted(g_cal, gate(rows), side="left")) / (len(g_cal) + 1)
            Pk_cal = prk.probs(Xk[cal_k])
            qk = qhat(1 - Pk_cal[np.arange(len(cal_k)), yk[cal_k]], args.route_alpha)
            te_known = te[yk[te] >= 0]
            te_held = te[(y[te] >= 0) & (yk[te] < 0)]
            te_oos = te[y[te] < 0]
            og = {}
            for nm, rows in (("known", te_known), ("held_out_routes", te_held), ("out_of_scope", te_oos)):
                gated = pval(rows) <= args.gate_alpha
                sets = (1 - prk.probs(Xk[rows])) <= qk
                silent = (~gated) & (sets.sum(1) == 1)            # a confident single (necessarily wrong for new routes)
                og[nm] = {"gated_to_new_route": float(gated.mean()), "silent_single_route": float(silent.mean()),
                          "n": int(len(rows))}
                if nm == "known":
                    og[nm]["coverage_when_not_gated"] = float(sets[np.arange(len(rows)), yk[rows]][~gated].mean())
            dres["open_set_gate"] = og
            log("   open-set gate (alpha_o " + str(args.gate_alpha) + "): " + " | ".join(
                f"{k}: gated {v['gated_to_new_route']:.3f}, silent single {v['silent_single_route']:.3f}" for k, v in og.items()))
        res["datasets"][dname] = dres
        log.dump(res)
    res["wall_s"] = round(time.time() - t0, 1)
    log.dump(res)
    log(f"done in {res['wall_s']}s -> {out}")


if __name__ == "__main__":
    main()
