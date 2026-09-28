"""Entry 2 (lit_incremental_oos.md): auto-route only when the misroute rate is provably <= q; escalate the rest.

Conformal selection turns any confidence score into a set of messages to act on with false-discovery-rate
control: among the auto-routed messages, the expected share that is misrouted is at most q.
  - Huang et al., "Selective Labeling with False Discovery Rate Control" (Conformal Labeling, arXiv
    2510.14581, 2025): a test item's p-value compares its uncertainty with the calibration items the model
    got WRONG; Benjamini-Hochberg at an adjusted level selects the items to trust.
  - Jin & Candes, "Selection by Prediction with Conformal p-values", JMLR 2023 (arXiv 2210.01408).
  - Gui, Jin & Ren, "Conformal Alignment", NeurIPS 2024 (arXiv 2405.10301).
The guarantee needs calibration and deployment to be exchangeable. CLINC150 breaks that on purpose (3.2%
out-of-scope in validation, 18.2% in test), so label-shift weighting (Podkopaev & Ramdas, UAI 2021, arXiv
2103.03323) is added: calibration items are weighted by pi_test(y) / pi_cal(y), with pi_test estimated by
Saerens EM on the unlabelled test pool; the test point gets the largest weight (conservative).

Arms, for q in {1, 2, 5, 10}%:
  fdr_search     the naive rule: the lowest confidence threshold whose empirical misroute rate on the
                 calibration set is <= q
  conformal      Conformal Labeling (unweighted)
  weighted_em    label-shift-weighted conformal selection, weights from the EM prior estimate
  weighted_true  the same with the true test prior (reference)
  weighted_em_predw  heuristic (no guarantee): the test point weighted by its predicted class, not the max
  batch_online   weighted_em applied to consecutive batches of 200 messages in arrival order (the prior
                 re-estimated from everything seen so far): what a deployment that routes hourly can do
Caveat built into the design: EM cannot estimate the prior of a label the model never predicts, so for the
150-way decision without an out-of-scope label weighted_em cannot up-weight out-of-scope errors (only
weighted_true can); entry 4 is the fix for that case.
Decisions: CLINC150 151-way (out-of-scope is a label; routing an out-of-scope message to an intent is an
error), CLINC150 150-way with no out-of-scope label (every out-of-scope message is an error), Banking77.
The probe is a calibrated linear probe at each depth. Validation is split in half: one half picks the
probe's L2 and temperature, the other is the conformal calibration set (so calibration stays exchangeable
with fresh data); deployment = test.
Metrics over bootstrap resamples of (calibration, test): mean realised misroute rate among auto-routed
messages (the FDR), the share auto-routed (power), and the rate of runs whose FDR exceeds q.
Expected: Banking77 all arms at or under q; CLINC unweighted arms over q, weighted arms at or under q.

Usage:
  .venv/bin/python explore/lit_oos_conformal_select.py --smoke
  .venv/bin/python explore/lit_oos_conformal_select.py --out results/tarski/explore_lit_conformal_select.json   # ~4 min A10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_oos_common import (Logger, Probe, class_prior, get_trunk, load_ds, pick_device, pooled, saerens_em,
                            seed_all, split_index, zscore)

import numpy as np

QS = (0.01, 0.02, 0.05, 0.10)


def select_weighted(s_cal: np.ndarray, wrong_cal: np.ndarray, w_cal: np.ndarray, s_test: np.ndarray,
                    w_test_max, q: float, rng) -> np.ndarray:
    """Weighted Conformal Labeling / conformal selection. Lower score = more confident. Returns a boolean
    mask of selected test items. With unit weights this is Huang et al.'s rule: BH on
    p_j = (#{wrong cal: s_i < s_j} + U (1 + #{wrong: s_i = s_j})) / (n0 + 1) at level q (n+1) / (n0+1)."""
    m = len(s_test)
    ws = w_cal[wrong_cal]
    sw = s_cal[wrong_cal]
    order = np.argsort(sw)
    sw, ws = sw[order], ws[order]
    cum = np.concatenate([[0.0], np.cumsum(ws)])
    # weighted mass of wrong calibration items with score strictly below each test score
    below = cum[np.searchsorted(sw, s_test, side="left")]
    ties = cum[np.searchsorted(sw, s_test, side="right")] - below
    num = below + rng.random(m) * (w_test_max + ties)
    W_all = float(w_cal.sum()) + float(np.mean(w_test_max))      # w_test_max may be per test point (heuristic arm)
    # estimated FDP when selecting the k most confident: m * num_(k) / (W_all * k)
    o = np.argsort(num)
    k = np.arange(1, m + 1)
    ok = m * num[o] / (W_all * k) <= q
    sel = np.zeros(m, dtype=bool)
    if ok.any():
        kmax = int(np.max(np.where(ok)[0])) + 1
        sel[o[:kmax]] = True
    return sel


def fdr_search(s_cal, wrong_cal, s_test, q) -> np.ndarray:
    order = np.argsort(s_cal)
    fdp = np.cumsum(wrong_cal[order]) / np.arange(1, len(order) + 1)
    ok = np.where(fdp <= q)[0]
    if len(ok) == 0:
        return np.zeros(len(s_test), dtype=bool)
    thr = s_cal[order][ok.max()]
    return s_test <= thr


def run_decision(P_cal, y_cal, P_te, y_te, prior_src, C, q_list, B, rng, batch, B_online):
    """prior_src: the class prior the probe's posteriors imply (its training prior), for EM."""
    s_cal, s_te = 1 - P_cal.max(-1), 1 - P_te.max(-1)
    wrong_cal = P_cal.argmax(-1) != y_cal
    wrong_te = P_te.argmax(-1) != y_te
    arms = ("fdr_search", "conformal", "weighted_em", "weighted_true", "batch_online", "weighted_em_predw")
    res = {f"{a}@{q}": {"fdr": [], "power": []} for a in arms for q in q_list}
    n_c, n_t = len(y_cal), len(y_te)
    for b in range(B):
        ic = rng.integers(0, n_c, n_c) if b > 0 else np.arange(n_c)
        it = rng.integers(0, n_t, n_t) if b > 0 else np.arange(n_t)
        sc, wc, yc = s_cal[ic], wrong_cal[ic], y_cal[ic]
        st, wt = s_te[it], wrong_te[it]
        pi_cal = class_prior(yc, C)
        sup = pi_cal > 0                                   # classes with calibration support
        pi_cal = np.where(sup, pi_cal, 1.0)
        _, pi_em = saerens_em(P_te[it], prior_src)
        pi_true = class_prior(y_te[it], C)
        wts = {"conformal": np.ones(C), "weighted_em": pi_em / pi_cal, "weighted_true": pi_true / pi_cal}
        online = b < B_online
        if online:
            # batch-online: arrival order = the resampled order; prior from all messages seen so far
            batches = [np.arange(s0, min(n_t, s0 + batch)) for s0 in range(0, n_t, batch)]
            w_batches = [saerens_em(P_te[it[: rows[-1] + 1]], prior_src, iters=50)[1] / pi_cal for rows in batches]
        for q in q_list:
            sel = fdr_search(sc, wc, st, q)
            res[f"fdr_search@{q}"]["fdr"].append(wt[sel].mean() if sel.any() else 0.0)
            res[f"fdr_search@{q}"]["power"].append(sel.mean())
            for name, w in wts.items():
                sel = select_weighted(sc, wc, w[yc], st, float(w[sup].max()), q, rng)
                res[f"{name}@{q}"]["fdr"].append(wt[sel].mean() if sel.any() else 0.0)
                res[f"{name}@{q}"]["power"].append(sel.mean())
            # heuristic, no guarantee: the test point weighted by its predicted class instead of the max
            w = wts["weighted_em"]
            sel = select_weighted(sc, wc, w[yc], st, w[P_te[it].argmax(-1)], q, rng)
            res[f"weighted_em_predw@{q}"]["fdr"].append(wt[sel].mean() if sel.any() else 0.0)
            res[f"weighted_em_predw@{q}"]["power"].append(sel.mean())
            if online:
                sel_all = np.zeros(n_t, dtype=bool)
                for rows, w in zip(batches, w_batches):
                    sel_all[rows] = select_weighted(sc, wc, w[yc], st[rows], float(w[sup].max()), q, rng)
                res[f"batch_online@{q}"]["fdr"].append(wt[sel_all].mean() if sel_all.any() else 0.0)
                res[f"batch_online@{q}"]["power"].append(sel_all.mean())
    out = {}
    for k, v in res.items():
        q = float(k.split("@")[1])
        f, p = np.array(v["fdr"]), np.array(v["power"])
        out[k] = {"fdr_mean": float(f.mean()), "fdr_se": float(f.std() / np.sqrt(len(f))),
                  "fdr_first_draw": float(f[0]), "power_mean": float(p.mean()),
                  "frac_runs_fdr_over_q": float((f > q).mean()), "runs": int(len(f))}
    out["_info"] = {"cal_error_rate": float(wrong_cal.mean()), "test_error_rate": float(wrong_te.mean()),
                    "n_cal": int(n_c), "n_test": int(n_t)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--depths", type=int, nargs="*", default=[4, 22])
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--boot-online", type=int, default=20, help="bootstrap draws that also run batch_online")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.depths, args.boot, args.batch, args.boot_online = [22], 5, 100, 2
    out = args.out or ("results/tarski/explore_lit_conformal_select_smoke.json" if args.smoke
                       else "results/tarski/explore_lit_conformal_select.json")
    log = Logger(out)
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = pick_device(args.smoke)
    trunk = get_trunk(args.smoke)
    t0 = time.time()
    res = {"args": vars(args), "decisions": {}}
    for dname in ("clinc150", "banking77"):
        ds = load_ds(dname, args.smoke, args.seed)
        allx, idx = split_index(ds)
        feats = pooled(trunk, [e.text for e in allx], args.depths, ds.max_len, log=log)
        tr, va_all, te = idx["train"], idx["val"], idx["test"]
        # validation halves: one to pick the probe's L2 and temperature, one for conformal calibration only
        perm = np.random.default_rng(args.seed + 1).permutation(va_all)
        va, cal = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
        decisions = {}
        if dname == "clinc150":
            intents = ds.tasks["intent"].labels
            oos_i = intents.index("oos")
            in_ids = [i for i in range(len(intents)) if i != oos_i]
            imap = {i: j for j, i in enumerate(in_ids)}
            y_oos = np.array([e.y["oos"] for e in allx])
            y150 = np.array([imap.get(e.y["intent"], -1) for e in allx])
            y151 = np.where(y_oos == 1, 150, y150)
            decisions["clinc151"] = (y151, 151, tr, None)
            decisions["clinc150_no_oos_label"] = (y150, 150, tr[y_oos[tr] == 0], va[y_oos[va] == 0])
        else:
            y = np.array([e.y["intent"] for e in allx])
            decisions["banking77"] = (y, len(ds.tasks["intent"].labels), tr, None)
        for d in args.depths:
            for name, (y, C, trr, vaT) in decisions.items():
                t1 = time.time()
                X = zscore(feats[d], trr)
                pr = Probe(X, y, trr, va if vaT is None else vaT, C, dev, steps=args.steps, seed=args.seed, va_T=vaT)
                P_cal, P_te = pr.probs(X[cal]), pr.probs(X[te])
                y_cal, y_te = y[cal].copy(), y[te].copy()
                if name == "clinc150_no_oos_label":
                    # an out-of-scope message can never be routed correctly: give it an impossible label
                    y_cal[y_cal < 0], y_te[y_te < 0] = C, C
                    P_cal_ext = np.concatenate([P_cal, np.zeros((len(P_cal), 1))], 1)
                    P_te_ext = np.concatenate([P_te, np.zeros((len(P_te), 1))], 1)
                    prior_src = np.r_[class_prior(y[trr], C), 0.0]
                    r = run_decision(P_cal_ext, y_cal, P_te_ext, y_te, np.clip(prior_src, 1e-9, None), C + 1,
                                     QS, args.boot, rng, args.batch, args.boot_online)
                else:
                    # the probe's posteriors imply its training prior (a temperature does not move it)
                    r = run_decision(P_cal, y_cal, P_te, y_te, np.clip(class_prior(y[trr], C), 1e-6, None), C,
                                     QS, args.boot, rng, args.batch, args.boot_online)
                res["decisions"].setdefault(name, {})[d] = r
                info = r["_info"]
                log(f"-- {name} @ depth {d} ({time.time() - t1:.0f}s): error rate cal {info['cal_error_rate']:.3f} "
                    f"test {info['test_error_rate']:.3f}")
                for q in QS:
                    log(f"   q={q:.2f}: " + " | ".join(
                        f"{a} FDR {r[f'{a}@{q}']['fdr_mean']:.3f} (over-q {r[f'{a}@{q}']['frac_runs_fdr_over_q']:.2f}) "
                        f"auto {r[f'{a}@{q}']['power_mean']:.2f}"
                        for a in ("fdr_search", "conformal", "weighted_em", "weighted_true", "batch_online",
                                  "weighted_em_predw")))
                log.dump(res)
    res["wall_s"] = round(time.time() - t0, 1)
    log.dump(res)
    log(f"done in {res['wall_s']}s -> {out}")


if __name__ == "__main__":
    main()
