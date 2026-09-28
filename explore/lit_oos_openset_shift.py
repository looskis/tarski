"""Entry 4 (lit_incremental_oos.md): open-set label shift. Estimate the share of traffic that fits no route,
with zero out-of-scope labels, then use it to set the out-of-scope threshold.

The skeptic's Saerens EM result (CLINC oos-F1 0.29 -> 0.57) needed a supervised oos branch. User-defined
routes rarely have "none of the above" labels. Open-set label shift estimation reads unlabelled traffic
and needs only the in-scope classifier plus an out-of-scope score:
  - Ye, Tsuchida, Petersson & Barnes, "Open Set Label Shift with Test Time Out-of-Distribution Reference",
    CVPR 2025 (arXiv 2505.05868): an ID/OOD classifier h trained against a reference OOD set, EM over K+1
    classes on the unlabelled target, then a correction for the ID/OOD classifier's bias;
  - Garg et al., "Domain Adaptation under Open Set Label Shift" (PULSE, NeurIPS 2022, arXiv 2207.13048),
    which reduces the novel-class fraction to positive-unlabelled mixture-proportion estimation with Best
    Bin Estimation (BBE; Garg et al., NeurIPS 2021, arXiv 2111.00980).

Setup (CLINC150, no out-of-scope training labels anywhere except the marked references):
  classifier   150-way in-scope probe at --cls-depth, temperature on in-scope validation
  OOD scores   Mahalanobis++ (centred + L2, entry 3) and kNN10 at --ood-depth, and 1 - MSP
  reference    Banking77 messages ("free outliers", as in learning idea J) or noise-mixed in-scope states
               (Ye et al.'s pseudo-OOD), used only to train h and to measure its bias
Estimators of the out-of-scope fraction pi in a pool:
  cc            classify-and-count at the threshold keeping 95% of in-scope validation messages
  em_k1         Ye stage 2: Saerens EM over the K+1 posterior [(1-h) f, h]
  acc_ref       Ye stage 3 in spirit: adjusted classify-and-count, (mean h - mu0) / (mu1 - mu0) with mu0 from
                held-out in-scope and mu1 from held-out reference
  bbe           PULSE-lite: in-scope train (P) vs pool (U) discriminator on trunk states, 2-fold cross-fitted,
                Best Bin Estimation of the in-scope share
  acc_realoos   reference that uses the 100 validation OOS labels for mu1
  sup_em        reference: the 151-way probe trained WITH the 250 OOS training messages + Saerens EM
Pools: the real test set (18.2% OOS) and resampled pools at 5, 10, 30 and 50% OOS (5 draws each).
Metrics: |pi_hat - pi|, and OOS F1 / 151-way macro-F1 when the Mahalanobis++ score is thresholded at the
(1 - pi_hat) quantile of the pool (and at the true pi, the oracle).
Expected: fraction error under ~4 points on the real split for the best estimator; oos-F1 0.5-0.65 with no
OOS labels (supervised Saerens reached 0.57).

Usage:
  .venv/bin/python explore/lit_oos_openset_shift.py --smoke
  .venv/bin/python explore/lit_oos_openset_shift.py --out results/tarski/explore_lit_openset_shift.json   # ~5 min A10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_oos_common import (Gauss, Logger, Probe, class_prior, f1_at, get_trunk, knn_dist, l2n, load_ds,
                            ood_metrics, pick_device, pooled, saerens_em, seed_all, split_index, zscore)

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def bbe(scores_p: np.ndarray, scores_u: np.ndarray, delta: float = 0.1, gamma: float = 0.01) -> float:
    """Best Bin Estimation (Garg et al. 2021): share of positives in U from top bins of a P-vs-U score."""
    cands = np.unique(np.quantile(scores_p, np.linspace(0, 0.99, 200)))
    n_p, n_u = len(scores_p), len(scores_u)
    pen = np.sqrt(np.log(4 / delta) / (2 * n_p)) + np.sqrt(np.log(4 / delta) / (2 * n_u))
    best, best_obj = 1.0, np.inf
    for c in cands:
        qp, qu = (scores_p >= c).mean(), (scores_u >= c).mean()
        if qp <= 0:
            continue
        obj = qu / qp + (1 + gamma) / qp * pen
        if obj < best_obj:
            best_obj, best = obj, qu / qp
    return float(min(1.0, best))


def pulse_lite(Zp: np.ndarray, Zu: np.ndarray, seed: int) -> float:
    """OOS share of U = 1 - (in-scope share), with a cross-fitted logistic P-vs-U discriminator."""
    rng = np.random.default_rng(seed)
    pp, pu = rng.permutation(len(Zp)), rng.permutation(len(Zu))
    halves_p, halves_u = np.array_split(pp, 2), np.array_split(pu, 2)
    est = []
    for a in (0, 1):
        b = 1 - a
        X = np.concatenate([Zp[halves_p[a]], Zu[halves_u[a]]])
        y = np.r_[np.ones(len(halves_p[a])), np.zeros(len(halves_u[a]))]
        clf = LogisticRegression(C=0.1, max_iter=300, class_weight="balanced").fit(X, y)
        est.append(bbe(clf.predict_proba(Zp[halves_p[b]])[:, 1], clf.predict_proba(Zu[halves_u[b]])[:, 1]))
    return float(1 - np.mean(est))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cls-depth", type=int, default=22)
    ap.add_argument("--ood-depth", type=int, default=4)
    ap.add_argument("--n-ref", type=int, default=3000, help="Banking77 reference messages")
    ap.add_argument("--rates", type=float, nargs="*", default=[0.05, 0.10, 0.30, 0.50])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--noise-lambda", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.n_ref, args.draws, args.rates = 200, 2, [0.1, 0.3]
    out = args.out or ("results/tarski/explore_lit_openset_shift_smoke.json" if args.smoke
                       else "results/tarski/explore_lit_openset_shift.json")
    log = Logger(out)
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = pick_device(args.smoke)
    trunk = get_trunk(args.smoke)
    t0 = time.time()
    ds = load_ds("clinc150", args.smoke, args.seed)
    allx, idx = split_index(ds)
    intents = ds.tasks["intent"].labels
    oos_i = intents.index("oos")
    in_ids = [i for i in range(len(intents)) if i != oos_i]
    imap = {i: j for j, i in enumerate(in_ids)}
    y_int = np.array([e.y["intent"] for e in allx])
    y_oos = np.array([e.y["oos"] for e in allx])
    y150 = np.array([imap.get(i, -1) for i in y_int])
    y151 = np.where(y_oos == 1, 150, y150)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    tr_in, va_in = tr[y_oos[tr] == 0], va[y_oos[va] == 0]
    va_oos = va[y_oos[va] == 1]
    depths = sorted({args.cls_depth, args.ood_depth})
    F_ = pooled(trunk, [e.text for e in allx], depths, ds.max_len, log=log)
    bank = load_ds("banking77", args.smoke, args.seed)
    ref_texts = [e.text for e in bank.train][: args.n_ref]
    R_ = pooled(trunk, ref_texts, depths, ds.max_len, log=log)
    log(f"== open-set label shift on CLINC150 | cls depth {args.cls_depth}, ood depth {args.ood_depth} | "
        f"{len(ref_texts)} Banking77 reference messages | fits {dev}")

    # z-scored states (one scaler from in-scope train), classifier, scores
    def zs(d):
        mu, sd = F_[d][tr_in].mean(0, keepdim=True), F_[d][tr_in].std(0, keepdim=True).clamp_min(1e-4)
        return (F_[d] - mu) / sd, (R_[d] - mu) / sd, mu, sd
    Xc, Rc, mu_c, sd_c = zs(args.cls_depth)
    Xo, Ro, mu_o, sd_o = zs(args.ood_depth)
    probe = Probe(Xc, y150, tr_in, va_in, 150, dev, steps=args.steps, seed=args.seed)
    cen = F_[args.ood_depth][tr_in].mean(0, keepdim=True)
    gm = Gauss(l2n(F_[args.ood_depth][tr_in] - cen), y150[tr_in], 150, 0.1)

    # held-out halves of in-scope validation (h training / bias estimation)
    perm = rng.permutation(va_in)
    va_a, va_b = perm[: len(perm) // 2], perm[len(perm) // 2:]

    def scores(raw_o: torch.Tensor, zc: torch.Tensor, knn_excl=None) -> dict:
        s = {"maha_cl2": gm.min_dist(l2n(raw_o - cen)),
             "knn10": knn_dist(F_[args.ood_depth][tr_in], raw_o, (10,), dev)[10],
             "msp": 1 - probe.probs(zc).max(-1)}
        return s

    S_all = scores(F_[args.ood_depth], Xc)                         # every CLINC message
    S_ref = scores(R_[args.ood_depth], Rc)
    # noise-mixed pseudo-OOD from the in-scope validation half A (Ye et al.'s reference-free variant)
    g = torch.Generator().manual_seed(args.seed)
    lam = args.noise_lambda

    def noisy(d, rows):
        X = F_[d][rows]
        return (1 - lam) * X + lam * X.std(0, keepdim=True) * torch.randn(X.shape, generator=g) + lam * X.mean(0, keepdim=True)
    Nraw_o, Nraw_c = noisy(args.ood_depth, va), noisy(args.cls_depth, va)
    S_noise = scores(Nraw_o, (Nraw_c - mu_c) / sd_c)

    P_all = probe.probs(Xc)                                         # 150-way in-scope posteriors
    res = {"args": vars(args), "ood_auroc_test": {k: ood_metrics(v[te], y_oos[te])["auroc"] for k, v in S_all.items()},
           "pools": {}}
    log("   OOD AUROC on the full test set: " + " | ".join(f"{k} {v:.3f}" for k, v in res["ood_auroc_test"].items()))

    # h: ID-vs-reference logistic regression on standardised [score, msp]
    feats_of = lambda S, rows: np.stack([S["maha_cl2"][rows] if rows is not None else S["maha_cl2"],
                                         S["msp"][rows] if rows is not None else S["msp"]], 1)
    H = {}
    ref_perm = rng.permutation(len(ref_texts))
    ra, rb = ref_perm[: len(ref_perm) // 2], ref_perm[len(ref_perm) // 2:]
    noise_perm = rng.permutation(len(va))
    na, nb = noise_perm[: len(noise_perm) // 2], noise_perm[len(noise_perm) // 2:]
    for ref_name, (Sr, a_rows, b_rows) in {"banking77": (S_ref, ra, rb), "noise": (S_noise, na, nb)}.items():
        Xin = feats_of(S_all, va_a)
        Xrf = feats_of(Sr, a_rows)
        m, s = Xin.mean(0), Xin.std(0) + 1e-9
        clf = LogisticRegression(C=1.0, max_iter=500).fit((np.r_[Xin, Xrf] - m) / s, np.r_[np.zeros(len(Xin)), np.ones(len(Xrf))])
        h = lambda X, clf=clf, m=m, s=s: clf.predict_proba((X - m) / s)[:, 1]
        rho = len(Xrf) / (len(Xin) + len(Xrf))
        mu0 = float(h(feats_of(S_all, va_b)).mean())
        mu1 = float(h(feats_of(Sr, b_rows)).mean())
        mu1_real = float(h(feats_of(S_all, va_oos)).mean())         # reference only: uses val OOS labels
        H[ref_name] = {"h": h, "rho": rho, "mu0": mu0, "mu1": mu1, "mu1_realoos": mu1_real}
        log(f"   h[{ref_name}]: rho {rho:.3f}, mean h on held-out in-scope {mu0:.3f}, on held-out reference {mu1:.3f}, "
            f"on real val OOS {mu1_real:.3f}")

    # supervised reference: 151-way probe with the OOS training class
    probe151 = Probe(Xc, y151, tr, va, 151, dev, steps=args.steps, seed=args.seed)
    P151 = probe151.probs(Xc)
    prior151 = class_prior(y151[tr], 151)
    thr95 = np.quantile(S_all["maha_cl2"][va_in], 0.95)

    def evaluate_pool(rows: np.ndarray) -> dict:
        pi_true = float(y_oos[rows].mean())
        est = {}
        est["cc"] = float((S_all["maha_cl2"][rows] > thr95).mean())
        for rn, hh in H.items():
            hv = hh["h"](feats_of(S_all, rows))
            Pk1 = np.concatenate([(1 - hv)[:, None] * P_all[rows], hv[:, None]], 1)
            prior_src = np.r_[(1 - hh["rho"]) * np.full(150, 1 / 150), hh["rho"]]
            _, pt = saerens_em(Pk1, prior_src)
            est[f"em_k1_{rn}"] = float(pt[-1])
            est[f"acc_ref_{rn}"] = float(np.clip((hv.mean() - hh["mu0"]) / max(1e-6, hh["mu1"] - hh["mu0"]), 0, 1))
            est[f"acc_realoos_{rn}"] = float(np.clip((hv.mean() - hh["mu0"]) / max(1e-6, hh["mu1_realoos"] - hh["mu0"]), 0, 1))
        Zp = Xc[tr_in].numpy()
        sub = rng.choice(len(Zp), min(len(Zp), 4 * len(rows)), replace=False)
        est["bbe"] = pulse_lite(Zp[sub], Xc[rows].numpy(), int(rng.integers(1 << 30)))
        adj, pt = saerens_em(P151[rows], prior151)
        est["sup_em"] = float(pt[150])
        out = {"pi_true": pi_true, "n": int(len(rows)), "estimates": est,
               "abs_err": {k: abs(v - pi_true) for k, v in est.items()}}
        # thresholding the Mahalanobis++ score at each estimated rate
        s = S_all["maha_cl2"][rows]
        yo = y_oos[rows]
        in_pred = P_all[rows].argmax(-1)
        f1 = {}
        for k, pi in list(est.items()) + [("oracle", pi_true)]:
            kk = int(round(np.clip(pi, 0, 1) * len(rows)))
            pred_oos = np.zeros(len(rows), dtype=int)
            if kk > 0:
                pred_oos[np.argsort(-s)[:kk]] = 1
            y_pred = np.where(pred_oos == 1, 150, in_pred)
            f1[k] = {"oos_f1": f1_at(pred_oos, yo)["f1"],
                     "macro_f1_151": float(f1_score(y151[rows], y_pred, average="macro", labels=np.arange(151), zero_division=0))}
        f1["sup_em_argmax"] = {"oos_f1": f1_at((adj.argmax(-1) == 150).astype(int), yo)["f1"],
                               "macro_f1_151": float(f1_score(y151[rows], adj.argmax(-1), average="macro",
                                                              labels=np.arange(151), zero_division=0))}
        out["f1_by_estimate"] = f1
        return out

    te_in, te_oos = te[y_oos[te] == 0], te[y_oos[te] == 1]
    pools = {"real": [te]}
    for r in args.rates:
        draws = []
        for _ in range(args.draws):
            n_oos = min(len(te_oos), int(round(r * len(te_in) / (1 - r))))
            n_in = min(len(te_in), int(round(n_oos * (1 - r) / r)))
            draws.append(np.concatenate([rng.choice(te_in, n_in, replace=False), rng.choice(te_oos, n_oos, replace=False)]))
        pools[f"rate_{r}"] = draws
    for name, draws in pools.items():
        rs = [evaluate_pool(rows) for rows in draws]
        keys = rs[0]["estimates"].keys()
        agg = {"pi_true": float(np.mean([r["pi_true"] for r in rs])),
               "mean_estimate": {k: float(np.mean([r["estimates"][k] for r in rs])) for k in keys},
               "mean_abs_err": {k: float(np.mean([r["abs_err"][k] for r in rs])) for k in keys},
               "oos_f1": {k: float(np.mean([r["f1_by_estimate"][k]["oos_f1"] for r in rs])) for k in rs[0]["f1_by_estimate"]},
               "macro_f1_151": {k: float(np.mean([r["f1_by_estimate"][k]["macro_f1_151"] for r in rs]))
                                for k in rs[0]["f1_by_estimate"]},
               "draws": rs}
        res["pools"][name] = agg
        log(f"-- pool {name}: true OOS share {agg['pi_true']:.3f}")
        log("   estimate (abs err): " + " | ".join(f"{k} {agg['mean_estimate'][k]:.3f} ({agg['mean_abs_err'][k]:.3f})" for k in keys))
        log("   OOS F1 thresholding Mahalanobis++ at the estimate: " + " | ".join(f"{k} {v:.3f}" for k, v in agg["oos_f1"].items()))
        log.dump(res)
    res["wall_s"] = round(time.time() - t0, 1)
    log.dump(res)
    log(f"done in {res['wall_s']}s -> {out}")


if __name__ == "__main__":
    main()
