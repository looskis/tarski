"""Entry 3 (lit_incremental_oos.md): Mahalanobis++ and kernel-PCA out-of-scope scores on frozen trunk states.

Our label-free Mahalanobis OOS score is the weakest one we have (CLINC150 AUROC 0.80-0.86, falling with
depth), while kNN on L2-normalised copies of the same states reaches 0.906. The implementation z-scores each
dimension but never normalises each message. Mueller & Hein, "Mahalanobis++: Improving OOD Detection via
Feature Normalization" (ICML 2025, arXiv 2505.18032) show that per-sample feature norms vary and correlate
with the Mahalanobis score regardless of in/out status, and that L2-normalising before fitting the means
and covariance fixes it. Fang et al., "Kernel PCA for Out-of-Distribution Detection" (NeurIPS 2024,
arXiv 2402.02949) score by PCA reconstruction error after a cosine (L2) or cosine + random-Fourier map.

Scores (all fitted on in-scope training messages only, no out-of-scope data), at every requested depth:
  maha_z        current recipe (per-dimension z-score, shared covariance shrunk by 0.1)
  maha_raw      no standardisation
  maha_l2       Mahalanobis++: L2-normalise the raw pooled state
  maha_cl2      centre by the training mean, then L2-normalise (transformer mean pools share a large common
                direction; plain L2 may keep it)
  maha_zl2      z-score, then L2-normalise
  rmd_cl2       relative Mahalanobis on cl2 features (min class distance minus background distance)
  kpca_cos_r    PCA reconstruction error on cl2 features, top components explaining ratio r of variance
  kpca_rff_r    the same after random Fourier features (D=2048, gamma = c / median squared distance)
  knn1, knn10   k-th nearest-neighbour distance on L2-normalised features (baseline, as in the syndrome run)
  msp           1 - max softmax of a 150-way in-scope probe at that depth (baseline)
Diagnostic: Spearman correlation between the raw feature norm and each Mahalanobis score on in-scope test
messages (the paper's check: high |rho| = the norm is driving the score).

Metrics: AUROC / AUPR / FPR@95 on the 5,500-message CLINC150 test set (1,000 out-of-scope). For each score
at its validation-best depth, OOS F1 three ways: at the validation-F1 threshold; after 1-D logistic
calibration on validation plus Saerens EM on the test pool (both use the 100 validation OOS labels); and at
the oracle test rate (reference). Hyperparameters picked on validation AUROC also use those 100 labels; the
fixed defaults (kpca ratio 0.9, c = 1) are reported too, for a label-free choice.

Baselines to beat (explore_syndrome.log): kNN1 0.906 and MSP 0.929 at depth 4. Expected: maha_l2 or
maha_cl2 at 0.90-0.94 with a flatter depth trend and a smaller norm-score correlation.

Usage:
  .venv/bin/python explore/lit_oos_mahapp.py --smoke
  .venv/bin/python explore/lit_oos_mahapp.py --out results/tarski/explore_lit_mahapp.json   # ~3 min on an A10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_oos_common import (RFF, Gauss, Logger, Probe, binary_em_f1, f1_at, get_trunk, knn_dist, l2n, load_ds,
                            maha_background, median_sqdist, ood_metrics, pca_recon_error, pick_device, pooled,
                            score_to_prob, seed_all, split_index, zscore)

import numpy as np
import torch
from scipy.stats import spearmanr

RATIOS = (0.5, 0.7, 0.9, 0.95)
CS = (0.5, 1.0, 2.0)


def scores_at_depth(Xraw, tr_in, rows_eval, y150, dev, args) -> dict:
    """All OOS scores for rows_eval (higher = more out-of-scope)."""
    S = {}
    Xz = zscore(Xraw, tr_in)
    Xcl2 = l2n(Xraw - Xraw[tr_in].mean(0, keepdim=True))
    feats = {"maha_z": Xz, "maha_raw": Xraw, "maha_l2": l2n(Xraw), "maha_cl2": Xcl2, "maha_zl2": l2n(Xz)}
    y_tr = y150[tr_in]
    for name, X in feats.items():
        g = Gauss(X[tr_in], y_tr, 150, shrink=args.shrink)
        S[name] = g.min_dist(X[rows_eval])
        if name == "maha_cl2":
            bg = maha_background(X[tr_in], shrink=args.shrink)
            S["rmd_cl2"] = (g.d2(X[rows_eval]) - bg.d2(X[rows_eval])).min(-1).values.numpy()
    rec = pca_recon_error(Xcl2[tr_in], Xcl2[rows_eval], RATIOS)
    for r, v in rec.items():
        S[f"kpca_cos_{r}"] = v
    med = median_sqdist(Xcl2[tr_in])
    for c in CS:
        rff = RFF(Xcl2.shape[1], args.rff_dim, gamma=c / max(med, 1e-9), seed=0)
        Ztr, Zev = rff(Xcl2[tr_in], dev), rff(Xcl2[rows_eval], dev)
        rec = pca_recon_error(Ztr, Zev, RATIOS)
        for r, v in rec.items():
            S[f"kpca_rff_c{c}_{r}"] = v
    kn = knn_dist(Xraw[tr_in], Xraw[rows_eval], (1, 10), dev)
    S["knn1"], S["knn10"] = kn[1], kn[10]
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2, 4, 6, 8, 11, 14, 16, 18, 20, 22])
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--rff-dim", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.depths, args.rff_dim = [4, 22], 256
    out = args.out or ("results/tarski/explore_lit_mahapp_smoke.json" if args.smoke
                       else "results/tarski/explore_lit_mahapp.json")
    log = Logger(out)
    seed_all(args.seed)
    dev = pick_device(args.smoke)
    trunk = get_trunk(args.smoke)
    ds = load_ds("clinc150", args.smoke, args.seed)
    allx, idx = split_index(ds)
    intents = ds.tasks["intent"].labels
    oos_i = intents.index("oos")
    in_ids = [i for i in range(len(intents)) if i != oos_i]
    imap = {i: j for j, i in enumerate(in_ids)}
    y_int = np.array([e.y["intent"] for e in allx])
    y_oos = np.array([e.y["oos"] for e in allx])
    y150 = np.array([imap.get(i, -1) for i in y_int])
    tr_in = idx["train"][y_oos[idx["train"]] == 0]
    va, te = idx["val"], idx["test"]
    va_in = va[y_oos[va] == 0]
    rows_eval = np.concatenate([va, te])
    nva = len(va)
    log(f"== Mahalanobis++ / KPCA on CLINC150 | {len(tr_in)} in-scope train, val {len(va)} "
        f"({int(y_oos[va].sum())} oos), test {len(te)} ({int(y_oos[te].sum())} oos) | depths {args.depths} | "
        f"trunk {trunk.device}, fits {dev}")
    t0 = time.time()
    feats = pooled(trunk, [e.text for e in allx], args.depths, ds.max_len, log=log)
    res = {"args": vars(args), "n": {"train_in": int(len(tr_in)), "val": int(nva), "test": int(len(te)),
                                     "test_oos": int(y_oos[te].sum())}, "depths": {}}
    val_scores, test_scores = {}, {}
    for d in args.depths:
        t1 = time.time()
        Xraw = feats[d]
        S = scores_at_depth(Xraw, tr_in, rows_eval, y150, dev, args)
        Xz = zscore(Xraw, tr_in)
        pr = Probe(Xz, y150, tr_in, va_in, 150, dev, steps=args.steps, seed=args.seed)
        S["msp"] = 1 - pr.probs(Xz[rows_eval]).max(-1)
        norms = Xraw[te].norm(dim=-1).numpy()
        te_in_local = y_oos[te] == 0
        per = {}
        for name, s in S.items():
            sv, st = s[:nva], s[nva:]
            m = ood_metrics(st, y_oos[te])
            m["val_auroc"] = ood_metrics(sv, y_oos[va])["auroc"] if y_oos[va].any() else float("nan")
            if name.startswith(("maha", "rmd")):
                m["norm_spearman_in_scope"] = float(spearmanr(norms[te_in_local], st[te_in_local]).correlation)
            per[name] = m
            val_scores.setdefault(name, {})[d] = sv
            test_scores.setdefault(name, {})[d] = st
        res["depths"][d] = per
        show = ["maha_z", "maha_raw", "maha_l2", "maha_cl2", "maha_zl2", "rmd_cl2", "kpca_cos_0.9",
                "kpca_rff_c1.0_0.9", "knn1", "msp"]
        log(f"-- depth {d} ({time.time() - t1:.1f}s): " + " | ".join(f"{k} {per[k]['auroc']:.3f}" for k in show))
        log("   norm-score Spearman (in-scope test): " + " | ".join(
            f"{k} {per[k]['norm_spearman_in_scope']:+.2f}" for k in ("maha_z", "maha_raw", "maha_l2", "maha_cl2", "maha_zl2")))
        log.dump(res)

    # per score: validation-best depth, then OOS F1 three ways
    y_te, y_va = y_oos[te], y_oos[va]
    prior_src = float(y_oos[idx["train"]].mean())
    summary = {}
    for name in val_scores:
        best_d = max(args.depths, key=lambda d: res["depths"][d][name]["val_auroc"])
        sv, st = val_scores[name][best_d], test_scores[name][best_d]
        cands = np.unique(np.quantile(sv, np.linspace(0, 1, 401)))
        f1s = [f1_at((sv > c).astype(int), y_va)["f1"] for c in cands]
        thr = float(cands[int(np.argmax(f1s))])
        f1_val = f1_at((st > thr).astype(int), y_te)
        to_p = score_to_prob(sv, y_va)
        # the logistic calibration is fit at the validation OOS rate; EM moves it to the test pool's rate
        em = binary_em_f1(to_p(st), float(y_va.mean()), y_te)
        k = int(round(y_te.mean() * len(st)))
        oracle = f1_at((st >= np.sort(st)[::-1][k - 1]).astype(int), y_te)
        summary[name] = {"best_depth_by_val_auroc": int(best_d), **res["depths"][best_d][name],
                         "best_test_auroc_any_depth": max(res["depths"][d][name]["auroc"] for d in args.depths),
                         "f1_val_threshold": f1_val, "f1_calibrated_em": em, "f1_oracle_rate": oracle}
    res["summary"] = summary
    res["prior_train_oos"] = prior_src
    order = sorted(summary, key=lambda n: -summary[n]["auroc"])
    log("== summary (score @ val-best depth: test AUROC | best AUROC any depth | F1 val-thr / cal+EM / oracle-rate)")
    for n in order:
        s = summary[n]
        log(f"   {n:22s} @{s['best_depth_by_val_auroc']:>2}: {s['auroc']:.3f} | {s['best_test_auroc_any_depth']:.3f} | "
            f"{s['f1_val_threshold']['f1']:.3f} / {s['f1_calibrated_em']['f1']:.3f} "
            f"(est rate {s['f1_calibrated_em']['est_oos_rate']:.3f}) / {s['f1_oracle_rate']['f1']:.3f}")
    res["wall_s"] = round(time.time() - t0, 1)
    log.dump(res)
    log(f"done in {res['wall_s']}s -> {out}")


if __name__ == "__main__":
    main()
