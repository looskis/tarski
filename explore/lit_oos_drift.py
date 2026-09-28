"""Entry 5 (lit_incremental_oos.md): a label-free harmful-drift alarm plus a label-efficient confirmation.

  - Amoukou et al., "Sequential Harmful Shift Detection Without Labels" (NeurIPS 2024, arXiv 2412.12910):
    train an error estimator on labelled source data, use its output as a proxy for the production error,
    and alarm when a time-uniform lower confidence bound on the production proxy-risk exceeds the source
    upper bound plus a tolerance (Podkopaev & Ramdas confidence sequences; false alarms controlled at any
    stopping time). A quantile variant (share of messages whose proxy exceeds the source 90th percentile)
    had the most power in their experiments.
  - Zrnic & Candes, "Active Statistical Inference" (ICML 2024): once an alarm fires, label the messages the
    model is unsure about (probability proportional to sqrt(e(1-e))) and get a valid confidence interval on
    current accuracy from far fewer labels than uniform sampling.
Tarski already computes several label-free error signals per message for free, which feed the estimator:
max softmax, entropy, margin, cross-depth disagreement (probes at 5 depths), Mahalanobis++ and kNN
distance at depth 4.

Setup (CLINC150): 20 intents (2 per domain) are held out as "future routes"; the routing branch is a
130-way probe at --primary-depth. The error estimator is fitted on half of the source validation data
(known-intent + out-of-scope messages), and the source bounds come from the other half.
Streams of 3,000 messages, shift at message 1,000, ~3% out-of-scope throughout (as in validation):
  S0 none         no shift (false alarms)
  S1 new_routes   25% of post-shift messages come from the 20 held-out intents (harmful: always misrouted)
  S2 prior        post-shift known-intent popularity ~ Dirichlet(0.3) (benign-ish: little accuracy change)
  S3 typos_p      post-shift messages get keyboard typos on a share p of letters (p = 0.1, 0.2)
Monitors:
  proxy_mean     PrPl-EB lower confidence sequence (Waudby-Smith & Ramdas) on the mean proxy error vs the
                 source upper bound + 0.02 (alpha 0.05 each side)
  proxy_quant    the same on 1{proxy > source 90th percentile}
  oos_rate       the same on 1{Mahalanobis++ > 95%-in-scope threshold}
  msp_cusum      CUSUM on 1 - max softmax, threshold calibrated on separate no-shift streams (5% false alarms)
  mmd            Gaussian-kernel MMD permutation test of the last 200 depth-4 states vs a source sample,
                 every 200 messages, Bonferroni over checkpoints ("Failing Loudly", Rabanser et al. 2019)
Ground truth "harmful" = post-shift accuracy more than 3 points below the source accuracy.
Active inference: on the 1,000 post-shift messages, 95% intervals for accuracy from n in {25, 50, 100, 200}
labels: uniform, prediction-powered uniform, and active sampling; coverage, width, and how often the upper
bound certifies a drop (upper < source accuracy - 0.03).
Expected: proxy tests fire on S1/S3 and stay quiet on S0/S2; MMD fires on everything; active inference needs
2-3x fewer labels than uniform for the same width.

Usage:
  .venv/bin/python explore/lit_oos_drift.py --smoke
  .venv/bin/python explore/lit_oos_drift.py --out results/tarski/explore_lit_drift.json   # ~10 min on an A10
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_oos_common import (Gauss, Logger, Probe, get_trunk, knn_dist, l2n, load_ds, pick_device, pooled,
                            seed_all, split_index, typo_noise)

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

DEPTHS = [4, 8, 12, 16, 22]


# ---------------------------------------------------------------------------------------------------
# confidence bounds
# ---------------------------------------------------------------------------------------------------

def eb_ucb(x: np.ndarray, alpha: float) -> float:
    """Maurer-Pontil empirical-Bernstein upper bound on the mean of [0,1] data (fixed sample)."""
    n = len(x)
    v = x.var(ddof=1) if n > 1 else 0.25
    return float(x.mean() + math.sqrt(2 * v * math.log(2 / alpha) / n) + 7 * math.log(2 / alpha) / (3 * (n - 1)))


def prpl_eb_lcb(x: np.ndarray, alpha: float, c: float = 0.5) -> np.ndarray:
    """Predictable-plug-in empirical-Bernstein lower confidence sequence for the running mean of [0,1]
    data (Waudby-Smith & Ramdas, JRSSB 2024), valid uniformly over time."""
    t = np.arange(1, len(x) + 1)
    mu_hat = (0.5 + np.cumsum(x)) / (t + 1)
    sig2 = (0.25 + np.cumsum((x - mu_hat) ** 2)) / (t + 1)
    mu_prev = np.r_[0.5, mu_hat[:-1]]
    sig_prev = np.r_[0.25, sig2[:-1]]
    lam = np.minimum(np.sqrt(2 * math.log(1 / alpha) / (sig_prev * t * np.log(1 + t))), c)
    v = 4 * (x - mu_prev) ** 2
    psi = (-np.log(1 - lam) - lam) / 4
    return (np.cumsum(lam * x) - (math.log(1 / alpha) + np.cumsum(v * psi))) / np.cumsum(lam)


def first_true(mask: np.ndarray):
    w = np.where(mask)[0]
    return int(w[0]) if len(w) else None


# ---------------------------------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------------------------------

def mmd_perm_p(A: torch.Tensor, B: torch.Tensor, bw: float, n_perm: int, g: torch.Generator) -> float:
    Z = torch.cat([A, B])
    K = torch.exp(-torch.cdist(Z, Z).pow(2) / (2 * bw ** 2))
    n, m = len(A), len(B)

    def stat(ix):
        Kp = K[ix][:, ix]
        kxx, kyy, kxy = Kp[:n, :n], Kp[n:, n:], Kp[:n, n:]
        return (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1)) \
            - 2 * kxy.mean()

    base = stat(torch.arange(n + m, device=Z.device))
    cnt = 0
    for _ in range(n_perm):
        cnt += int(stat(torch.randperm(n + m, generator=g).to(Z.device)) >= base)
    return (1 + cnt) / (1 + n_perm)


# ---------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--primary-depth", type=int, default=22)
    ap.add_argument("--n-heldout-per-domain", type=int, default=2)
    ap.add_argument("--stream-len", type=int, default=3000)
    ap.add_argument("--t0", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--oos-rate", type=float, default=0.03)
    ap.add_argument("--new-route-rate", type=float, default=0.25)
    ap.add_argument("--noise", type=float, nargs="*", default=[0.1, 0.2])
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--mmd-every", type=int, default=200)
    ap.add_argument("--mmd-ref", type=int, default=400)
    ap.add_argument("--mmd-perm", type=int, default=100)
    ap.add_argument("--budgets", type=int, nargs="*", default=[25, 50, 100, 200])
    ap.add_argument("--ai-reps", type=int, default=100)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.stream_len, args.t0, args.reps, args.mmd_every, args.mmd_ref, args.mmd_perm = 600, 200, 3, 200, 100, 20
        args.budgets, args.ai_reps = [25, 50], 20
    out = args.out or ("results/tarski/explore_lit_drift_smoke.json" if args.smoke else "results/tarski/explore_lit_drift.json")
    log = Logger(out)
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = pick_device(args.smoke)
    trunk = get_trunk(args.smoke)
    t_start = time.time()
    ds = load_ds("clinc150", args.smoke, args.seed)
    allx, idx = split_index(ds)
    intents = ds.tasks["intent"].labels
    oos_i = intents.index("oos")
    import json as _json
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tarski", "resources",
                           "clinc_domains.json")) as f:
        domains = _json.load(f)
    held = []
    for dname in sorted(domains):
        its = sorted(domains[dname])
        held += list(rng.choice(its, args.n_heldout_per_domain, replace=False))
    held_ids = {intents.index(h) for h in held}
    known = [i for i in range(len(intents)) if i != oos_i and i not in held_ids]
    kmap = {i: j for j, i in enumerate(known)}
    CK = len(known)
    y_int = np.array([e.y["intent"] for e in allx])
    ylab = np.array([kmap.get(i, -1) for i in y_int])              # -1 = held-out intent or out-of-scope
    is_held = np.isin(y_int, list(held_ids))
    is_oos = y_int == oos_i
    tr, va, te = idx["train"], idx["val"], idx["test"]
    tr_k = tr[ylab[tr] >= 0]
    va_src = va[~is_held[va]]                                       # the source world has no held-out intents
    known_test = te[ylab[te] >= 0]
    oos_test = te[is_oos[te]]
    held_pool = np.concatenate([tr[is_held[tr]], te[is_held[te]]])
    log(f"== drift monitors on CLINC150 | {CK} known intents, {len(held_ids)} held out ({', '.join(held[:6])}...) | "
        f"primary probe depth {args.primary_depth} | fits {dev}")

    # universe of messages: every CLINC message + noised copies of the known-intent test messages
    texts = [e.text for e in allx]
    feats = pooled(trunk, texts, DEPTHS, ds.max_len, log=log)
    noised = {}
    for p in args.noise:
        r = random.Random(args.seed + int(p * 1000))
        ntexts = [typo_noise(allx[i].text, p, r) for i in known_test]
        noised[p] = pooled(trunk, ntexts, DEPTHS, ds.max_len, log=log)
    U_feats = {d: torch.cat([feats[d]] + [noised[p][d] for p in args.noise]) for d in DEPTHS}
    U_label = np.concatenate([ylab] + [ylab[known_test] for _ in args.noise])
    n_all = len(allx)
    noised_offset = {p: n_all + k * len(known_test) for k, p in enumerate(args.noise)}

    # probes at every depth (known intents only), signals for the whole universe
    mu = {d: feats[d][tr_k].mean(0, keepdim=True) for d in DEPTHS}
    sd = {d: feats[d][tr_k].std(0, keepdim=True).clamp_min(1e-4) for d in DEPTHS}
    Z = {d: (U_feats[d] - mu[d]) / sd[d] for d in DEPTHS}
    va_k = va[ylab[va] >= 0]
    ylab_u = U_label
    probes = {d: Probe(Z[d][:n_all], ylab, tr_k, va_k, CK, dev, steps=args.steps, seed=args.seed) for d in DEPTHS}
    Pp = probes[args.primary_depth].probs(Z[args.primary_depth])
    pred = Pp.argmax(-1)
    correct = (pred == ylab_u) & (ylab_u >= 0)
    srt = np.sort(Pp, -1)
    sig = {"msp": srt[:, -1], "entropy": -(Pp * np.log(Pp + 1e-12)).sum(-1), "margin": srt[:, -1] - srt[:, -2]}
    other = [d for d in DEPTHS if d != args.primary_depth]
    sig["disagree"] = np.mean([probes[d].probs(Z[d]).argmax(-1) != pred for d in other], 0)
    cen = feats[4][tr_k].mean(0, keepdim=True)
    g4 = Gauss(l2n(feats[4][tr_k] - cen), ylab[tr_k], CK, 0.1)
    sig["maha"] = g4.min_dist(l2n(U_feats[4] - cen))
    sig["knn10"] = knn_dist(feats[4][tr_k], U_feats[4], (10,), dev)[10]
    S = np.stack([sig[k] for k in ("msp", "entropy", "margin", "disagree", "maha", "knn10")], 1)

    # error estimator on source half A; source bounds from half B
    perm = rng.permutation(va_src)
    A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
    m_, s_ = S[A].mean(0), S[A].std(0) + 1e-9
    est = LogisticRegression(C=1.0, max_iter=1000).fit((S[A] - m_) / s_, (~correct[A]).astype(int))
    e = est.predict_proba((S - m_) / s_)[:, 1]
    q90 = float(np.quantile(e[A], 0.9))
    thr_oos = float(np.quantile(sig["maha"][va_k], 0.95))
    xs = {"proxy_mean": e, "proxy_quant": (e > q90).astype(float), "oos_rate": (sig["maha"] > thr_oos).astype(float)}
    src_ucb = {k: eb_ucb(v[B], args.alpha) for k, v in xs.items()}
    acc_src = float(correct[B].mean())
    msp_mu, msp_sd = float((1 - sig["msp"][B]).mean()), float((1 - sig["msp"][B]).std())
    from sklearn.metrics import roc_auc_score
    res = {"args": vars(args), "held_out_intents": held,
           "source": {"acc": acc_src, "true_error": 1 - acc_src, "proxy_mean": float(e[B].mean()),
                      "ucb": src_ucb, "error_estimator_auroc_B": float(roc_auc_score((~correct[B]).astype(int), e[B]))}}
    log(f"   source accuracy (half B) {acc_src:.4f}; proxy mean {e[B].mean():.4f}; error-estimator AUROC on B "
        f"{res['source']['error_estimator_auroc_B']:.3f}; UCBs " + ", ".join(f"{k} {v:.4f}" for k, v in src_ucb.items()))

    # MMD features: depth-4 z-states projected to 32 PCA dims (fitted on source)
    Z4 = Z[4]
    U_, S_, V_ = torch.pca_lowrank(Z4[va_src], q=32, center=True)
    mmd_x = ((Z4 - Z4[va_src].mean(0, keepdim=True)) @ V_).to(dev)
    ref_rows = rng.choice(B, min(args.mmd_ref, len(B)), replace=False)
    bw = float(torch.cdist(mmd_x[ref_rows], mmd_x[ref_rows]).median())
    gperm = torch.Generator().manual_seed(args.seed)

    def make_stream(kind: str, r: np.random.Generator) -> np.ndarray:
        N, t0 = args.stream_len, args.t0
        rows = np.empty(N, dtype=int)
        w_known = None
        if kind == "prior":
            w_cls = r.dirichlet(np.full(CK, 0.3))
            w_known = w_cls[ylab[known_test]]
            w_known = w_known / w_known.sum()
        for t in range(N):
            post = t >= t0
            if r.random() < args.oos_rate:
                rows[t] = r.choice(oos_test)
                continue
            if post and kind == "new_routes" and r.random() < args.new_route_rate:
                rows[t] = r.choice(held_pool)
                continue
            if post and kind == "prior":
                rows[t] = known_test[r.choice(len(known_test), p=w_known)]
                continue
            k = r.integers(len(known_test))
            if post and kind.startswith("typos_"):
                rows[t] = noised_offset[float(kind.split("_")[1])] + k
            else:
                rows[t] = known_test[k]
        return rows

    kinds = ["none", "new_routes", "prior"] + [f"typos_{p}" for p in args.noise]
    # CUSUM threshold from separate no-shift calibration streams
    k_ref = 0.5 * msp_sd

    def cusum_path(rows):
        x = (1 - sig["msp"][rows]) - msp_mu - k_ref
        G = np.zeros(len(x))
        g = 0.0
        for t, v in enumerate(x):
            g = max(0.0, g + v)
            G[t] = g
        return G
    cal_max = [cusum_path(make_stream("none", np.random.default_rng(10_000 + i))).max() for i in range(max(20, args.reps))]
    h_cusum = float(np.quantile(cal_max, 0.95))
    checkpoints = list(range(args.mmd_every, args.stream_len + 1, args.mmd_every))
    streams = {}
    for kind in kinds:
        per = {m: {"false_alarm": [], "detected": [], "delay": []} for m in
               ("proxy_mean", "proxy_quant", "oos_rate", "msp_cusum", "mmd")}
        harmful, post_acc, ai = [], [], []
        for rep in range(args.reps):
            r = np.random.default_rng(args.seed * 1000 + rep + 7 * kinds.index(kind))
            rows = make_stream(kind, r)
            t0 = args.t0
            pa = float(correct[rows[t0:]].mean())
            post_acc.append(pa)
            harmful.append(pa < acc_src - 0.03)
            alarms = {}
            for m, x in xs.items():
                lcb = prpl_eb_lcb(x[rows], args.alpha)
                alarms[m] = first_true(lcb > src_ucb[m] + args.tol)
            alarms["msp_cusum"] = first_true(cusum_path(rows) > h_cusum)
            a_mmd = None
            for c in checkpoints:
                win = rows[c - args.mmd_every:c]
                p = mmd_perm_p(mmd_x[win], mmd_x[ref_rows], bw, args.mmd_perm, gperm)
                if p < args.alpha / len(checkpoints):
                    a_mmd = c - 1
                    break
            alarms["mmd"] = a_mmd
            for m, a in alarms.items():
                per[m]["false_alarm"].append(a is not None and a < t0)
                det = a is not None and a >= t0
                per[m]["detected"].append(det)
                if det:
                    per[m]["delay"].append(a - t0)
            # active inference on the post-shift window
            W = rows[t0:t0 + 1000]
            ai.append(active_inference(correct[W].astype(float), e[W], args.budgets, args.ai_reps, acc_src, r))
        summ = {m: {"false_alarm_rate": float(np.mean(v["false_alarm"])), "detect_rate": float(np.mean(v["detected"])),
                    "mean_delay": float(np.mean(v["delay"])) if v["delay"] else None} for m, v in per.items()}
        ai_s = {str(n): {meth: {k: float(np.mean([a[n][meth][k] for a in ai])) for k in ai[0][n][meth]}
                         for meth in ai[0][n]} for n in args.budgets}
        streams[kind] = {"harmful_rate": float(np.mean(harmful)), "post_shift_acc": float(np.mean(post_acc)),
                         "monitors": summ, "active_inference": ai_s}
        log(f"-- {kind}: post-shift acc {np.mean(post_acc):.4f} (source {acc_src:.4f}), harmful {np.mean(harmful):.2f} | "
            + " | ".join(f"{m} FA {v['false_alarm_rate']:.2f} det {v['detect_rate']:.2f}"
                         + (f" delay {v['mean_delay']:.0f}" if v["mean_delay"] is not None else "") for m, v in summ.items()))
        for n in args.budgets:
            log(f"   labels {n}: " + " | ".join(f"{meth} width {v['width']:.3f} cover {v['coverage']:.2f} certify-drop {v['certify_drop']:.2f}"
                                             for meth, v in ai_s[str(n)].items()))
        res["streams"] = streams
        res["cusum_threshold"] = h_cusum
        log.dump(res)
    res["wall_s"] = round(time.time() - t_start, 1)
    log.dump(res)
    log(f"done in {res['wall_s']}s -> {out}")


def active_inference(y: np.ndarray, e: np.ndarray, budgets, reps: int, acc_src: float, rng) -> dict:
    """95% intervals for the window's accuracy from n labels. f = 1 - e is the predicted correctness."""
    N = len(y)
    theta = y.mean()
    f = 1 - e
    u = np.sqrt(np.clip(e * (1 - e), 1e-6, None))
    out = {}
    for n in budgets:
        res = {k: {"width": [], "coverage": [], "certify_drop": []} for k in ("uniform", "ppi_uniform", "active")}
        for _ in range(reps):
            # uniform: plain mean of n labels (finite-population correction)
            s = rng.choice(N, n, replace=False)
            m = y[s].mean()
            half = 1.96 * math.sqrt(max(m * (1 - m), 1 / n) / n * (1 - n / N))
            add(res["uniform"], m, half, theta, acc_src)
            for name, pi in (("ppi_uniform", np.full(N, n / N)), ("active", np.minimum(1.0, n * u / u.sum()))):
                xi = rng.random(N) < pi
                terms = f + xi / pi * (y - f)
                est = terms.mean()
                half = 1.96 * terms.std(ddof=1) / math.sqrt(N)
                add(res[name], est, half, theta, acc_src)
        out[n] = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in res.items()}
    return out


def add(d, est, half, theta, acc_src):
    d["width"].append(2 * half)
    d["coverage"].append(float(est - half <= theta <= est + half))
    d["certify_drop"].append(float(est + half < acc_src - 0.03))


if __name__ == "__main__":
    main()
