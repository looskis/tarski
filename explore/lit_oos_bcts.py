"""Entry 7 (lit_incremental_oos.md): bias-corrected calibration + prior tracking for every routing branch.

Alexandari, Kundaje & Shrikumar, "Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at
Label Shift Adaptation" (ICML 2020, arXiv 1901.06852): Saerens EM needs class-wise calibration. Temperature
plus per-class bias (BCTS) makes EM beat BBSE/RLLS; a single temperature (what tarski fits) often does
not. Route popularity drifts all the time (incident spikes, launches), so the correction should run on every
branch and track the prior over time; Baby et al., "Online Label Shift: Optimal Dynamic Regret meets
Practical Algorithms" (NeurIPS 2023, arXiv 2305.19570) is the reference for unsupervised tracking. Their
FLH/online-regression tracker is NOT reimplemented here: the streaming arm compares simpler causal trackers
(windowed EM, exponentially weighted online EM) against a static model and an oracle.

Calibrators, fitted on validation logits of a linear probe:
  none   T = 1
  ts     one temperature (tarski's current calibration)
  bcts   temperature + per-class bias, L2 on the biases (lambda by 2-fold CV on validation NLL: validation
         has only 13-20 rows per class here, so unregularised biases overfit)
  vs     per-class scale + bias (vector scaling), same regularisation
Estimators of the target prior: EM with each calibrator; BBSE (Lipton et al. 2018) from the validation
confusion matrix. EM's source prior is the training prior for none/ts (a temperature cannot move the prior
the probe learned) and the validation prior for bcts/vs (their biases are fitted there).

Experiments:
  dirichlet   test sets resampled to Dirichlet(alpha) class priors, alpha in {0.1, 0.3, 1, 10}, 20 draws of
              3,000 messages: L1 error of the estimated prior, accuracy change from the EM correction
  clinc_real  CLINC150 151-way on the real test set (18.2% out-of-scope): prior error, OOS F1
  stream      10 segments x 500 messages with a new Dirichlet(0.3) prior per segment; causal trackers
              (static, global EM, windowed EM w=200/1000 re-estimated every 50 messages, the same with a
              MAP Dirichlet prior of C pseudo-counts toward the source prior, EWMA online EM eta=0.01/0.03,
              oracle segment prior), mean accuracy over 5 streams
Expected: BCTS cuts prior-estimation error by 20-50% vs TS and adds 1-5 accuracy points at alpha <= 0.3;
possible overfitting at 13 examples per class is reported, not hidden.

Usage:
  .venv/bin/python explore/lit_oos_bcts.py --smoke
  .venv/bin/python explore/lit_oos_bcts.py --out results/tarski/explore_lit_bcts.json   # ~5 min on an A10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_oos_common import (Logger, Probe, class_prior, f1_at, get_trunk, load_ds, pick_device, pooled, saerens_em,
                            seed_all, split_index, zscore)

import numpy as np
import torch
import torch.nn.functional as F

LAMS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)


def fit_cal(z: torch.Tensor, y: torch.Tensor, kind: str, lam: float = 0.0):
    C = z.shape[1]
    log_t = torch.zeros(1, requires_grad=True)
    b = torch.zeros(C, requires_grad=True)
    a = torch.zeros(C, requires_grad=True)             # vs: log per-class scale
    params = {"none": [], "ts": [log_t], "bcts": [log_t, b], "vs": [a, b]}[kind]
    if not params:
        return lambda zz: zz

    def f(zz):
        if kind == "ts":
            return zz / log_t.exp()
        if kind == "bcts":
            return zz / log_t.exp() + b
        return zz * a.exp() + b

    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(f(z), y) + lam * ((b ** 2).sum() + (a ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return lambda zz: f(zz).detach()


def fit_cal_cv(z, y, kind, seed):
    if kind in ("none", "ts"):
        return fit_cal(z, y, kind), 0.0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    folds = np.array_split(perm, 2)
    best = None
    for lam in LAMS:
        nll = 0.0
        for k in (0, 1):
            trn, tst = folds[1 - k], folds[k]
            g = fit_cal(z[trn], y[trn], kind, lam)
            nll += float(F.cross_entropy(g(z[tst]), y[tst]))
        if best is None or nll < best[0]:
            best = (nll, lam)
    return fit_cal(z, y, kind, best[1]), best[1]


def bbse(pred_val: np.ndarray, y_val: np.ndarray, pred_tgt: np.ndarray, C: int) -> np.ndarray:
    Cm = np.zeros((C, C))
    np.add.at(Cm, (pred_val, y_val), 1.0)
    Cm /= len(y_val)
    mu = np.bincount(pred_tgt, minlength=C) / len(pred_tgt)
    w, *_ = np.linalg.lstsq(Cm, mu, rcond=None)
    w = np.clip(w, 0, None)
    pi_val = class_prior(y_val, C)
    p = w * pi_val
    return p / max(p.sum(), 1e-12)


def sample_pool(y_te: np.ndarray, pi: np.ndarray, N: int, rng) -> np.ndarray:
    C = len(pi)
    by = [np.where(y_te == c)[0] for c in range(C)]
    avail = np.array([len(b) > 0 for b in by])
    p = pi * avail
    p /= p.sum()
    counts = rng.multinomial(N, p)
    return np.concatenate([rng.choice(by[c], counts[c], replace=True) for c in range(C) if counts[c] > 0])


def map_em(P: np.ndarray, ps: np.ndarray, beta: float, iters: int = 100) -> np.ndarray:
    """EM with a Dirichlet prior of `beta` pseudo-counts centred on the source prior (MAP label-shift
    estimation): stays near the source prior until the window holds enough evidence."""
    pt = ps.copy()
    for _ in range(iters):
        a = P * (pt / ps)
        a /= a.sum(-1, keepdims=True)
        pt = (a.sum(0) + beta * ps) / (len(P) + beta)
    return pt


def run_stream(P: np.ndarray, y: np.ndarray, prior_src: np.ndarray, C: int, rng, n_seg: int, seg_len: int,
               alpha: float, recompute: int = 50) -> dict:
    segs, pis = [], []
    for _ in range(n_seg):
        pi = rng.dirichlet(np.full(C, alpha))
        rows = sample_pool(y, pi, seg_len, rng)
        rng.shuffle(rows)
        segs.append(rows)
        pis.append(pi)
    rows = np.concatenate(segs)
    Ps, ys = P[rows], y[rows]
    T = len(rows)
    ps = np.clip(prior_src, 1e-12, None)

    def adjust(p, pri):
        a = p * (pri / ps)
        return a / a.sum(-1, keepdims=True)

    out = {}
    out["static"] = float((Ps.argmax(-1) == ys).mean())
    seg_of = np.repeat(np.arange(n_seg), seg_len)
    oracle = np.stack([pis[s] for s in seg_of])
    out["oracle_segment_prior"] = float((np.array([adjust(Ps[t], oracle[t]) for t in range(T)]).argmax(-1) == ys).mean())
    for name, w in (("global_em", None), ("window_em_200", 200), ("window_em_1000", 1000)):
        pri, correct = ps.copy(), []
        for t in range(T):
            if t > 0 and t % recompute == 0:
                lo = 0 if w is None else max(0, t - w)
                _, pri = saerens_em(Ps[lo:t], ps, iters=100)
            correct.append(adjust(Ps[t], pri).argmax() == ys[t])
        out[name] = float(np.mean(correct))
    for name, w in (("global_mapem", None), ("window_mapem_200", 200), ("window_mapem_1000", 1000)):
        pri, correct = ps.copy(), []
        for t in range(T):
            if t > 0 and t % recompute == 0:
                lo = 0 if w is None else max(0, t - w)
                pri = map_em(Ps[lo:t], ps, beta=float(C))
            correct.append(adjust(Ps[t], pri).argmax() == ys[t])
        out[name] = float(np.mean(correct))
    for eta in (0.01, 0.03):
        pri, correct = ps.copy(), []
        for t in range(T):
            a = adjust(Ps[t], pri)
            correct.append(a.argmax() == ys[t])
            pri = (1 - eta) * pri + eta * a
        out[f"ewma_em_{eta}"] = float(np.mean(correct))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--depth", type=int, default=22)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.1, 0.3, 1.0, 10.0])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--streams", type=int, default=5)
    ap.add_argument("--seg-len", type=int, default=500)
    ap.add_argument("--n-seg", type=int, default=10)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.draws, args.pool, args.streams, args.seg_len, args.n_seg, args.alphas = 3, 500, 1, 100, 3, [0.3, 1.0]
    out = args.out or ("results/tarski/explore_lit_bcts_smoke.json" if args.smoke else "results/tarski/explore_lit_bcts.json")
    log = Logger(out)
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = pick_device(args.smoke)
    trunk = get_trunk(args.smoke)
    t0 = time.time()
    res = {"args": vars(args), "decisions": {}}
    for dname in ("banking77", "clinc150"):
        ds = load_ds(dname, args.smoke, args.seed)
        allx, idx = split_index(ds)
        X = pooled(trunk, [e.text for e in allx], [args.depth], ds.max_len, log=log)[args.depth]
        tr, va, te = idx["train"], idx["val"], idx["test"]
        if dname == "clinc150":
            y = np.array([e.y["intent"] for e in allx])          # 151-way, "oos" is a label
            C = len(ds.tasks["intent"].labels)
            oos_label = ds.tasks["intent"].labels.index("oos")
        else:
            y = np.array([e.y["intent"] for e in allx])
            C = len(ds.tasks["intent"].labels)
            oos_label = None
        Xz = zscore(X, tr)
        pr = Probe(Xz, y, tr, va, C, dev, steps=args.steps, seed=args.seed)
        zv, zt = pr.logits(Xz[va]).detach(), pr.logits(Xz[te]).detach()
        yv, yt = torch.as_tensor(y[va]), y[te]
        pri_tr, pri_va = class_prior(y[tr], C), class_prior(y[va], C)
        cal = {}
        for kind in ("none", "ts", "bcts", "vs"):
            g, lam = fit_cal_cv(zv, yv, kind, args.seed)
            Pv = torch.softmax(g(zv), -1).numpy().astype(np.float64)
            Pt = torch.softmax(g(zt), -1).numpy().astype(np.float64)
            src = pri_tr if kind in ("none", "ts") else pri_va
            cal[kind] = {"Pt": Pt, "Pv": Pv, "src": np.clip(src, 1e-9, None), "lam": lam,
                         "val_nll": float(-np.log(Pv[np.arange(len(yv)), yv.numpy()] + 1e-12).mean())}
        log(f"== {dname} @ depth {args.depth}: {C} classes | test acc {float((cal['ts']['Pt'].argmax(-1) == yt).mean()):.4f} | "
            + " | ".join(f"{k} val NLL {v['val_nll']:.3f} (lam {v['lam']})" for k, v in cal.items()))
        dres = {"calibrators": {k: {"lam": v["lam"], "val_nll": v["val_nll"]} for k, v in cal.items()}, "dirichlet": {}}
        pred_val_ts = cal["ts"]["Pv"].argmax(-1)
        for alpha in args.alphas:
            acc = {k: [] for k in list(cal) + ["bbse"]}
            l1 = {k: [] for k in list(cal) + ["bbse"]}
            base = {k: [] for k in cal}
            for _ in range(args.draws):
                pi = rng.dirichlet(np.full(C, alpha))
                rows = sample_pool(yt, pi, args.pool, rng)
                emp = class_prior(yt[rows], C)
                for k, v in cal.items():
                    adj, pt = saerens_em(v["Pt"][rows], v["src"])
                    l1[k].append(float(np.abs(pt - emp).sum()))
                    acc[k].append(float((adj.argmax(-1) == yt[rows]).mean()))
                    base[k].append(float((v["Pt"][rows].argmax(-1) == yt[rows]).mean()))
                pb = bbse(pred_val_ts, y[va], cal["ts"]["Pt"][rows].argmax(-1), C)
                a = cal["ts"]["Pt"][rows] * (np.clip(pb, 1e-12, None) / cal["ts"]["src"])
                l1["bbse"].append(float(np.abs(pb - emp).sum()))
                acc["bbse"].append(float((a.argmax(-1) == yt[rows]).mean()))
            r = {"prior_l1": {k: float(np.mean(v)) for k, v in l1.items()},
                 "acc_after": {k: float(np.mean(v)) for k, v in acc.items()},
                 "acc_before": {k: float(np.mean(v)) for k, v in base.items()}}
            r["acc_gain"] = {k: r["acc_after"][k] - r["acc_before"].get(k, r["acc_before"]["ts"]) for k in acc}
            dres["dirichlet"][alpha] = r
            log(f"-- {dname} Dirichlet({alpha}): prior L1 " + " | ".join(f"{k} {v:.3f}" for k, v in r["prior_l1"].items())
                + " || acc gain " + " | ".join(f"{k} {v:+.4f}" for k, v in r["acc_gain"].items())
                + f" (before, ts: {r['acc_before']['ts']:.4f})")
        if oos_label is not None:
            real = {}
            yo = (yt == oos_label).astype(int)
            for k, v in cal.items():
                adj, pt = saerens_em(v["Pt"], v["src"])
                real[k] = {"est_oos_share": float(pt[oos_label]), "true_oos_share": float(yo.mean()),
                           "prior_l1": float(np.abs(pt - class_prior(yt, C)).sum()),
                           "oos_f1_before": f1_at((v["Pt"].argmax(-1) == oos_label).astype(int), yo)["f1"],
                           "oos_f1_after": f1_at((adj.argmax(-1) == oos_label).astype(int), yo)["f1"],
                           "acc_before": float((v["Pt"].argmax(-1) == yt).mean()),
                           "acc_after": float((adj.argmax(-1) == yt).mean())}
            dres["clinc_real"] = real
            log("-- CLINC real shift: " + " | ".join(
                f"{k}: OOS share {v['est_oos_share']:.3f}, OOS F1 {v['oos_f1_before']:.3f}->{v['oos_f1_after']:.3f}, "
                f"acc {v['acc_before']:.4f}->{v['acc_after']:.4f}" for k, v in real.items()))
        streams = {}
        for kind in ("ts", "bcts"):
            rs = [run_stream(cal[kind]["Pt"], yt, cal[kind]["src"], C, rng, args.n_seg, args.seg_len, 0.3)
                  for _ in range(args.streams)]
            streams[kind] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            log(f"-- {dname} stream ({kind}): " + " | ".join(f"{k} {v:.4f}" for k, v in streams[kind].items()))
        dres["stream"] = streams
        res["decisions"][dname] = dres
        log.dump(res)
    res["wall_s"] = round(time.time() - t0, 1)
    log.dump(res)
    log(f"done in {res['wall_s']}s -> {out}")


if __name__ == "__main__":
    main()
