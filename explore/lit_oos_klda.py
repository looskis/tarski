"""Entry 1 (lit_incremental_oos.md): kernelised sufficient-statistics routes (KLDA / RanPAC) on trunk taps.

Momeni, Mazumder & Liu, "Continual Learning Using a Kernel-Based Method Over Foundation Models" (KLDA,
AAAI 2025, arXiv 2412.15571) put fixed random Fourier features (an RBF kernel, D~5000) between frozen
text-encoder features and an LDA head (class means + one shared covariance). New classes are new
statistics: no gradient steps, no replay, and the result is order-independent, so class-incremental
accuracy equals joint accuracy. They report on frozen BART-base: CLINC 93.7 (LDA) -> 95.9 (KLDA) -> 96.6
(5-seed ensemble), Banking77 89.1 -> 92.2 -> 93.0. RanPAC (McDonnell et al., NeurIPS 2023, arXiv 2307.02251)
uses a ReLU random projection and a ridge (Gram-inverse) head instead.

Because the RFF map is fixed and seeded (a property of the trunk, not of any user's route), KLDA statistics
stay additive: geometry idea 10's exact merge / unlearning / add-a-route guarantees carry over unchanged.

Per dataset (Banking77 intents; CLINC150 in-scope intents) and depth, on test accuracy:
  ncm          nearest class mean (Euclidean, z-scored features)
  lda_z        LDA on z-scored features (shrinkage chosen on validation), = geometry_sufficient's LDA
  lda_zl2      LDA on z-scored then L2-normalised features
  klda_2k      KLDA, D=2000: feature variant (z / zl2), bandwidth c (gamma = c / median squared distance) and
               shrinkage chosen on validation
  klda_5k      KLDA, D=5000 at the chosen configuration; klda_e5 = mean softmax of 5 seeds (validation-best depth only)
  ranpac       ReLU random projection (M=5000) + ridge on one-hot targets, lambda chosen on validation
  probe        multinomial logistic regression on z-scored features (L2 chosen on validation; the tarski
               probe proxy), 300 full-batch Adam steps
At the validation-best depth, also:
  - class-incremental protocol (KLDA's splits: CLINC 10 sessions x 15 classes, Banking77 7 x 11): average
    incremental accuracy for LDA/KLDA (exact statistics), a probe retrained on all data seen so far (joint,
    upper reference) and a SEQ*-style probe that freezes old rows and trains only the new rows on the new
    session's data (no replay; Zheng et al., ACL 2024);
  - a new route from k examples (k = 1, 5, 10, 20) added to LDA vs KLDA statistics (10 held-out classes);
  - CLINC only: out-of-scope AUROC of the minimum Mahalanobis distance in LDA-z, LDA-zl2 and KLDA space.
Expected (from the literature scan): KLDA +1.5-3 points over LDA, within ~1 point of the probe or above it.
Kill: under +1 point over LDA at every depth.

Usage:
  .venv/bin/python explore/lit_oos_klda.py --smoke
  .venv/bin/python explore/lit_oos_klda.py --out results/tarski/explore_lit_klda.json   # ~15 min on an A10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_oos_common import (RFF, Gauss, Logger, fit_logreg, get_trunk, l2n, load_ds, median_sqdist, ood_metrics,
                            pick_device, pooled, seed_all, split_index, zscore)

import numpy as np
import torch
import torch.nn.functional as F

SHRINKS = (1e-3, 1e-2, 0.1, 0.3)
CS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def acc(z, y) -> float:
    return float((torch.as_tensor(z).argmax(-1).numpy() == y).mean())


def best_lda(X, y, tr, va, C, shrinks=SHRINKS):
    best = None
    g0 = Gauss(X[tr], y[tr], C, shrinks[0])
    for s in shrinks:
        g = g0.reshrink(s)
        a = acc(g.lda_logits(X[va]), y[va])
        if best is None or a > best[0]:
            best = (a, s, g)
    return best                                   # (val acc, shrink, Gauss)


def ranpac(X, y, tr, va, te, C, M, seed, dev):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(X.shape[1], M, generator=g)
    H = lambda rows: torch.relu(X[rows].float().to(dev) @ W.to(dev)).double().cpu()
    Htr = H(tr)
    Y = F.one_hot(torch.as_tensor(y[tr]), C).double()
    G = Htr.T @ Htr
    evals, V = torch.linalg.eigh(G)
    Cm = V.T @ (Htr.T @ Y)
    Hva, Hte = H(va), H(te)
    best = None
    for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4):
        Wl = V @ (Cm / (evals.clamp_min(0) + lam)[:, None])
        a = acc(Hva @ Wl, y[va])
        if best is None or a > best[0]:
            best = (a, lam, Wl)
    return {"val_acc": best[0], "lambda": best[1], "acc": acc(Hte @ best[2], y[te])}


def klda_search(variants, y, tr, va, C, D, seed, dev):
    """Grid over feature variant x bandwidth c x shrinkage at dimension D; returns the val-best config."""
    best = None
    for vname, X in variants.items():
        med = median_sqdist(X[tr])
        for c in CS:
            rff = RFF(X.shape[1], D, c / max(med, 1e-9), seed)
            Z = rff(X, dev)
            a, s, g = best_lda(Z, y, tr, va, C)
            if best is None or a > best["val_acc"]:
                best = {"val_acc": a, "variant": vname, "c": c, "shrink": s, "gamma": c / max(med, 1e-9)}
    return best


def klda_fit(X, y, tr, C, D, gamma, shrink, seed, dev):
    rff = RFF(X.shape[1], D, gamma, seed)
    Z = rff(X, dev)
    return Z, Gauss(Z[tr], y[tr], C, shrink)


def sessions_for(C: int, n_sessions: int, seed: int):
    order = np.random.default_rng(seed).permutation(C)
    return [np.sort(part) for part in np.array_split(order, n_sessions)]


def incremental(name_to_X, y, tr, te, C, sess, cfg, dev, steps, seed, l2_probe):
    """Average incremental accuracy (mean over sessions of test accuracy on classes seen so far)."""
    out = {}
    seen = np.array([], dtype=int)
    rows = {k: [] for k in ("lda", "klda", "probe_joint", "probe_newrows")}
    Xz = name_to_X["z"]
    Zk = name_to_X["klda"]
    W_old = torch.zeros(Xz.shape[1], C)
    b_old = torch.zeros(C)
    for s, cls in enumerate(sess):
        seen = np.concatenate([seen, cls])
        mask_tr = np.isin(y[tr], seen)
        mask_te = np.isin(y[te], seen)
        trs, tes = tr[mask_tr], te[mask_te]
        act = np.zeros(C, dtype=bool)
        act[seen] = True
        # LDA / KLDA: statistics of every class seen so far (exactly the joint fit on those classes)
        g = Gauss(name_to_X["lda_X"][trs], y[trs], C, cfg["lda_shrink"])
        rows["lda"].append(acc(g.lda_logits(name_to_X["lda_X"][tes]), y[tes]))
        gk = Gauss(Zk[trs], y[trs], C, cfg["klda_shrink"])
        rows["klda"].append(acc(gk.lda_logits(Zk[tes]), y[tes]))
        # probe retrained on all data so far (replay of everything: the joint reference)
        r = fit_logreg(Xz[trs], y[trs], C, [Xz[tes]], dev, l2s=(l2_probe,), steps=steps, seed=seed)
        z = r["logits"][0][0].clone()
        z[:, ~act] = -1e9
        rows["probe_joint"].append(acc(z, y[tes]))
        # SEQ*-style: old rows frozen, new rows trained on the new session's data only
        new_rows = torch.as_tensor(cls)
        trn = tr[np.isin(y[tr], cls)]
        Wn = torch.zeros(Xz.shape[1], len(cls), requires_grad=True)
        bn = torch.zeros(len(cls), requires_grad=True)
        opt = torch.optim.Adam([Wn, bn], lr=1e-2)
        old = torch.as_tensor(np.setdiff1d(seen, cls))
        Xn = Xz[trn].float()
        pos = {int(c): i for i, c in enumerate(cls)}
        yn = torch.as_tensor([pos[int(v)] for v in y[trn]])
        for _ in range(max(300, steps)):
            zn = Xn @ Wn + bn
            zo = (Xn @ W_old[:, old] + b_old[old]) if len(old) else torch.zeros(len(Xn), 0)
            logits = torch.cat([zo, zn], 1)
            loss = F.cross_entropy(logits, yn + len(old)) + l2_probe * (Wn ** 2).sum() / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            W_old[:, new_rows] = Wn.detach()
            b_old[new_rows] = bn.detach()
            z = Xz[tes].float() @ W_old + b_old
            z[:, ~act] = -1e9
        rows["probe_newrows"].append(acc(z, y[tes]))
    for k, v in rows.items():
        out[k] = {"per_session": v, "avg_incremental": float(np.mean(v)), "final": v[-1]}
    return out


class Stats:
    """Additive LDA statistics: per-class counts and sums and one total scatter (geometry idea 10)."""

    def __init__(self, X: torch.Tensor, y: np.ndarray, C: int):
        X = X.double()
        yt = torch.as_tensor(y, dtype=torch.long)
        self.n = torch.bincount(yt, minlength=C).double()
        self.s = torch.zeros(C, X.shape[1], dtype=torch.float64).index_add_(0, yt, X)
        self.S = X.T @ X

    def minus_class(self, c: int, Xc: torch.Tensor) -> "Stats":
        out = Stats.__new__(Stats)
        Xc = Xc.double()
        out.n, out.s, out.S = self.n.clone(), self.s.clone(), self.S - Xc.T @ Xc
        out.n[c], out.s[c] = 0, 0
        return out

    def plus(self, c: int, Xk: torch.Tensor) -> "Stats":
        out = Stats.__new__(Stats)
        Xk = Xk.double()
        out.n, out.s, out.S = self.n.clone(), self.s.clone(), self.S + Xk.T @ Xk
        out.n[c] += len(Xk)
        out.s[c] += Xk.sum(0)
        return out

    def logits(self, X: torch.Tensor, shrink: float) -> torch.Tensor:
        act = self.n > 0
        mu = self.s[act] / self.n[act, None]
        Sw = self.S - (self.s[act].T / self.n[act]) @ self.s[act]
        cov = Sw / max(1.0, float(self.n.sum()) - int(act.sum()))
        D = cov.shape[0]
        cov = (1 - shrink) * cov + shrink * cov.diagonal().mean() * torch.eye(D, dtype=torch.float64)
        PmuT = torch.linalg.solve(cov, mu.T)
        z = torch.full((len(X), len(self.n)), -1e30, dtype=torch.float64)
        z[:, act] = X.double() @ PmuT - 0.5 * (mu * PmuT.T).sum(-1)[None]
        return z


def new_route(Xl, Zk, y, tr, te, C, cfg, ks, n_classes, rng):
    """Remove one class's statistics, add back k shots: a new route with no retraining, both spaces."""
    ytr = y[tr]
    classes = [c for c in range(C) if (ytr == c).sum() >= max(ks) and (y[te] == c).sum() > 0]
    chosen = rng.choice(classes, min(n_classes, len(classes)), replace=False)
    res = {k: {"lda_new_recall": [], "lda_overall": [], "klda_new_recall": [], "klda_overall": []} for k in ks}
    full = {"lda": (Xl, Stats(Xl[tr], ytr, C), cfg["lda_shrink"]),
            "klda": (Zk, Stats(Zk[tr], ytr, C), cfg["klda_shrink"])}
    for c in chosen:
        idx_c = tr[ytr == c]
        for k in ks:
            shots = rng.choice(idx_c, k, replace=False)
            for name, (X, st, s) in full.items():
                st2 = st.minus_class(int(c), X[idx_c]).plus(int(c), X[shots])
                pred = st2.logits(X[te], s).argmax(-1).numpy()
                res[k][f"{name}_new_recall"].append(float((pred[y[te] == c] == c).mean()))
                res[k][f"{name}_overall"].append(float((pred == y[te]).mean()))
    return {int(k): {m: float(np.mean(v)) for m, v in d.items()} for k, d in res.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150"])
    ap.add_argument("--depths", type=int, nargs="*", default=[4, 8, 11, 16, 22])
    ap.add_argument("--dim-search", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=5000)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--ranpac-dim", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 5, 10, 20])
    ap.add_argument("--new-classes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.depths, args.dim_search, args.dim, args.ensemble, args.ranpac_dim = [4, 22], 256, 512, 2, 512
        args.ks, args.new_classes = [1, 5], 3
    out = args.out or ("results/tarski/explore_lit_klda_smoke.json" if args.smoke
                       else "results/tarski/explore_lit_klda.json")
    log = Logger(out)
    seed_all(args.seed)
    dev = pick_device(args.smoke)
    trunk = get_trunk(args.smoke)
    rng = np.random.default_rng(args.seed)
    res = {"args": vars(args), "datasets": {}}
    for name in args.datasets:
        t0 = time.time()
        ds = load_ds(name, args.smoke, args.seed)
        allx, idx = split_index(ds)
        labels = ds.tasks["intent"].labels
        if name == "clinc150":
            oos_i = labels.index("oos")
            in_ids = [i for i in range(len(labels)) if i != oos_i]
            imap = {i: j for j, i in enumerate(in_ids)}
            y = np.array([imap.get(e.y["intent"], -1) for e in allx])
            is_oos = np.array([e.y["oos"] for e in allx])
            C = 150
        else:
            y = np.array([e.y["intent"] for e in allx])
            is_oos = np.zeros(len(allx), dtype=int)
            C = len(labels)
        tr = idx["train"][is_oos[idx["train"]] == 0]
        va = idx["val"][is_oos[idx["val"]] == 0]
        te = idx["test"][is_oos[idx["test"]] == 0]
        te_all = idx["test"]
        feats = pooled(trunk, [e.text for e in allx], args.depths, ds.max_len, log=log)
        log(f"== KLDA on {name}: {C} classes, {len(tr)}/{len(va)}/{len(te)} in-scope train/val/test | depths "
            f"{args.depths} | fits {dev}")
        per_depth = {}
        for d in args.depths:
            t1 = time.time()
            Xz = zscore(feats[d], tr)
            Xzl2 = l2n(Xz)
            r = {}
            mu = torch.stack([Xz[tr][y[tr] == c].mean(0) for c in range(C)])
            r["ncm"] = acc(-torch.cdist(Xz[te].float(), mu.float()), y[te])
            a, s, g = best_lda(Xz, y, tr, va, C)
            r["lda_z"], r["lda_z_shrink"] = acc(g.lda_logits(Xz[te]), y[te]), s
            a2, s2, g2 = best_lda(Xzl2, y, tr, va, C)
            r["lda_zl2"], r["lda_zl2_shrink"] = acc(g2.lda_logits(Xzl2[te]), y[te]), s2
            r["lda_val"] = {"z": a, "zl2": a2}
            variants = {"z": Xz, "zl2": Xzl2}
            cfg = klda_search(variants, y, tr, va, C, args.dim_search, args.seed, dev)
            Xk = variants[cfg["variant"]]
            Z, gk = klda_fit(Xk, y, tr, C, args.dim_search, cfg["gamma"], cfg["shrink"], args.seed, dev)
            r["klda_2k"] = acc(gk.lda_logits(Z[te]), y[te])
            Z5, gk5 = klda_fit(Xk, y, tr, C, args.dim, cfg["gamma"], cfg["shrink"], args.seed, dev)
            r["klda_5k"] = acc(gk5.lda_logits(Z5[te]), y[te])
            r["klda_5k_val"] = acc(gk5.lda_logits(Z5[va]), y[va])
            r["klda_cfg"] = cfg
            r["ranpac"] = ranpac(Xz, y, tr, va, te, C, args.ranpac_dim, args.seed, dev)
            pr = fit_logreg(Xz[tr], y[tr], C, [Xz[va], Xz[te]], dev, steps=args.steps, seed=args.seed)
            k = int(np.argmax((pr["logits"][0].argmax(-1).numpy() == y[va][None]).mean(-1)))
            r["probe"], r["probe_l2"] = acc(pr["logits"][1][k], y[te]), pr["l2s"][k]
            r["probe_val"] = acc(pr["logits"][0][k], y[va])
            if name == "clinc150":
                oo = is_oos[te_all]
                r["ood_auroc"] = {
                    "maha_lda_z": ood_metrics(g.min_dist(Xz[te_all]), oo)["auroc"],
                    "maha_lda_zl2": ood_metrics(g2.min_dist(Xzl2[te_all]), oo)["auroc"],
                    "maha_klda_5k": ood_metrics(gk5.min_dist(Z5[te_all]), oo)["auroc"]}
            per_depth[d] = r
            log(f"-- {name} @ {d} ({time.time() - t1:.0f}s): ncm {r['ncm']:.4f} | lda_z {r['lda_z']:.4f} | "
                f"lda_zl2 {r['lda_zl2']:.4f} | klda_2k {r['klda_2k']:.4f} | klda_5k {r['klda_5k']:.4f} "
                f"({cfg['variant']}, c={cfg['c']}, shrink={cfg['shrink']}) | ranpac {r['ranpac']['acc']:.4f} | "
                f"probe {r['probe']:.4f}" + (f" | OOS AUROC {r['ood_auroc']}" if "ood_auroc" in r else ""))
            res["datasets"].setdefault(name, {})["depths"] = per_depth
            log.dump(res)
        # validation-best depth for KLDA: ensemble, incremental protocol, new routes
        bd = max(args.depths, key=lambda d: per_depth[d]["klda_5k_val"])
        cfg = per_depth[bd]["klda_cfg"]
        Xz = zscore(feats[bd], tr)
        Xk = {"z": Xz, "zl2": l2n(Xz)}[cfg["variant"]]
        P = None
        for s in range(args.ensemble):
            Z, gk = klda_fit(Xk, y, tr, C, args.dim, cfg["gamma"], cfg["shrink"], args.seed + s, dev)
            p = torch.softmax(gk.lda_logits(Z[te]).float(), -1)
            P = p if P is None else P + p
        ens = acc(P, y[te])
        Z5, _ = klda_fit(Xk, y, tr, C, args.dim, cfg["gamma"], cfg["shrink"], args.seed, dev)
        n_sess = 10 if name == "clinc150" else 7
        if args.smoke:
            n_sess = 3
        sess = sessions_for(C, n_sess, args.seed)
        lda_shrink = per_depth[bd]["lda_z_shrink"]
        inc = incremental({"z": Xz, "klda": Z5, "lda_X": Xz}, y, tr, te, C, sess,
                          {"lda_shrink": lda_shrink, "klda_shrink": cfg["shrink"]}, dev, args.steps, args.seed,
                          per_depth[bd]["probe_l2"])
        nr = new_route(Xz, Z5, y, tr, te, C, {"lda_shrink": lda_shrink, "klda_shrink": cfg["shrink"]},
                       args.ks, args.new_classes, rng)
        res["datasets"][name].update({"best_depth": int(bd), "klda_e_acc": ens, "incremental": inc,
                                      "new_route": nr, "wall_s": round(time.time() - t0, 1)})
        log(f"== {name}: val-best depth {bd}: klda_e{args.ensemble} {ens:.4f} (klda_5k {per_depth[bd]['klda_5k']:.4f}, "
            f"lda_z {per_depth[bd]['lda_z']:.4f}, probe {per_depth[bd]['probe']:.4f})")
        log("   incremental (avg over sessions / final): " + " | ".join(
            f"{k} {v['avg_incremental']:.4f}/{v['final']:.4f}" for k, v in inc.items()))
        log("   new route from k shots (new-class recall, overall): " + " | ".join(
            f"k={k}: lda {v['lda_new_recall']:.2f}/{v['lda_overall']:.3f}, klda {v['klda_new_recall']:.2f}/{v['klda_overall']:.3f}"
            for k, v in nr.items()))
        log.dump(res)
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
