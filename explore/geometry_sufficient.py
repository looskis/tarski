"""Idea 10: sufficient-statistics branches. Mergeable, exactly unlearnable routes with OOD for free.

A branch is stored as sufficient statistics of mean-pooled trunk states at depth k: per-class (soft)
counts n_c, per-class sums s_c and one total scatter S = sum_i x_i x_i^T. Everything else is derived in
closed form: the class means, the shared within-class covariance (shrunk towards a scaled identity by
gamma, chosen on validation) and a linear discriminant, i.e. Gaussian class-conditionals with a shared
covariance (LDA).

What that buys, and what this script measures:
  accuracy    LDA vs a tuned logistic-regression probe on the same states, and vs nearest class mean (NCM,
              identity covariance).
  merge       two users' branches (disjoint halves of the training set) are merged by adding statistics.
              Should equal the branch fitted on everything (max |logit diff|, prediction agreement).
  unlearn     deleting 10% of the training examples is subtracting their statistics. Should equal refitting
              on the remaining 90%.
  new label   a new route from k examples (k = 1, 5, 10) is adding one class's statistics, with no retraining,
              vs a logistic probe retrained from scratch on the same data. Reported: new-class recall and
              overall accuracy, mean over 10 held-out classes (Banking77, CLINC intents).
  OOD         the same statistics score out-of-scope messages (CLINC150): minimum Mahalanobis distance to
              the in-scope classes (AUROC), without any oos training data.

Features are standardized with one fixed scaler (a trunk-level constant shared by every user), so
statistics from different users live in the same space and merge exactly.

Prediction (geometry.md idea 10): LDA within ~2 points of the logistic probe on intent tasks; merge and
unlearning exact to float precision; with 5 examples a new route reaches >= 50% recall with no retraining.

Usage:
  .venv/bin/python explore/geometry_sufficient.py --smoke
  .venv/bin/python explore/geometry_sufficient.py --out results/tarski/explore_sufficient.json   # ~5 min on an A10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_common import (Logger, acc, device, fit_logreg, labels, load_dataset, pick_by_val, pooled, splits,
                             targets, task_split)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from tarski.trunk import Trunk


class GaussStats:
    """Additive sufficient statistics of a shared-covariance Gaussian classifier."""

    def __init__(self, C: int, D: int):
        self.n = torch.zeros(C, dtype=torch.float64)
        self.s = torch.zeros(C, D, dtype=torch.float64)
        self.S = torch.zeros(D, D, dtype=torch.float64)

    def add(self, X: torch.Tensor, P: torch.Tensor, sign: float = 1.0) -> "GaussStats":
        X, P = X.double(), P.double()
        w = P.sum(1, keepdim=True)
        self.n += sign * P.sum(0)
        self.s += sign * P.T @ X
        self.S += sign * (X * w).T @ X
        return self

    def merge(self, other: "GaussStats") -> "GaussStats":
        out = GaussStats(len(self.n), self.s.shape[1])
        out.n, out.s, out.S = self.n + other.n, self.s + other.s, self.S + other.S
        return out

    def copy(self) -> "GaussStats":
        out = GaussStats(len(self.n), self.s.shape[1])
        out.n, out.s, out.S = self.n.clone(), self.s.clone(), self.S.clone()
        return out

    def classifier(self, gamma: float, identity: bool = False, active=None):
        """Returns (W, b, mu, precision) for logits = X W + b over classes with n_c > 0 (others -inf)."""
        C, D = self.s.shape
        act = (self.n > 1e-9) if active is None else active
        mu = self.s[act] / self.n[act, None]
        if identity:
            P = torch.eye(D, dtype=torch.float64)
        else:
            Sw = self.S - (self.s[act].T / self.n[act]) @ self.s[act]
            cov = Sw / max(1.0, float(self.n.sum()) - int(act.sum()))
            cov = (1 - gamma) * cov + gamma * cov.diagonal().mean() * torch.eye(D, dtype=torch.float64)
            P = torch.linalg.inv(cov)
        Wa = P @ mu.T
        ba = -0.5 * ((mu @ P) * mu).sum(-1)
        W = torch.zeros(D, C, dtype=torch.float64)
        b = torch.full((C,), -1e30, dtype=torch.float64)
        W[:, act], b[act] = Wa, ba
        return W, b, mu, P


def logits(X, clf):
    W, b = clf[0], clf[1]
    return X.double() @ W + b


GAMMAS = (0.01, 0.03, 0.1, 0.3, 0.6)


def best_gamma(st, Xva, yva):
    scores = {g: acc(logits(Xva, st.classifier(g)), yva) for g in GAMMAS}
    return max(scores, key=scores.get)


def onehot(y, C):
    return torch.nn.functional.one_hot(torch.as_tensor(y), C).double()


def run_task(X, allx, ds, t, n_tr, n_va, dev, args, rng, oos_label=None):
    tr, va, te = task_split(allx, t, n_tr, n_va)
    C = len(ds.tasks[t].labels)
    T = targets(allx, tr, t, C)
    yva, yte = labels(allx, va, t), labels(allx, te, t)
    D = X.shape[1]
    out = {}
    full = GaussStats(C, D).add(X[tr], T)
    g = best_gamma(full, X[va], yva)
    clf = full.classifier(g)
    out["lda"] = {"acc": acc(logits(X[te], clf), yte), "gamma": g}
    out["ncm"] = {"acc": acc(logits(X[te], full.classifier(0.0, identity=True)), yte)}
    r = fit_logreg(X[tr], T, [X[va], X[te]], dev, steps=args.steps)
    k = pick_by_val(r["logits"][0], yva)
    out["logreg"] = {"acc": acc(r["logits"][1][k], yte), "l2": (1e-4, 1e-3, 1e-2)[k]}
    # merge: two users with disjoint halves
    perm = rng.permutation(len(tr))
    half = len(tr) // 2
    A = GaussStats(C, D).add(X[[tr[i] for i in perm[:half]]], T[perm[:half]])
    B = GaussStats(C, D).add(X[[tr[i] for i in perm[half:]]], T[perm[half:]])
    zm, zf = logits(X[te], A.merge(B).classifier(g)), logits(X[te], clf)
    fin = torch.isfinite(zf) & (zf > -1e29)
    out["merge"] = {"max_abs_logit_diff": float((zm - zf)[fin].abs().max()), "max_abs_logit": float(zf[fin].abs().max()),
                    "prediction_agreement": float((zm.argmax(-1) == zf.argmax(-1)).double().mean())}
    # unlearn 10%
    gone = perm[: max(1, len(tr) // 10)]
    keep = perm[max(1, len(tr) // 10):]
    zu = logits(X[te], full.copy().add(X[[tr[i] for i in gone]], T[gone], sign=-1.0).classifier(g))
    zr = logits(X[te], GaussStats(C, D).add(X[[tr[i] for i in keep]], T[keep]).classifier(g))
    fin = zr > -1e29
    out["unlearn"] = {"max_abs_logit_diff": float((zu - zr)[fin].abs().max()),
                      "prediction_agreement": float((zu.argmax(-1) == zr.argmax(-1)).double().mean())}
    # OOD: min Mahalanobis distance to in-scope classes
    if oos_label is not None:
        ins = torch.ones(C, dtype=torch.bool)
        ins[oos_label] = False
        st_in = GaussStats(C, D).add(X[tr], T * ins.float())
        _, _, mu, P = st_in.classifier(g, active=ins & (st_in.n > 0))
        Xt = X[te].double()
        d = ((Xt @ P) * Xt).sum(-1, keepdim=True) - 2 * (Xt @ P) @ mu.T + ((mu @ P) * mu).sum(-1)[None]
        score = d.min(-1).values.numpy()
        out["ood_auroc_min_mahalanobis"] = float(roc_auc_score((yte == oos_label).astype(int), score))
    # new label from k examples (single-label tasks with many classes)
    if args.new_label and C >= 20:
        classes = [c for c in range(C) if c != oos_label and (labels(allx, tr, t) == c).sum() >= max(args.ks)
                   and (yte == c).sum() > 0]
        chosen = rng.choice(classes, min(args.new_label, len(classes)), replace=False)
        ytr = labels(allx, tr, t)
        res = {k: {"lda_new_recall": [], "lda_overall": [], "logreg_new_recall": [], "logreg_overall": []} for k in args.ks}
        for c in chosen:
            idx_c = np.where(ytr == c)[0]
            idx_o = np.where(ytr != c)[0]
            base = GaussStats(C, D).add(X[[tr[i] for i in idx_o]], T[idx_o])
            for k in args.ks:
                shots = rng.choice(idx_c, k, replace=False)
                st = base.copy().add(X[[tr[i] for i in shots]], T[shots])
                z = logits(X[te], st.classifier(g))
                pred = z.argmax(-1).numpy()
                res[k]["lda_new_recall"].append(float((pred[yte == c] == c).mean()))
                res[k]["lda_overall"].append(float((pred == yte).mean()))
                rows = np.r_[idx_o, shots]
                rr = fit_logreg(X[[tr[i] for i in rows]], T[rows], [X[te]], dev, l2s=[out["logreg"]["l2"]], steps=args.steps)
                pz = rr["logits"][0][0].argmax(-1).numpy()
                res[k]["logreg_new_recall"].append(float((pz[yte == c] == c).mean()))
                res[k]["logreg_overall"].append(float((pz == yte).mean()))
        out["new_label"] = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in res.items()}
        out["new_label"]["classes"] = [int(c) for c in chosen]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150", "typed-decisions"])
    ap.add_argument("--depths", type=int, nargs="*", default=[11, 22])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 5, 10])
    ap.add_argument("--new-label", type=int, default=10, help="held-out classes for the new-label test")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.ks, args.new_label, args.depths = 40, [1, 2], 2, [11]
    out_path = args.out or ("results/tarski/explore_sufficient_smoke.json" if args.smoke
                            else "results/tarski/explore_sufficient.json")
    log = Logger(out_path)
    dev = device(args.smoke)
    trunk = Trunk(device="cpu" if args.smoke else None)
    log(f"== sufficient-statistics branches | trunk {trunk.device}, fits {dev} | depths {args.depths}")
    res = {"args": vars(args), "datasets": {}}
    rng = np.random.default_rng(0)
    for name in args.datasets:
        t0 = time.time()
        ds = load_dataset(name, args.smoke)
        allx, n_tr, n_va = splits(ds)
        feats = pooled(trunk, ds, args.depths)
        res["datasets"][name] = {}
        for d in args.depths:
            F_ = feats[d]
            mu, sd = F_[:n_tr].mean(0, keepdim=True), F_[:n_tr].std(0, keepdim=True).clamp_min(1e-4)
            X = ((F_ - mu) / sd).double()                  # one fixed scaler for every user's statistics
            per = {}
            for t in ds.tasks:
                labs = ds.tasks[t].labels
                oos = labs.index("oos") if (name == "clinc150" and t == "intent" and "oos" in labs) else None
                per[t] = run_task(X, allx, ds, t, n_tr, n_va, dev, args, rng, oos_label=oos)
            summ = {m: float(np.mean([v[m]["acc"] for v in per.values()])) for m in ("lda", "ncm", "logreg")}
            summ["merge_max_rel_diff"] = float(max(v["merge"]["max_abs_logit_diff"] / max(v["merge"]["max_abs_logit"], 1e-12)
                                                   for v in per.values()))
            summ["merge_agreement_min"] = float(min(v["merge"]["prediction_agreement"] for v in per.values()))
            summ["unlearn_max_abs_diff"] = float(max(v["unlearn"]["max_abs_logit_diff"] for v in per.values()))
            res["datasets"][name][d] = {"tasks": per, "summary": summ}
            log(f"-- {name} @ depth {d}: LDA {summ['lda']:.4f} | NCM {summ['ncm']:.4f} | logreg {summ['logreg']:.4f} | "
                f"merge rel diff {summ['merge_max_rel_diff']:.1e} (agree {summ['merge_agreement_min']:.3f}) | "
                f"unlearn abs diff {summ['unlearn_max_abs_diff']:.1e}")
            for t, v in per.items():
                if "new_label" in v:
                    log(f"   {t} new label: " + " | ".join(
                        f"k={k}: LDA recall {v['new_label'][k]['lda_new_recall']:.3f} overall {v['new_label'][k]['lda_overall']:.3f}, "
                        f"retrained logreg recall {v['new_label'][k]['logreg_new_recall']:.3f} overall {v['new_label'][k]['logreg_overall']:.3f}"
                        for k in args.ks))
                if "ood_auroc_min_mahalanobis" in v:
                    log(f"   {t} OOD AUROC (min Mahalanobis, no oos data): {v['ood_auroc_min_mahalanobis']:.4f}")
        res["datasets"][name]["wall_s"] = round(time.time() - t0, 1)
        json.dump(res, open(out_path, "w"), indent=1, default=float)
    log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
