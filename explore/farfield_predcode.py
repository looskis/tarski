"""Far-field idea 4: prediction error across depth (predictive coding) on CLINC150.

In predictive coding a higher area predicts the lower area's activity and only the residual (surprise)
propagates. Here a ridge map W predicts the trunk's pooled state at depth k+j from the state at depth k,
fitted on in-scope training messages. Two tests:

  * OOS score = whitened norm of the prediction error at test time (in-scope messages should be
    predictable; out-of-scope messages should surprise the map). Compared with max-softmax, entropy and
    1-NN distance on the same depth, alone and in z-score combination.
  * "innovation probe": an intent probe that reads only the residual e = z_{k+j} - W z_k (the branch
    processes only what the trunk failed to predict) vs the usual probe on z_{k+j} and a probe on
    [z_k, z_{k+j}]. Prediction: the residual probe loses several points, because the predictable part of
    the deeper state carries most of the decision; the error score is a weak OOS signal on pooled states
    (AUROC below 1-NN), since pooled residual-stream states are close to linearly predictable across depth.

Usage:
  .venv/bin/python explore/farfield_predcode.py --smoke
  .venv/bin/python explore/farfield_predcode.py --out results/tarski/explore_predcode.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import numpy as np
import torch

from explore.farfield_common import (Logger, Standardiser, dump, entropy, fit_linear, knn_distance, logits_of, ood_metrics,
                                     out_paths, softmax_np, split_indices, subsample)
from tarski import data
from tarski.autosplit import pooled_by_depth
from tarski.trunk import Trunk


def ridge(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """W minimising ||xW - y||^2 + lam ||W||^2 (x, y centred by the caller)."""
    d = x.shape[1]
    a = x.T @ x + lam * torch.eye(d, dtype=x.dtype)
    return torch.linalg.solve(a, x.T @ y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--pairs", nargs="*", default=["4-8", "8-12", "12-16", "16-22", "4-12", "4-22"], help="source-target depths")
    ap.add_argument("--ridge", type=float, default=1e-1, help="ridge penalty relative to n_train")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.device = args.device or "cpu"
        args.pairs = ["4-8", "4-12"] if args.pairs == ["4-8", "8-12", "12-16", "16-22", "4-12", "4-22"] else args.pairs
    out, logp = out_paths(args, "predcode")
    L = Logger(logp)
    t_start = time.time()
    pairs = [tuple(int(v) for v in p.split("-")) for p in args.pairs]
    depths = sorted({d for p in pairs for d in p})

    trunk = Trunk(device=args.device)
    dev = trunk.device
    ds = data.load_clinc()
    train, val, test = ds.train, ds.val, ds.test
    if args.smoke:
        train = subsample(train, 1500, args.seed, key=lambda e: e.y["intent"])
        val = subsample(val, 300, args.seed + 1, key=lambda e: e.y["intent"])
        test = subsample([e for e in test if e.y["oos"] == 0], 600, args.seed + 2, key=lambda e: e.y["intent"]) + \
            subsample([e for e in test if e.y["oos"] == 1], 150, args.seed + 3)
    allx = train + val + test
    idx = split_indices(len(train), len(val), len(allx))
    intents = ds.tasks["intent"].labels
    oos_i = intents.index("oos")
    y_int = np.array([e.y["intent"] for e in allx])
    y_oos = np.array([e.y["oos"] for e in allx])
    in_ids = [i for i in range(len(intents)) if i != oos_i]
    imap = {i: j for j, i in enumerate(in_ids)}
    y150 = np.array([imap.get(i, -1) for i in y_int])
    tr_in = idx["train"][y_oos[idx["train"]] == 0]
    va_in = idx["val"][y_oos[idx["val"]] == 0]
    te = idx["test"]
    te_in = te[y_oos[te] == 0]
    L(f"== predictive coding on clinc150: {len(train)}/{len(val)}/{len(test)}, pairs {pairs}, device {dev}, smoke={args.smoke}")
    t0 = time.time()
    feats = pooled_by_depth(trunk, [e.text for e in allx], depths, ds.max_len)
    L(f"  trunk pass: {time.time() - t0:.1f}s")
    z = {d: Standardiser(feats[d][tr_in])(feats[d]).double() for d in depths}

    results = {"config": vars(args), "device": str(dev), "n_test": int(len(te)), "n_test_oos": int(y_oos[te].sum()), "pairs": {}}

    def probe_acc(x: torch.Tensor):
        """In-scope intent accuracy on test, plus test and in-scope-validation probabilities."""
        lin = fit_linear(x[tr_in].float(), torch.tensor(y150[tr_in]), len(in_ids), dev, args.steps, seed=args.seed)
        p = softmax_np(logits_of(lin, x[te].float(), dev))
        p_val = softmax_np(logits_of(lin, x[va_in].float(), dev))
        return float((p[y_oos[te] == 0].argmax(-1) == y150[te_in]).mean()), p, p_val

    def zsum(*pairs):
        """Sum of scores z-scored against their in-scope validation distribution (label-free combination)."""
        return sum((s - v.mean()) / (v.std() + 1e-9) for s, v in pairs)

    for k, kj in pairs:
        t0 = time.time()
        xs, xt = z[k], z[kj]
        mu_s, mu_t = xs[tr_in].mean(0, keepdim=True), xt[tr_in].mean(0, keepdim=True)
        W = ridge(xs[tr_in] - mu_s, xt[tr_in] - mu_t, args.ridge * len(tr_in))
        pred = (xs - mu_s) @ W + mu_t
        err = xt - pred
        var = err[tr_in].var(0, keepdim=True).clamp_min(1e-6)
        r2 = float(1 - (err[tr_in] ** 2).sum() / ((xt[tr_in] - mu_t) ** 2).sum())
        score_err = (err ** 2 / var).sum(-1).numpy()                          # whitened prediction error
        score_err_raw = (err ** 2).sum(-1).numpy()
        # probes: usual (on z_{k+j}), innovation-only (on e), both, and the source depth alone
        acc_t, p_t, p_t_val = probe_acc(xt)
        acc_e, _, _ = probe_acc(err / var.sqrt())
        acc_both, _, _ = probe_acc(torch.cat([xs, xt], 1))
        acc_s, _, _ = probe_acc(xs)
        msp, msp_val = 1 - p_t.max(-1), 1 - p_t_val.max(-1)
        knn1 = knn_distance(xt[tr_in].float(), xt[te].float(), 1, dev)
        knn1_val = knn_distance(xt[tr_in].float(), xt[va_in].float(), 1, dev)
        scores = {"pred_error_whitened": score_err[te], "pred_error_raw": score_err_raw[te], "msp_int_target": msp,
                  "ent_int_target": entropy(p_t), "knn1_target": knn1,
                  "sum_pred_error+msp": zsum((score_err[te], score_err[va_in]), (msp, msp_val)),
                  "sum_pred_error+knn1": zsum((score_err[te], score_err[va_in]), (knn1, knn1_val))}
        m = {n: ood_metrics(s, y_oos[te]) for n, s in scores.items()}
        results["pairs"][f"{k}-{kj}"] = {"ridge_r2_train": r2, "ood": m,
                                          "probe_acc_inscope": {"source_z_k": acc_s, "target_z_kj": acc_t, "residual_only": acc_e,
                                                                "concat": acc_both},
                                          "seconds": round(time.time() - t0, 1)}
        L(f"-- {k} -> {kj}: ridge R^2 on train {r2:.3f}; in-scope intent acc: z_k {acc_s:.4f}, z_kj {acc_t:.4f}, "
          f"residual only {acc_e:.4f}, concat {acc_both:.4f}")
        for n in scores:
            L(f"   OOS {n:>22}: AUROC {m[n]['auroc']:.4f}  AUPR {m[n]['aupr']:.4f}  FPR@95 {m[n]['fpr95']:.3f}")
    results["wall_seconds"] = round(time.time() - t_start, 1)
    dump(results, out)
    L(f"== done in {results['wall_seconds']}s; wrote {out}")


if __name__ == "__main__":
    main()
