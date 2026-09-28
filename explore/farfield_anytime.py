"""Far-field ideas 5 and 7: anytime decision bundles (real-time systems) and inverse-variance fusion over
depth (Kalman filtering), on CLINC150 (three decisions per message) and Banking77 (one).

Probes at every depth 1..22 give each decision a performance profile (accuracy vs trunk depth) and, per
message, a trajectory of decisions over depth. Two mechanisms are tested on top of that:

  Idea 7, fusion: at cut-off depth d, fuse the probes at depths <= d with weights inversely proportional to
    their validation NLL (log-linear pooling), the way a Kalman filter fuses noisy measurements. Prediction:
    the fused anytime curve dominates the single-depth curve at every d, and fused@8 matches single@22.
  Idea 5, anytime bundle stopping: a per-task flip model P(argmax_d != argmax_22 | margin_d, entropy_d, d)
    is fitted on validation. For a bundle of requested decisions the trunk stops at the first depth where
    every requested decision's flip probability is below tau. Reported: bundle accuracy vs mean stopping
    depth for a grid of tau, against fixed-depth probes, a max-softmax threshold exit (DeeBERT-style) and a
    patience exit (PABEE-style). Prediction: on Banking77 the profile is flat, so the bundle stops around
    depth 4-6 at no accuracy cost; on CLINC the bundle waits for intent, the slowest decision.

Usage:
  .venv/bin/python explore/farfield_anytime.py --smoke
  .venv/bin/python explore/farfield_anytime.py --out results/tarski/explore_anytime.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from explore.farfield_common import (EPS, Logger, Standardiser, dump, entropy, fit_linear, logits_of, out_paths, smoke_subset,
                                     split_indices)
from tarski import data
from tarski.autosplit import pooled_by_depth
from tarski.train import fit_temperature
from tarski.trunk import Trunk


def log_softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def run_dataset(name: str, trunk: Trunk, args, L) -> Dict:
    ds = data.load(name)
    if args.smoke:
        key = (lambda e: e.y["intent"]) if "intent" in ds.tasks else None
        ds = smoke_subset(ds, 1500, 300, 600, args.seed, key)
    allx = ds.train + ds.val + ds.test
    idx = split_indices(len(ds.train), len(ds.val), len(allx))
    depths = list(range(1, trunk.n_layers + 1)) if not args.smoke else [2, 4, 6, 8, 12, 16, 22]
    dev = trunk.device
    L(f"== {ds.summary()} | depths {depths}")
    t0 = time.time()
    feats = pooled_by_depth(trunk, [e.text for e in allx], depths, ds.max_len)
    L(f"  trunk pass: {time.time() - t0:.1f}s")
    tasks = list(ds.tasks)
    out = {"tasks": {}, "depths": depths, "n_test": int(len(idx["test"]))}

    # --- probes at every depth, calibrated; keep val/test log-probs ------------------------------------
    logp: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {t: {"val": {}, "test": {}} for t in tasks}
    y: Dict[str, Dict[str, np.ndarray]] = {}
    for t in tasks:
        rows = {s: np.array([i for i in idx[s] if t in allx[i].y]) for s in idx}
        y[t] = {s: np.array([allx[i].y[t] for i in rows[s]]) for s in rows}
        n_cls = len(ds.tasks[t].labels)
        single = {}
        for d in depths:
            st = Standardiser(feats[d][rows["train"]])
            lin = fit_linear(st(feats[d][rows["train"]]), torch.tensor(y[t]["train"]), n_cls, dev, args.steps, seed=args.seed)
            zv, zt = logits_of(lin, st(feats[d][rows["val"]]), dev), logits_of(lin, st(feats[d][rows["test"]]), dev)
            temp = fit_temperature(zv, torch.tensor(y[t]["val"])) if len(rows["val"]) >= 30 else 1.0
            logp[t]["val"][d] = log_softmax_np((zv / temp).numpy().astype(np.float64))
            logp[t]["test"][d] = log_softmax_np((zt / temp).numpy().astype(np.float64))
            single[d] = float((logp[t]["test"][d].argmax(-1) == y[t]["test"]).mean())
        out["tasks"][t] = {"single_acc_by_depth": single}
        L(f"  [{t}] single-depth probe acc: " + " ".join(f"{d}:{a:.3f}" for d, a in single.items()))

    # --- idea 7: inverse-NLL fusion over depths <= d ----------------------------------------------------
    for t in tasks:
        nll_val = {d: float(-logp[t]["val"][d][np.arange(len(y[t]["val"])), y[t]["val"]].mean()) for d in depths}
        fused_inv, fused_mean, fused_best = {}, {}, {}
        for i, d in enumerate(depths):
            upto = depths[: i + 1]
            w = np.array([1.0 / nll_val[k] for k in upto])
            w = w / w.sum()
            z_inv = sum(wk * logp[t]["test"][k] for wk, k in zip(w, upto))
            z_mean = sum(logp[t]["test"][k] for k in upto) / len(upto)
            fused_inv[d] = float((z_inv.argmax(-1) == y[t]["test"]).mean())
            fused_mean[d] = float((z_mean.argmax(-1) == y[t]["test"]).mean())
            # "best single so far" = oracle choice of one depth <= d by validation NLL
            kb = min(upto, key=lambda k: nll_val[k])
            fused_best[d] = float((logp[t]["test"][kb].argmax(-1) == y[t]["test"]).mean())
        single = out["tasks"][t]["single_acc_by_depth"]
        dominates = sum(fused_inv[d] >= single[d] - 1e-9 for d in depths)
        out["tasks"][t].update({"fused_invnll_acc_by_depth": fused_inv, "fused_mean_acc_by_depth": fused_mean,
                                "best_single_leq_d_by_valnll": fused_best, "val_nll_by_depth": nll_val,
                                "fused_dominates_single_at_n_depths": int(dominates)})
        mid = depths[len(depths) // 3]
        L(f"  [{t}] fusion: fused(inv-NLL) >= single at {dominates}/{len(depths)} depths; "
          f"fused@{mid} {fused_inv[mid]:.4f} vs single@{mid} {single[mid]:.4f} vs single@{depths[-1]} {single[depths[-1]]:.4f}; "
          f"fused@{depths[-1]} {fused_inv[depths[-1]]:.4f}")

    # --- idea 5: anytime bundle stopping ---------------------------------------------------------------
    final = depths[-1]
    n_test = len(idx["test"])
    # all CLINC/banking rows carry every task, so bundle = all tasks on every message
    flip_models = {}
    for t in tasks:
        Xv, yv = [], []
        for d in depths[:-1]:
            lp = logp[t]["val"][d]
            p = np.exp(lp)
            srt = np.sort(p, -1)
            Xv.append(np.stack([srt[:, -1] - srt[:, -2], entropy(p), np.full(len(p), d / final)], 1))
            yv.append((lp.argmax(-1) != logp[t]["val"][final].argmax(-1)).astype(int))
        Xv, yv = np.concatenate(Xv), np.concatenate(yv)
        flip_models[t] = LogisticRegression(max_iter=2000).fit(Xv, yv) if yv.min() != yv.max() else None

    def flip_prob(t, d, split="test"):
        lp = logp[t][split][d]
        p = np.exp(lp)
        srt = np.sort(p, -1)
        X = np.stack([srt[:, -1] - srt[:, -2], entropy(p), np.full(len(p), d / final)], 1)
        return flip_models[t].predict_proba(X)[:, 1] if flip_models[t] is not None else np.zeros(len(p))

    def decide_at(t, stop_depth, use_fused: bool):
        """Per-message decision at each message's stopping depth (single probe or inverse-NLL fusion)."""
        pred = np.zeros(n_test, dtype=int)
        for d in np.unique(stop_depth):
            m = stop_depth == d
            if use_fused:
                upto = [k for k in depths if k <= d]
                w = np.array([1.0 / out["tasks"][t]["val_nll_by_depth"][k] for k in upto])
                w = w / w.sum()
                z = sum(wk * logp[t]["test"][k][m] for wk, k in zip(w, upto))
            else:
                z = logp[t]["test"][d][m]
            pred[m] = z.argmax(-1)
        return pred

    def bundle_metrics(stop_depth, use_fused):
        accs = {t: float((decide_at(t, stop_depth, use_fused) == y[t]["test"]).mean()) for t in tasks}
        return {"mean_depth": float(stop_depth.mean()), "acc": accs, "mean_acc": float(np.mean(list(accs.values())))}

    policies = {"voi_flip": {}, "msp_threshold": {}, "patience": {}}
    fp = {t: {d: flip_prob(t, d) for d in depths[:-1]} for t in tasks}
    for tau in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        stop = np.full(n_test, final)
        for d in reversed(depths[:-1]):                         # first depth where all tasks are settled
            ok = np.all([fp[t][d] < tau for t in tasks], 0)
            stop[ok] = d
        policies["voi_flip"][str(tau)] = {"single": bundle_metrics(stop, False), "fused": bundle_metrics(stop, True)}
    for thr in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
        stop = np.full(n_test, final)
        for d in reversed(depths[:-1]):
            ok = np.all([np.exp(logp[t]["test"][d]).max(-1) > thr for t in tasks], 0)
            stop[ok] = d
        policies["msp_threshold"][str(thr)] = {"single": bundle_metrics(stop, False)}
    for pat in (1, 2, 3, 4):
        stop = np.full(n_test, final)
        arg = {t: np.stack([logp[t]["test"][d].argmax(-1) for d in depths]) for t in tasks}   # (D, N)
        for i in range(pat, len(depths) - 1):
            ok = np.all([np.all(arg[t][i - pat: i + 1] == arg[t][i], 0) for t in tasks], 0)
            d = depths[i]
            stop = np.where((stop == final) & ok, d, stop)
        policies["patience"][str(pat)] = {"single": bundle_metrics(stop, False)}
    fixed = {str(d): {"mean_depth": float(d), "mean_acc": float(np.mean([out["tasks"][t]["single_acc_by_depth"][d] for t in tasks]))}
             for d in depths}
    agreement = {t: {d: float((logp[t]["test"][d].argmax(-1) == logp[t]["test"][final].argmax(-1)).mean()) for d in depths} for t in tasks}
    out.update({"anytime_policies": policies, "fixed_depth": fixed, "agreement_with_final_by_depth": agreement})
    L(f"  anytime bundle over {tasks}: fixed@{final} mean acc {fixed[str(final)]['mean_acc']:.4f}")
    for tau, r in policies["voi_flip"].items():
        L(f"    voi tau={tau:>4}: mean depth {r['single']['mean_depth']:5.2f}  acc single {r['single']['mean_acc']:.4f}  fused {r['fused']['mean_acc']:.4f}")
    for thr, r in policies["msp_threshold"].items():
        L(f"    msp>{thr:>4}: mean depth {r['single']['mean_depth']:5.2f}  acc {r['single']['mean_acc']:.4f}")
    for pat, r in policies["patience"].items():
        L(f"    patience {pat}: mean depth {r['single']['mean_depth']:5.2f}  acc {r['single']['mean_acc']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--datasets", nargs="*", default=["clinc150", "banking77"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.device = args.device or "cpu"
        args.datasets = ["clinc150"] if args.datasets == ["clinc150", "banking77"] else args.datasets
    out, logp = out_paths(args, "anytime")
    L = Logger(logp)
    t0 = time.time()
    trunk = Trunk(device=args.device)
    results = {"config": vars(args), "device": str(trunk.device), "datasets": {}}
    for name in args.datasets:
        results["datasets"][name] = run_dataset(name, trunk, args, L)
        dump(results, out)
    results["wall_seconds"] = round(time.time() - t0, 1)
    dump(results, out)
    L(f"== done in {results['wall_seconds']}s; wrote {out}")


if __name__ == "__main__":
    main()
