"""Skeptic lens, idea 1: CLINC150's "oos" binary task has a severe, uncorrected train/test class-prior
shift, and tarski does nothing about it.

Evidence (measured directly from tarski.data.load_clinc()):
  out-of-scope fraction   train 1.6%   val 3.2%   test 18.2%
This is the CLINC150 "plus" split's own design (Larson et al., EMNLP 2019): a handful of OOS examples
in train/val, many more at test time, to check generalisation to unseen-shape OOS traffic. tarski's
`oos` branch is trained with plain (soft-or-hard) cross-entropy and calibrated with a single scalar
temperature (tarski/train.py fit_temperature / evaluate). Neither can correct an asymmetric class-prior
shift: temperature scaling only sharpens or flattens a distribution, it cannot move probability mass
between classes in proportion to a prior that changed between train and test. results/tarski/tables.md
reports oos accuracy of 85-87%, barely above the 81.8% "always answer in-scope" baseline on this same
test set -- consistent with a classifier whose decision threshold was tuned for a ~2% positive rate
being run against an 18% positive rate.

This script trains one oos branch the normal tarski way (no core-code changes), then applies a classic,
well-established label/prior-shift correction on top of its already-trained, already-calibrated softmax
outputs -- no retraining, and no test labels are used by the correction itself (only for evaluation) --
and reports accuracy / macro-F1 / oos recall & precision before and after.

Fix and its prior art (see docs/research/notes/explore/skeptic.md for the full citations):
  - Saerens, Latinne & Decaestecker, "Adjusting the output of a classifier to new a priori
    probabilities: a simple procedure", Neural Computation 14(1), 2002. The EM re-estimation used here.
  - Lipton, Wang & Smola, "Detecting and Correcting for Label Shift with Black Box Predictors",
    ICML 2018 (BBSE) -- the modern generalisation/estimation-theory treatment of the same idea.
  - Two 2024/2025 papers already flag *this exact* CLINC150 train/test OOS prior mismatch and propose
    threshold-style fixes: "Improved Out-of-Scope Intent Classification with Dual Encoding and
    Threshold-based Re-Classification" (arXiv 2405.19967) and DROID (arXiv 2510.14110). Both calibrate a
    *decision threshold* on in-domain validation data; this script instead corrects the *class prior*
    the softmax implicitly encodes, which is a different (and, unlike a tuned threshold, label-free at
    correction time) angle on the same problem.
  Verdict: the correction technique is NOT novel (it is 20+ years old and has already been applied to
  CLINC OOS specifically). What is new here is plugging it into tarski's frozen-trunk branch harness,
  where it costs nothing beyond the softmax outputs tarski already computes -- and demonstrating how
  much of the "weak OOS detection" finding in results/tarski/tables.md is attributable to this
  uncorrected shift rather than to the split-trunk architecture.

Usage:
  python -m explore.skeptic_oos_priorshift --smoke
  python -m explore.skeptic_oos_priorshift --out results/tarski/explore_oos_priorshift.json
  python -m explore.skeptic_oos_priorshift --kind blocks --split 11 --depth 2 \
      --out results/tarski/explore_oos_priorshift_blocks.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from tarski import data, train
from tarski.trunk import Trunk


def subsample(ds: data.Dataset, n_train: int, n_val: int, n_test: int, seed: int,
             min_oos_train: int = 0, min_oos_val: int = 0) -> data.Dataset:
    """A random, size-capped copy of the dataset. Train/val optionally guarantee a small floor of
    OOS examples (`min_oos_*`) -- with only ~1.6% OOS in the real train split, an unstratified random
    slice of a few hundred rows can easily contain zero, which would make the OOS branch degenerate for
    reasons that have nothing to do with the phenomenon under test (see the smoke-mode note below).
    Test is always a plain random slice, since it is only used for evaluation, never for fitting."""
    rng = random.Random(seed)

    def take(examples: List, n: int, min_pos: int) -> List:
        examples = list(examples)
        rng.shuffle(examples)
        if min_pos <= 0:
            return examples[:min(n, len(examples))]
        pos = [e for e in examples if e.y["oos"] == 1][:min_pos]
        neg = [e for e in examples if e.y["oos"] == 0]
        rest = neg[:max(0, n - len(pos))]
        out = pos + rest
        rng.shuffle(out)
        return out

    return data.Dataset(ds.name + "-sub", ds.tasks, take(ds.train, n_train, min_oos_train),
                        take(ds.val, n_val, min_oos_val), take(ds.test, n_test, 0), ds.max_len)


def class_prior(y: np.ndarray, n_labels: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_labels).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def saerens_correction(probs: np.ndarray, prior_train: np.ndarray, iters: int = 100,
                       tol: float = 1e-7) -> "tuple[np.ndarray, np.ndarray]":
    """Saerens, Latinne & Decaestecker (2002): EM re-estimation of the test class prior from a
    classifier's own posteriors, then a Bayes-corrected posterior. No test labels are used."""
    prior_tr = np.clip(prior_train, 1e-12, None)
    prior_te = prior_train.copy()
    for _ in range(iters):
        adj = probs * (prior_te / prior_tr)
        adj = adj / np.clip(adj.sum(-1, keepdims=True), 1e-12, None)
        new_prior = adj.mean(0)
        if np.abs(new_prior - prior_te).max() < tol:
            prior_te = new_prior
            break
        prior_te = new_prior
    adj = probs * (prior_te / prior_tr)
    adj = adj / np.clip(adj.sum(-1, keepdims=True), 1e-12, None)
    return adj, prior_te


def oracle_correction(probs: np.ndarray, prior_train: np.ndarray, prior_test_true: np.ndarray) -> np.ndarray:
    """Upper bound: correction using the TRUE test prior (as if it were known in advance)."""
    adj = probs * (prior_test_true / np.clip(prior_train, 1e-12, None))
    return adj / np.clip(adj.sum(-1, keepdims=True), 1e-12, None)


def report(name: str, probs: np.ndarray, y: np.ndarray, labels: List[str]) -> Dict:
    pred = probs.argmax(-1)
    acc = float((pred == y).mean())
    macro_f1 = float(f1_score(y, pred, average="macro", labels=np.arange(len(labels)), zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=np.arange(len(labels)), zero_division=0)
    oos_idx = labels.index("out_of_scope")
    return {"name": name, "acc": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "oos_precision": round(float(p[oos_idx]), 4), "oos_recall": round(float(r[oos_idx]), 4),
            "oos_f1": round(float(f[oos_idx]), 4), "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny subset, finishes in well under 2 minutes")
    ap.add_argument("--kind", default="probe", choices=["probe", "blocks"])
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--depth", type=int, default=2, help="blocks branch depth (ignored for probe)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ds = data.load_clinc(seed=a.seed)
    if a.smoke:
        # A plain random 300-row slice of a 1.6%-OOS train split would very likely contain 0-2 OOS
        # rows -- too few for the branch to learn ANY oos-vs-in-scope signal, which would make the
        # correction look broken for a reason unrelated to prior shift (a degenerate, uninformative
        # classifier can't be rescued by any prior correction, oracle included -- see skeptic.md). A
        # small guaranteed floor keeps the smoke run a real, if tiny, instance of the same phenomenon.
        ds = subsample(ds, n_train=300, n_val=80, n_test=150, seed=a.seed, min_oos_train=15, min_oos_val=8)
        a.split = min(a.split, 4)
        device = "cpu"
        epochs = a.epochs or 2
    else:
        device = a.device
        epochs = a.epochs or (20 if a.kind == "probe" else 6)

    trunk = Trunk(device=device, max_len=ds.max_len)
    task = "oos"
    labels = ds.tasks[task].labels
    texts = [e.text for e in ds.train + ds.val + ds.test]

    t0 = time.time()
    fc = train.FeatureCache(trunk, texts, [a.split], ds.max_len)
    n_tr, n_va = len(ds.train), len(ds.val)
    idx = {"train": list(range(0, n_tr)), "val": list(range(n_tr, n_tr + n_va)),
          "test": list(range(n_tr + n_va, len(fc.lengths)))}
    allx = ds.train + ds.val + ds.test
    y = {s: torch.tensor([allx[i].y[task] for i in idx[s]]) for s in idx}
    print(f"train/val/test = {len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}, "
         f"oos frac = {y['train'].float().mean():.3f}/{y['val'].float().mean():.3f}/{y['test'].float().mean():.3f}")

    branch = train.make_branch(a.kind, a.split, labels, trunk, a.depth, "next")
    train.train_branch(branch, fc, idx["train"], y["train"], None, idx["val"], y["val"],
                       epochs=epochs, seed=a.seed, log=print)
    t_cal = train.fit_temperature(train.predict_logits(branch, fc, idx["val"]), y["val"]) \
        if len(idx["val"]) >= 30 else 1.0
    branch.temperature.fill_(t_cal)

    z_test = train.predict_logits(branch, fc, idx["test"])
    probs_base = torch.softmax(z_test / t_cal, -1).numpy()
    y_test = y["test"].numpy()

    prior_train = class_prior(y["train"].numpy(), len(labels))
    prior_test_true = class_prior(y_test, len(labels))
    probs_em, prior_te_est = saerens_correction(probs_base, prior_train)
    probs_oracle = oracle_correction(probs_base, prior_train, prior_test_true)

    always_inscope_acc = float((y_test == labels.index("in_scope")).mean())
    results = {
        "dataset": "clinc150" + ("-smoke" if a.smoke else ""), "kind": a.kind, "split": a.split,
        "depth": a.depth if a.kind == "blocks" else None, "epochs": epochs,
        "n_train": len(idx["train"]), "n_val": len(idx["val"]), "n_test": len(idx["test"]),
        "prior_train": prior_train.tolist(), "prior_test_true": prior_test_true.tolist(),
        "prior_test_estimated_by_em": prior_te_est.tolist(), "temperature": round(t_cal, 4),
        "always_in_scope_baseline_acc": round(always_inscope_acc, 4),
        "baseline": report("baseline (uncorrected, temperature-scaled)", probs_base, y_test, labels),
        "saerens_em": report("saerens-em prior-shift correction (label-free)", probs_em, y_test, labels),
        "oracle_prior": report("oracle prior correction (upper bound, uses true test prior)",
                              probs_oracle, y_test, labels),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(results, indent=2))

    default_out = "results/tarski/explore_oos_priorshift_smoke.json" if a.smoke \
        else "results/tarski/explore_oos_priorshift.json"
    out = a.out or default_out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
