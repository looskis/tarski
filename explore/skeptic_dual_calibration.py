"""Skeptic lens, idea 3 (also quantifies bug 2): decouple correctness-calibration from soft-label
distribution calibration, and report both ECE variants on typed-decisions.

Background (see research_notes/explore/skeptic.md, item 2). `fit_temperature(logits, y, soft=None)`
(`tarski/train.py:103-115`) minimizes cross-entropy against `soft` whenever it is available, and
against hard one-hot otherwise. `tarski/train.py:258` (in `fit()`) passes `soft["val"]` whenever the
task has soft labels. Banking77/CLINC150 examples carry no `soft` field, so their one temperature is
fit the textbook way (Guo, Pleiss, Sun & Weinberger, "On Calibration of Modern Neural Networks", ICML
2017: minimize NLL of hard correctness) -- exactly what `ece_score` (`tarski/train.py:81-88`) measures.
typed-decisions is the only dataset with soft teacher labels, so it is the only one whose temperature is
fit to match the TEACHER's label distribution instead of the BRANCH's own correctness -- but the
reported ECE always measures against hard correctness regardless. This predicts that typed-decisions'
reported ECE understates how good a *correctness*-calibrated version of the same branch could be, and
overstates how good today's default is at the thing ECE actually measures.

This script trains one branch per typed-decisions task exactly the way `tarski.train.fit` does (same
`FeatureCache`/`make_branch`/`train_branch` calls, no core-file changes), then fits THREE temperatures
on the same validation logits and reports test-set metrics for each:
  T=1        uncalibrated baseline
  T_hard     fit_temperature(logits, y)                    -- what banking77/CLINC already get for free
  T_soft     fit_temperature(logits, y, soft)               -- today's typed-decisions default
Accuracy is identical across all three (temperature scaling never changes argmax) -- included only as a
sanity check. ECE (vs hard correctness) and Brier-vs-soft are expected to trade off between T_hard and
T_soft; T=1 is the "no calibration at all" floor.

Novelty check: standard temperature scaling calibrates against hard correctness (Guo et al. 2017).
Calibrating under label noise / soft targets has related literature (Muller, Kornblith & Hinton, "When
Does Label Smoothing Help?", NeurIPS 2019; Lukasik et al., "Does label smoothing mitigate label noise?",
ICML 2020) but a dual-temperature report that explicitly separates "match the teacher's distribution"
from "be calibrated w.r.t. your own correctness" for a distilled/soft-label classifier does not appear
to be a named, standard practice. This is a modest, mostly-methodological contribution: a diagnostic +
a one-line-scale recommendation (report/pick the temperature matching what you'll actually use ECE for),
not a new algorithm.

Usage:
  python -m explore.skeptic_dual_calibration --smoke
  python -m explore.skeptic_dual_calibration --out results/tarski/explore_dual_calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from tarski import data, train
from tarski.trunk import Trunk


def subsample(ds: data.Dataset, n_train: int, n_val: int, n_test: int, seed: int) -> data.Dataset:
    rng = random.Random(seed)

    def take(examples: List, n: int) -> List:
        examples = list(examples)
        rng.shuffle(examples)
        return examples[:min(n, len(examples))]

    return data.Dataset(ds.name + "-sub", ds.tasks, take(ds.train, n_train), take(ds.val, n_val),
                        take(ds.test, n_test), ds.max_len)


def train_one_task(trunk: Trunk, fc: train.FeatureCache, split_idx: Dict, allx: List, task: str,
                   ds: data.Dataset, kind: str, split: int, depth: int, epochs: int, seed: int,
                   log) -> Dict:
    sel = {s: [i for i in split_idx[s] if task in allx[i].y] for s in split_idx}
    y: Dict[str, torch.Tensor] = {}
    soft: Dict[str, Optional[torch.Tensor]] = {}
    for s in sel:
        ex = [allx[i] for i in sel[s]]
        y[s] = torch.tensor([e.y[task] for e in ex])
        soft[s] = torch.tensor(np.stack([e.soft[task] for e in ex])) \
            if ex and all(task in e.soft for e in ex) else None
    labels = ds.tasks[task].labels
    branch = train.make_branch(kind, split, labels, trunk, depth, "next")
    train.train_branch(branch, fc, sel["train"], y["train"], soft["train"], sel["val"], y["val"],
                       epochs=epochs, seed=seed, log=log)
    return {"labels": labels, "y_val": y["val"], "soft_val": soft["val"], "y_test": y["test"],
           "soft_test": soft["test"], "val_logits": train.predict_logits(branch, fc, sel["val"]),
           "test_logits": train.predict_logits(branch, fc, sel["test"]), "n_val": len(sel["val"])}


def dual_calibrate(info: Dict) -> Dict:
    """Fit T=1 / T_hard / T_soft on the same val logits, report test metrics under each."""
    val_logits, y_val, soft_val = info["val_logits"], info["y_val"], info["soft_val"]
    test_logits, y_test, soft_test = info["test_logits"], info["y_test"], info["soft_test"]
    y_test_np = y_test.numpy()
    soft_test_np = soft_test.numpy() if soft_test is not None else None
    enough_val = info["n_val"] >= 30

    temps = {"T1": 1.0}
    temps["T_hard"] = train.fit_temperature(val_logits, y_val) if enough_val else 1.0
    temps["T_soft"] = (train.fit_temperature(val_logits, y_val, soft_val)
                       if (enough_val and soft_val is not None) else temps["T_hard"])

    out = {"temperatures": temps, "has_soft": soft_val is not None, "n_val": info["n_val"]}
    for name, t in temps.items():
        probs = torch.softmax(test_logits / t, -1).numpy()
        out[name] = train.evaluate(probs, y_test_np, soft_test_np)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny subset, finishes in well under 2 minutes")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--kind", default="probe", choices=["probe", "blocks"])
    ap.add_argument("--split", type=int, default=22)
    ap.add_argument("--depth", type=int, default=2, help="blocks branch depth (ignored for probe)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ds = data.load_typed_decisions(seed=a.seed)
    if a.smoke:
        # Val is left at its real (small, per-task 22-41-row) size on purpose: subsampling it further
        # would push every task below the n_val>=30 threshold and make T_hard/T_soft both silently
        # fall back to T=1, so the smoke run would "pass" without ever exercising the code path this
        # script exists to probe. Train/test are cut down; that only affects branch quality, not
        # whether the dual-calibration logic itself runs correctly.
        ds = subsample(ds, n_train=150, n_val=len(ds.val), n_test=60, seed=a.seed)
        a.split = min(a.split, 4)
        a.kind, a.depth = "probe", 1
        device = "cpu"
        # invoice_processing has the largest per-task validation split (41 rows) of the four
        # workflows, the best chance at smoke scale of clearing the n_val>=30 threshold below.
        a.tasks = a.tasks or [t for t in ds.tasks if t.startswith("invoice_processing.")][:3]
    else:
        device = a.device
    tasks = a.tasks or list(ds.tasks)

    trunk = Trunk(device=device, max_len=ds.max_len)
    texts = [e.text for e in ds.train + ds.val + ds.test]
    t0 = time.time()
    fc = train.FeatureCache(trunk, texts, [a.split], ds.max_len)
    n_tr, n_va = len(ds.train), len(ds.val)
    split_idx = {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va),
                "test": range(n_tr + n_va, len(fc.lengths))}
    allx = ds.train + ds.val + ds.test

    per_task = {}
    for task in tasks:
        print(f"-- {task}", flush=True)
        info = train_one_task(trunk, fc, split_idx, allx, task, ds, a.kind, a.split, a.depth,
                              a.epochs, a.seed, log=lambda s: None)
        per_task[task] = dual_calibrate(info)
        r = per_task[task]
        print(f"   n_val={r['n_val']} has_soft={r['has_soft']} T_hard={r['temperatures']['T_hard']:.2f} "
             f"T_soft={r['temperatures']['T_soft']:.2f} | ECE  T1={r['T1']['ece']:.3f} "
             f"T_hard={r['T_hard']['ece']:.3f} T_soft={r['T_soft']['ece']:.3f}" +
             (f" | Brier-soft  T_hard={r['T_hard'].get('brier_soft', float('nan')):.3f} "
              f"T_soft={r['T_soft'].get('brier_soft', float('nan')):.3f}" if r["has_soft"] else ""))

    def mean_over(metric, variant):
        vals = [per_task[t][variant][metric] for t in tasks if metric in per_task[t][variant]]
        return float(np.mean(vals)) if vals else None

    mean = {variant: {m: mean_over(m, variant) for m in ("acc", "macro_f1", "ece", "nll", "brier_soft")}
           for variant in ("T1", "T_hard", "T_soft")}
    n_with_soft = sum(1 for t in tasks if per_task[t]["has_soft"])
    print(json.dumps({"mean": mean, "n_tasks": len(tasks), "n_tasks_with_soft_labels": n_with_soft},
                     indent=2))
    print(f"headline: mean ECE today's default (T_soft) = {mean['T_soft']['ece']:.4f} vs "
         f"correctness-optimal (T_hard) = {mean['T_hard']['ece']:.4f} "
         f"(lower is better-calibrated w.r.t. hard correctness, which is what ECE measures); "
         f"mean Brier-vs-soft T_soft = {mean['T_soft']['brier_soft']:.4f} vs "
         f"T_hard = {mean['T_hard']['brier_soft']:.4f} (lower is closer to the teacher's distribution). "
         f"acc is identical across variants by construction: "
         f"{mean['T1']['acc']:.4f} / {mean['T_hard']['acc']:.4f} / {mean['T_soft']['acc']:.4f}")

    results = {"dataset": "typed-decisions" + ("-smoke" if a.smoke else ""), "kind": a.kind,
              "split": a.split, "depth": a.depth if a.kind == "blocks" else None, "epochs": a.epochs,
              "tasks": tasks, "n_tasks_with_soft_labels": n_with_soft, "per_task": per_task,
              "mean": mean, "seconds": round(time.time() - t0, 1)}
    default_out = "results/tarski/explore_dual_calibration_smoke.json" if a.smoke \
        else "results/tarski/explore_dual_calibration.json"
    out = a.out or default_out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
