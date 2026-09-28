"""Skeptic lens, idea 6: per-instance depth gating on top of tarski's per-task static split depth.

tarski picks one split depth per TASK (`tarski/autosplit.py`) and runs every instance of that task to
that same depth. `research_notes/novelty_check.md` (Claim 2) states plainly: "I found no NLP paper that
fixes a static exit layer per *task* and serves many such tasks from one trunk, other than Wei et al." --
and per-*instance* early exit (DeeBERT, PABEE, F-PABEE) is single-task/single-model, never combined with
per-task heterogeneous branches sharing one trunk. Combining both axes -- keep tarski's per-task branch
specialization, but let individual EASY instances of that task exit through a cheap shallow branch while
HARD instances fall through to the task's own deeper, more accurate branch -- is the untested combination.

Mechanism tested here: train a cheap shallow branch (depth k1) and a more accurate deep branch (depth k2
> k1) for the same task (both already exist as tarski concepts: a probe or shallow blocks branch, and a
deeper blocks branch). At serving time, run the shallow branch first; if its (temperature-calibrated)
confidence clears a threshold, use its prediction and never run the trunk past k1 for that instance; only
low-confidence instances get promoted to depth k2. Reported as an accuracy-vs-average-depth curve swept
over the threshold, compared against the two STATIC baselines tarski already reports
(results/tarski/tables.md): "always k1" (today's shallow choice) and "always k2" (today's deep choice).
A win looks like: accuracy close to "always k2" at an average depth much closer to k1 than k2.

Note on cost accounting: this script computes both depths' cached states up front (for evaluation
convenience -- FeatureCache supports multiple depths from one trunk pass). The "average depth" figure
reported is NOT what this script itself spends; it is what a real per-instance-gated deployment would
spend, under the (accurate, since ModernBERT's layers run strictly in order) assumption that the trunk
can simply stop at k1 for confident instances and only continue to k2 for escalated ones.

Novelty check: per-instance confidence-gated early exit is well established for a single classifier
(Xin et al., "DeeBERT", ACL 2020; Zhou et al., "PABEE", NeurIPS 2020; F-PABEE, arXiv:2305.11916, 2023).
Static per-task exit depth on a shared frozen trunk is Wei et al. (ACL 2022) and Mainstream (ATC 2018),
both of which apply ONE fixed depth per task to EVERY instance -- no per-instance adaptivity on top.
Combining the two axes for a single task's own branch (not early-exiting a whole multi-task model, just
letting one task's easy instances skip that task's own deep layers) does not appear in the reviewed
prior art (research_notes/novelty_check.md, Claim 2's "gap" paragraph names exactly this combination as
untested). This script does not implement per-request trunk truncation (that lives in tarski/engine.py,
out of scope here); it validates the ACCURACY/depth trade-off the mechanism would need to be worth
building, using the FeatureCache states tarski already computes.

Usage:
  python -m explore.skeptic_depth_gating --smoke
  python -m explore.skeptic_depth_gating --out results/tarski/explore_depth_gating.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import List

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


def train_at_depth(trunk, fc, split_idx, allx, task, ds, kind, split, depth, epochs, seed):
    sel = {s: [i for i in split_idx[s] if task in allx[i].y] for s in split_idx}
    y = {s: torch.tensor([allx[i].y[task] for i in sel[s]]) for s in sel}
    labels = ds.tasks[task].labels
    branch = train.make_branch(kind, split, labels, trunk, depth, "next")
    train.train_branch(branch, fc, sel["train"], y["train"], None, sel["val"], y["val"],
                       epochs=epochs, seed=seed, log=lambda s: None)
    t = train.fit_temperature(train.predict_logits(branch, fc, sel["val"]), y["val"]) \
        if len(sel["val"]) >= 30 else 1.0
    branch.temperature.fill_(t)
    z_test = train.predict_logits(branch, fc, sel["test"])
    probs_test = torch.softmax(z_test / t, -1).numpy()
    z_val = train.predict_logits(branch, fc, sel["val"])
    probs_val = torch.softmax(z_val / t, -1).numpy()
    return {"y_test": y["test"].numpy(), "probs_test": probs_test, "probs_val": probs_val,
           "y_val": y["val"].numpy(), "temperature": t, "n_test": len(sel["test"])}


def gated_curve(shallow: dict, deep: dict, thresholds: List[float]) -> List[dict]:
    """Sweep the shallow branch's confidence threshold: below it, escalate to the deep branch."""
    conf = shallow["probs_test"].max(-1)
    shallow_pred = shallow["probs_test"].argmax(-1)
    deep_pred = deep["probs_test"].argmax(-1)
    y = shallow["y_test"]
    assert np.array_equal(y, deep["y_test"]), "shallow/deep test label order mismatch"
    rows = []
    for th in thresholds:
        escalate = conf < th
        pred = np.where(escalate, deep_pred, shallow_pred)
        acc = float((pred == y).mean())
        frac_esc = float(escalate.mean())
        rows.append({"threshold": th, "frac_escalated": frac_esc, "acc": acc})
    return rows


def pick_threshold_by_val(shallow: dict, deep: dict, thresholds: List[float]) -> float:
    """Choose the threshold using VAL accuracy only (test labels are for reporting, not tuning)."""
    conf = shallow["probs_val"].max(-1)
    shallow_pred = shallow["probs_val"].argmax(-1)
    deep_pred = deep["probs_val"].argmax(-1)
    y = shallow["y_val"]
    best_th, best_acc = thresholds[0], -1.0
    for th in thresholds:
        escalate = conf < th
        pred = np.where(escalate, deep_pred, shallow_pred)
        acc = float((pred == y).mean())
        if acc > best_acc:
            best_acc, best_th = acc, th
    return best_th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny subset, finishes in well under 2 minutes")
    ap.add_argument("--dataset", default="banking77", choices=["banking77", "clinc150"])
    ap.add_argument("--task", default="intent")
    ap.add_argument("--k-shallow", type=int, default=4)
    ap.add_argument("--k-deep", type=int, default=18)
    ap.add_argument("--shallow-kind", default="probe", choices=["probe", "blocks"])
    ap.add_argument("--deep-kind", default="blocks", choices=["probe", "blocks"])
    ap.add_argument("--deep-depth", type=int, default=1, help="blocks depth for the deep branch")
    ap.add_argument("--epochs-shallow", type=int, default=20)
    ap.add_argument("--epochs-deep", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.smoke:
        a.dataset, a.task = "banking77", "intent"
        a.k_shallow, a.k_deep, a.deep_depth = 2, 6, 1
        a.epochs_shallow, a.epochs_deep = 4, 2
        device = "cpu"
    else:
        device = a.device

    ds = data.load(a.dataset, seed=a.seed) if a.dataset == "banking77" else data.load_clinc(seed=a.seed)
    if a.smoke:
        ds = subsample(ds, n_train=300, n_val=100, n_test=150, seed=a.seed)

    trunk = Trunk(device=device, max_len=ds.max_len)
    texts = [e.text for e in ds.train + ds.val + ds.test]
    t0 = time.time()
    fc = train.FeatureCache(trunk, texts, [a.k_shallow, a.k_deep], ds.max_len)
    n_tr, n_va = len(ds.train), len(ds.val)
    split_idx = {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va),
                "test": range(n_tr + n_va, len(fc.lengths))}
    allx = ds.train + ds.val + ds.test

    print(f"training shallow branch: {a.shallow_kind}@{a.k_shallow}", flush=True)
    shallow = train_at_depth(trunk, fc, split_idx, allx, a.task, ds, a.shallow_kind, a.k_shallow, 1,
                             a.epochs_shallow, a.seed)
    print(f"training deep branch: {a.deep_kind}@{a.k_deep}+{a.deep_depth}", flush=True)
    deep = train_at_depth(trunk, fc, split_idx, allx, a.task, ds, a.deep_kind, a.k_deep, a.deep_depth,
                          a.epochs_deep, a.seed)

    acc_shallow_only = float((shallow["probs_test"].argmax(-1) == shallow["y_test"]).mean())
    acc_deep_only = float((deep["probs_test"].argmax(-1) == deep["y_test"]).mean())
    print(f"static baselines: always-{a.k_shallow} acc={acc_shallow_only:.4f}   "
         f"always-{a.k_deep} acc={acc_deep_only:.4f}", flush=True)

    thresholds = [round(t, 3) for t in np.linspace(0.05, 0.999, 25 if not a.smoke else 10)]
    curve = gated_curve(shallow, deep, thresholds)
    for row in curve:
        avg_depth = a.k_shallow * (1 - row["frac_escalated"]) + a.k_deep * row["frac_escalated"]
        row["avg_depth"] = round(avg_depth, 2)
        print(f"  th={row['threshold']:.3f}  escalate={row['frac_escalated']:.3f}  "
             f"avg_depth={avg_depth:.2f}  acc={row['acc']:.4f}", flush=True)

    val_th = pick_threshold_by_val(shallow, deep, thresholds)
    picked = next(r for r in curve if r["threshold"] == val_th)
    picked_avg_depth = a.k_shallow * (1 - picked["frac_escalated"]) + a.k_deep * picked["frac_escalated"]

    results = {
        "dataset": a.dataset + ("-smoke" if a.smoke else ""), "task": a.task,
        "k_shallow": a.k_shallow, "k_deep": a.k_deep,
        "shallow_kind": a.shallow_kind, "deep_kind": a.deep_kind, "deep_depth": a.deep_depth,
        "n_train": len(list(split_idx["train"])), "n_val": len(list(split_idx["val"])),
        "n_test": len(list(split_idx["test"])),
        "static_baselines": {f"always_{a.k_shallow}": acc_shallow_only,
                             f"always_{a.k_deep}": acc_deep_only},
        "threshold_curve": curve,
        "picked_by_val": {"threshold": val_th, "avg_depth": round(picked_avg_depth, 2),
                          "test_acc": picked["acc"], "frac_escalated": picked["frac_escalated"]},
        "seconds": round(time.time() - t0, 1),
    }
    gain_captured = ((picked["acc"] - acc_shallow_only) / max(acc_deep_only - acc_shallow_only, 1e-9))
    depth_saved_frac = 1.0 - (picked_avg_depth - a.k_shallow) / max(a.k_deep - a.k_shallow, 1e-9)
    print(f"headline: val-picked threshold {val_th:.3f} -> test acc {picked['acc']:.4f} at avg depth "
         f"{picked_avg_depth:.2f} (vs always-{a.k_shallow}={acc_shallow_only:.4f} at depth {a.k_shallow}, "
         f"always-{a.k_deep}={acc_deep_only:.4f} at depth {a.k_deep}). Captured "
         f"{gain_captured * 100:.0f}% of the shallow-to-deep accuracy gain while running the trunk only "
         f"{depth_saved_frac * 100:.0f}% of the way from k_shallow to k_deep on average.", flush=True)
    results["gain_captured_frac"] = gain_captured
    results["depth_saved_frac"] = depth_saved_frac

    default_out = "results/tarski/explore_depth_gating_smoke.json" if a.smoke \
        else "results/tarski/explore_depth_gating.json"
    out = a.out or default_out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
