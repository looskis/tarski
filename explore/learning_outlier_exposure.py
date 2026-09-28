"""Exploration J: outlier exposure using OTHER tasks' messages as free auxiliary negatives.

The brief reports CLINC150's out-of-scope (oos) binary decision as weak: 85-87% accuracy vs. 81.8% for
the trivial "always answer in-scope" baseline. Outlier Exposure (Hendrycks, Mazeika & Dietterich, ICLR
2019: https://arxiv.org/abs/1812.04606) trains a classifier to push its output toward UNIFORM on a set of
known outliers, alongside the normal loss on real labels -- it needs an auxiliary corpus of "definitely
not this task's real distribution" text. This harness happens to have one for free: since tarski serves
many tasks from one shared frozen trunk, ANOTHER already-loaded dataset's messages (here, Banking77's
banking intents -- a different domain entirely) are natural "known outliers" for CLINC's oos detector,
requiring no extra data collection, only reading a second `Dataset` already supported by `tarski.data`.

This trains a `ProbeBranch` on CLINC's `oos` task with two loss terms per step: normal cross-entropy on a
CLINC batch, plus `lambda * CE(branch(banking77_batch), uniform)` on a Banking77 batch (same split depth,
same frozen trunk, different `FeatureCache`). Baseline is `lambda=0` (today's objective).

Novelty note (see research_notes/explore/learning.md, idea J): outlier exposure itself is established,
but it assumes a deliberately curated auxiliary set. Using another task ALREADY being served by the same
local multi-task trunk as that auxiliary set -- free, no extra collection, "whatever else this deployment
happens to route" -- is the part we did not find prior work on. This script only tests the CLINC oos use
case (a real distribution-shift / OOD detection problem, exactly what outlier exposure was designed for);
the typed-decisions majority-class collapse is a different failure mode (an objective/under-training
issue, addressed by ideas A-D) and is not expected to respond the same way, so it is not tested here.

Usage:
  .venv/bin/python explore/learning_outlier_exposure.py --smoke
  .venv/bin/python explore/learning_outlier_exposure.py --out results/tarski/explore_outlier_exposure.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `tarski`

from tarski.branches import ProbeBranch
from tarski.data import load_banking77, load_clinc
from tarski.train import FeatureCache, evaluate, predict_logits
from tarski.trunk import Trunk


def scheduled_epochs(n_train: int, bs: int, epochs: int, min_steps: int) -> int:
    per_epoch = max(1, (n_train + bs - 1) // bs)
    return max(epochs, -(-min_steps // per_epoch))


def train_with_oe(branch: ProbeBranch, cache: FeatureCache, train_idx: List[int], y_train: torch.Tensor,
                  val_idx: List[int], y_val: torch.Tensor, aux_cache: Optional[FeatureCache],
                  aux_idx: List[int], lam: float, epochs: int, bs: int, lr: float, seed: int,
                  min_steps: int = 300, min_val: int = 50, patience: int = 3) -> Dict:
    """Mirrors `tarski.train.train_branch`'s min-steps/last-epoch schedule (see
    learning_proper_score_objectives.py's `train_with_objective` for why: a custom loop that mixes two
    datasets per step can't reuse `train_branch` directly, but must still not under-train)."""
    epochs = scheduled_epochs(len(train_idx), bs, epochs, min_steps)
    select = len(val_idx) >= min_val
    min_epochs = max(1, (epochs + 1) // 2)
    torch.manual_seed(seed)
    dev = cache.trunk.device
    branch.to(dev).train()
    opt = torch.optim.AdamW(branch.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    aux_perm = list(range(len(aux_idx))) if aux_idx else []
    rng.shuffle(aux_perm)
    cursor = [0]

    def next_aux(k: int) -> List[int]:
        if not aux_perm:
            return []
        if cursor[0] + k > len(aux_perm):
            rng.shuffle(aux_perm)
            cursor[0] = 0
        sel = aux_perm[cursor[0]:cursor[0] + k]
        cursor[0] += k
        return [aux_idx[i] for i in sel]

    best, best_state, bad, ep = -1.0, None, 0, 0
    for ep in range(epochs):
        branch.train()
        order = list(range(len(train_idx)))
        rng.shuffle(order)
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            idx = [train_idx[j] for j in sel]
            h, ctx = cache.batch(branch.split, idx)
            z = branch.logits(h, ctx).float()
            loss = F.cross_entropy(z, y_train[sel].to(dev))
            if lam > 0 and aux_cache is not None:
                aux_ids = next_aux(len(sel))
                ha, actx = aux_cache.batch(branch.split, aux_ids)
                za = branch.logits(ha, actx).float()
                oe_loss = -F.log_softmax(za, -1).mean(-1).mean()  # cross-entropy to the uniform target
                loss = loss + lam * oe_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
            opt.step()
        va = float((predict_logits(branch, cache, val_idx).argmax(-1) == y_val).float().mean())
        if not select:
            best = va
            continue
        if va > best:
            best, best_state, bad = va, {k: v.clone() for k, v in branch.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience and ep + 1 >= min_epochs:
                break
    if select and best_state is not None:
        branch.load_state_dict(best_state)
    branch.eval()
    return {"val_acc": best, "epochs_run": ep + 1, "epochs_scheduled": epochs}


def run(split: int, lambdas: List[float], epochs: int, bs: int, lr: float, seed: int, device: str,
       cap_clinc_train: Optional[int], cap_clinc_val: Optional[int], cap_clinc_test: Optional[int],
       cap_aux: Optional[int], min_steps: int, log=print) -> Dict:
    t0 = time.time()
    trunk = Trunk(device=device)
    clinc = load_clinc(seed=seed)
    bank = load_banking77(seed=seed)
    log(f"loaded {clinc.summary()}")
    log(f"loaded {bank.summary()} (used only as auxiliary out-of-task text)")

    rng = np.random.default_rng(seed)

    def cap(rows, n):
        if n and len(rows) > n:
            idx = sorted(rng.choice(len(rows), n, replace=False).tolist())
            return [rows[i] for i in idx]
        return rows

    train = cap(clinc.train, cap_clinc_train)
    val = cap(clinc.val, cap_clinc_val)
    test = cap(clinc.test, cap_clinc_test)
    aux = cap(bank.train, cap_aux)
    labels = clinc.tasks["oos"].labels  # ["in_scope", "out_of_scope"]
    in_scope_idx = labels.index("in_scope")

    texts = [e.text for e in train] + [e.text for e in val] + [e.text for e in test]
    cache = FeatureCache(trunk, texts, [split], clinc.max_len)
    n_tr, n_va = len(train), len(val)
    train_idx = list(range(0, n_tr))
    val_idx = list(range(n_tr, n_tr + n_va))
    test_idx = list(range(n_tr + n_va, len(texts)))
    y_train = torch.tensor([e.y["oos"] for e in train])
    y_val = torch.tensor([e.y["oos"] for e in val])
    y_test = torch.tensor([e.y["oos"] for e in test])
    log(f"cached trunk depth {split} for {len(texts)} CLINC messages in {cache.seconds:.1f}s")

    aux_cache, aux_idx = None, []
    if aux:
        aux_cache = FeatureCache(trunk, [e.text for e in aux], [split], bank.max_len)
        aux_idx = list(range(len(aux)))
        log(f"cached trunk depth {split} for {len(aux)} Banking77 auxiliary messages in "
            f"{aux_cache.seconds:.1f}s")

    always_in_scope_acc = float((y_test.numpy() == in_scope_idx).mean())
    log(f"trivial 'always in-scope' baseline: {always_in_scope_acc:.3f}")

    results = {}
    for lam in lambdas:
        branch = ProbeBranch(split, labels, trunk.hidden)
        info = train_with_oe(branch, cache, train_idx, y_train, val_idx, y_val,
                             aux_cache if lam > 0 else None, aux_idx, lam, epochs, bs, lr, seed,
                             min_steps=min_steps)
        z = predict_logits(branch, cache, test_idx)
        probs = torch.softmax(z, -1).numpy()
        m = evaluate(probs, y_test.numpy())
        m.update({"epochs_run": info["epochs_run"], "epochs_scheduled": info["epochs_scheduled"],
                  "lambda": lam})
        results[f"lambda={lam}"] = m
        log(f"  lambda={lam}: acc={m['acc']:.3f} macro_f1={m['macro_f1']:.3f} ece={m['ece']:.3f} "
            f"epochs={m['epochs_run']}/{m['epochs_scheduled']}")

    return {"config": {"split": split, "lambdas": lambdas, "epochs": epochs, "bs": bs, "lr": lr,
                       "seed": seed, "min_steps": min_steps, "n_train": len(train), "n_val": len(val),
                       "n_test": len(test), "n_aux": len(aux)},
           "always_in_scope_baseline_acc": always_in_scope_acc, "results": results,
           "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--lambdas", default="0,0.5,1.0,2.0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]
    if args.smoke:
        device = "cpu"
        caps = dict(cap_clinc_train=40, cap_clinc_val=16, cap_clinc_test=20, cap_aux=16)
        min_steps = 8
    else:
        device = args.device
        caps = dict(cap_clinc_train=6000, cap_clinc_val=800, cap_clinc_test=1500, cap_aux=1500)
        min_steps = 300

    result = run(args.split, lambdas, args.epochs, args.bs, args.lr, args.seed, device, **caps,
                min_steps=min_steps)

    print("\n=== outlier-exposure summary (CLINC150 oos) ===")
    print(f"always-in-scope baseline acc: {result['always_in_scope_baseline_acc']:.3f}")
    for name, m in result["results"].items():
        print(f"  {name:12s} acc={m['acc']:.3f} macro_f1={m['macro_f1']:.3f} ece={m['ece']:.3f}")

    out = args.out or (None if args.smoke else "results/tarski/explore_outlier_exposure.json")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {out}")
    print(f"total elapsed: {result['elapsed_s']}s")


if __name__ == "__main__":
    main()
