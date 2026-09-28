"""Exploration F: self-training on UNLABELLED traffic, no teacher, no extra labels at all.

Idea E (`explore/learning_active_distillation.py`) spends compute calling a real, heavier local model
(`laya`) for extra supervision. This script asks the cheaper, more pessimistic question: if there is no
teacher and no budget for more human labels, can a branch still improve itself by looking at raw
unlabelled traffic it already has -- e.g. a user's un-annotated Slack history -- via self-training?

Mechanism: train a small `ProbeBranch` on a small labelled seed set (simulating "a user only labelled a
few dozen messages so far"). For each round, run TWO stochastic forward passes of the CURRENT branch over
the still-unlabelled pool with dropout left on (`ProbeBranch` already has `nn.Dropout`; we force
`branch.train()` for the forward but wrap it in `torch.no_grad()`, so this only injects two different
dropout masks, never updates any weights) and score how much the two views AGREE with
`laya.common.proper_reward` (used here purely as a symmetric agreement/confidence score between two
model outputs, not against any external target: `0.5*(proper_reward(p1,p2,...) + proper_reward(p2,p1,
...))`, maximal when both views are confident and identical). Admit the highest-agreement fraction of the
remaining pool each round as pseudo-labelled, with soft target `(p1+p2)/2`, and retrain (from scratch, on
seed + all admitted pseudo-labels so far) with `tarski.train.train_branch` unchanged. Because this is a
benchmark with real gold labels, we also report (diagnostic only, never used for training) how often the
admitted pseudo-labels actually agree with the hidden gold label -- i.e. whether the procedure is
confidently right or confidently fooling itself.

Novelty note (see docs/research/notes/explore/learning.md, idea F): self-training (Noisy Student, Xie et al.
2020) and MC-dropout consistency are both established. Using a strictly proper scoring rule as the
admission/agreement criterion between two stochastic views of the SAME tiny frozen-trunk branch, instead
of the usual symmetric KL or a raw softmax-confidence threshold, is a small variant we did not find
written up; this script is an honest test of whether that variant is worth anything here, not a claim
that self-training itself is new.

Usage:
  .venv/bin/python explore/learning_self_training.py --smoke
  .venv/bin/python explore/learning_self_training.py --out results/tarski/explore_self_training.json
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `tarski`

from tarski.branches import ProbeBranch
from tarski.data import Example, load_typed_decisions
from tarski.train import FeatureCache, evaluate, predict_logits, train_branch
from tarski.trunk import Trunk

from laya.common import QTYPES, proper_reward

DEFAULT_TASKS_FULL = ["customer_service.action", "customer_service.urgency"]
DEFAULT_TASK_SMOKE = "customer_service.action"


def infer_qtype(labels: List[str]) -> int:
    """typed-decisions stringifies "score" question options as "0".."K-1"; see the identical helper in
    learning_proper_score_objectives.py (duplicated rather than imported, so this script stays a single
    self-contained file)."""
    return QTYPES["score"] if all(l.isdigit() for l in labels) and \
        [int(l) for l in labels] == list(range(len(labels))) else QTYPES["choice"]


@torch.no_grad()
def mc_forward(branch: ProbeBranch, cache: FeatureCache, idx: List[int], bs: int = 128) -> torch.Tensor:
    """One stochastic forward pass (dropout ON) over `idx`, no gradient, weights untouched."""
    if not idx:
        return torch.zeros(0, branch.n_labels)
    branch.train()
    order = sorted(range(len(idx)), key=lambda j: cache.lengths[idx[j]])
    out = torch.zeros(len(idx), branch.n_labels)
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        h, ctx = cache.batch(branch.split, [idx[j] for j in sel])
        out[sel] = branch.logits(h, ctx).float().cpu()
    branch.eval()
    return out


def _cap(rows: List[Example], cap: Optional[int], rng: np.random.Generator) -> List[Example]:
    if cap and len(rows) > cap:
        idx = sorted(rng.choice(len(rows), cap, replace=False).tolist())
        rows = [rows[i] for i in idx]
    return rows


def run_task(task: str, trunk: Trunk, ds, split: int, labelled_n: int, rounds: int, admit_frac: float,
            epochs: int, min_steps: int, seed: int, cap_pool: Optional[int], cap_val: Optional[int],
            cap_test: Optional[int], log=print) -> Dict:
    rng = np.random.default_rng(seed)
    pool = _cap([e for e in ds.train if task in e.y], cap_pool, rng)
    val = _cap([e for e in ds.val if task in e.y], cap_val, rng)
    test = _cap([e for e in ds.test if task in e.y], cap_test, rng)
    labels = ds.tasks[task].labels
    qtype_val = infer_qtype(labels)
    labelled_n = min(labelled_n, max(2, len(pool) // 2))
    log(f"[{task}] pool={len(pool)} (labelled={labelled_n}, unlabelled={len(pool) - labelled_n}) "
        f"val={len(val)} test={len(test)} labels={labels}")

    texts = [e.text for e in pool] + [e.text for e in val] + [e.text for e in test]
    cache = FeatureCache(trunk, texts, [split], ds.max_len)
    n_pool, n_val = len(pool), len(val)
    val_idx = list(range(n_pool, n_pool + n_val))
    test_idx = list(range(n_pool + n_val, len(texts)))
    y_val = torch.tensor([e.y[task] for e in val])
    y_test = torch.tensor([e.y[task] for e in test])
    soft_test = torch.tensor(np.stack([e.soft[task] for e in test]))

    def eval_branch(branch) -> Dict:
        z = predict_logits(branch, cache, test_idx)
        probs = torch.softmax(z, -1).numpy()
        return evaluate(probs, y_test.numpy(), soft_test.numpy())

    def fit(train_list: List[int], y_map, soft_map) -> ProbeBranch:
        y = torch.tensor([y_map(i) for i in train_list])
        soft = torch.stack([soft_map(i) for i in train_list])
        branch = ProbeBranch(split, labels, trunk.hidden)
        train_branch(branch, cache, train_list, y, soft, val_idx, y_val, epochs=epochs, seed=seed,
                    min_steps=min_steps)
        return branch

    order = list(range(n_pool))
    rng.shuffle(order)
    labelled_idx = order[:labelled_n]
    unlabelled_idx = set(order[labelled_n:])

    branch = fit(labelled_idx, lambda i: pool[i].y[task], lambda i: torch.tensor(pool[i].soft[task]))
    baseline_eval = eval_branch(branch)
    log(f"  baseline (labelled-only, n={labelled_n}): acc={baseline_eval['acc']:.3f} "
        f"f1={baseline_eval['macro_f1']:.3f}")

    oracle_branch = fit(list(range(n_pool)), lambda i: pool[i].y[task], lambda i: torch.tensor(pool[i].soft[task]))
    oracle_eval = eval_branch(oracle_branch)
    log(f"  oracle (full pool, TRUE labels, n={n_pool}): acc={oracle_eval['acc']:.3f} "
        f"f1={oracle_eval['macro_f1']:.3f}")

    pseudo: Dict[int, np.ndarray] = {}
    remaining = set(unlabelled_idx)
    curve = [{"round": 0, "n_labelled": labelled_n, "n_pseudo": 0, "pseudo_label_acc_vs_hidden_gold": None,
             **baseline_eval}]
    for r in range(1, rounds + 1):
        cand = sorted(remaining)
        if not cand:
            break
        p1 = torch.softmax(mc_forward(branch, cache, cand), -1)
        p2 = torch.softmax(mc_forward(branch, cache, cand), -1)
        qtype = torch.full((len(cand),), qtype_val)
        mask = torch.ones_like(p1)
        agree = 0.5 * (proper_reward(p1, p2, qtype, mask) + proper_reward(p2, p1, qtype, mask))
        k = max(1, int(round(len(cand) * admit_frac)))
        top = torch.topk(agree, min(k, len(cand))).indices.tolist()
        admitted = [cand[i] for i in top]
        correct = []
        for i, ti in zip(admitted, top):
            soft_i = ((p1[ti] + p2[ti]) / 2).numpy()
            pseudo[i] = soft_i
            remaining.discard(i)
            correct.append(int(np.argmax(soft_i)) == pool[i].y[task])
        pseudo_acc = float(np.mean(correct)) if correct else None

        train_list = labelled_idx + sorted(pseudo.keys())
        branch = fit(train_list,
                    lambda i: pool[i].y[task] if i in set(labelled_idx) else int(np.argmax(pseudo[i])),
                    lambda i: torch.tensor(pool[i].soft[task]) if i in set(labelled_idx)
                    else torch.tensor(pseudo[i]))
        ev = eval_branch(branch)
        curve.append({"round": r, "n_labelled": labelled_n, "n_pseudo": len(pseudo),
                     "pseudo_label_acc_vs_hidden_gold": pseudo_acc, **ev})
        log(f"  round {r}: +{len(admitted)} pseudo (agreement-admitted; acc-vs-hidden-gold "
            f"{pseudo_acc if pseudo_acc is None else f'{pseudo_acc:.2f}'}), total pseudo={len(pseudo)} "
            f"-> test acc={ev['acc']:.3f} f1={ev['macro_f1']:.3f}")

    return {"labels": labels, "n_pool": n_pool, "n_test": len(test), "baseline": baseline_eval,
           "oracle_full_pool_true_labels": oracle_eval, "curve": curve}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", type=int, default=8)
    ap.add_argument("--labelled-n", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--admit-frac", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tasks", default=None)
    args = ap.parse_args()

    if args.smoke:
        device = "cpu"
        tasks = [args.tasks] if args.tasks else [DEFAULT_TASK_SMOKE]
        labelled_n, rounds, admit_frac, epochs, min_steps = 8, 2, 0.34, 1, 8
        cap_pool, cap_val, cap_test = 24, 8, 8
    else:
        device = args.device
        tasks = args.tasks.split(",") if args.tasks else DEFAULT_TASKS_FULL
        labelled_n, rounds, admit_frac, epochs, min_steps = args.labelled_n, args.rounds, args.admit_frac, \
            args.epochs, 300
        cap_pool, cap_val, cap_test = None, None, None

    t_start = time.time()
    trunk = Trunk(device=device)
    ds = load_typed_decisions(seed=args.seed)
    print(f"loaded {ds.summary()}")

    results = {}
    for task in tasks:
        results[task] = run_task(task, trunk, ds, args.split, labelled_n, rounds, admit_frac, epochs,
                                 min_steps, args.seed, cap_pool, cap_val, cap_test)

    print("\n=== self-training summary ===")
    for task, r in results.items():
        last = r["curve"][-1]
        print(f"[{task}] baseline acc={r['baseline']['acc']:.3f} -> self-trained acc={last['acc']:.3f} "
              f"(n_pseudo={last['n_pseudo']})  vs oracle (full pool, true labels) acc="
              f"{r['oracle_full_pool_true_labels']['acc']:.3f}")

    out_data = {"config": vars(args), "results": results, "elapsed_s": round(time.time() - t_start, 1)}
    out = args.out or (None if args.smoke else "results/tarski/explore_self_training.json")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(out_data, f, indent=1)
        print(f"wrote {out}")
    print(f"total elapsed: {out_data['elapsed_s']}s")


if __name__ == "__main__":
    main()
