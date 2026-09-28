"""Exploration: how many labels does a user actually need, if the "teacher" is free to query?

`laya` (the question-in-input ModernBERT-large cross-encoder tarski is meant to replace at inference
time) is installed locally and its `typed-decisions` checkpoint is cached
(`convaiinnovations/laya`, subfolder="typed-decisions"). That means it can also serve as a live
DISTILLATION TEACHER while a tarski branch is being trained: instead of paying for human labels, query
laya for a soft label on any unlabelled message a branch is uncertain about, at train time only. Once
the branch is trained it runs alone (laya is never in the serving path), so this only spends teacher
compute, never inference latency.

This script builds label-efficiency curves for one of the typed-decisions tasks that
`docs/research/notes/explore/BRIEF.md` reports as collapsing (probes stuck near the majority class), under
three label-acquisition strategies at matched label budgets:

  random_soft   label random unlabelled pool messages with laya's live soft distribution.
  active_soft   label the pool messages the CURRENT branch is most uncertain about (predictive entropy)
                with laya's live soft distribution -- uncertainty sampling (Lewis & Gale, SIGIR 1994),
                applied to which examples get a distilled soft label rather than which get a human label.
  random_hard   the SAME random draw as random_soft (paired, so the only difference is the label itself),
                but using the dataset's own one-hot gold label instead of a teacher query.

`random_soft` vs `random_hard` isolates the value of a soft label over a hard one at equal data
(distillation's classic benefit: Hinton et al. 2015; Menon et al., "A Statistical Perspective on
Distillation", ICML 2021, gives soft labels a variance-reduction / sample-complexity argument). Active
vs random isolates the value of choosing WHICH examples to spend teacher queries on.

Novelty note (see docs/research/notes/explore/learning.md): distilling a per-question cross-encoder into a
shared-trunk branch is implicit in how the typed-decisions dataset itself was built (its "soft" column
already came from a teacher). What we could not find prior work on is using the teacher live, in an
active-learning loop, to decide which unlabelled messages are worth a soft label at all -- i.e. active
learning as a label-BUDGET optimiser for knowledge distillation into a much smaller, differently
-shaped student (a frozen-trunk branch, not a distilled copy of the same architecture). "Diversity
Enhanced Active Learning with Strictly Proper Scoring Rules" (Wu et al., 2021) uses proper scores to
pick which examples a HUMAN should label; here the "oracle" is itself a model, queried at train time.

Usage:
  .venv/bin/python explore/learning_active_distillation.py --smoke
  .venv/bin/python explore/learning_active_distillation.py --out results/tarski/explore_active_distillation.json
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
from tarski.data import Dataset, Example, load_typed_decisions
from tarski.train import FeatureCache, evaluate, predict_logits, train_branch
from tarski.trunk import Trunk

import laya

DEFAULT_FULL_TASKS = ["customer_service.action", "customer_service.category", "customer_service.churn_risk",
                     "customer_service.needs_human", "customer_service.urgency"]
DEFAULT_SMOKE_TASK = "agent_trace_observability.action"

_QUESTION_CACHE: Dict[str, Dict] = {}


def get_question_def(task: str) -> Dict:
    """typed-decisions' HF rows carry the question schema laya needs (type/instructions/criteria) in
    their `questions` column. tarski's own `Dataset`/`Example` only keep the resolved label set, so we
    pull the raw schema once per workflow straight from the (already locally cached) HF dataset."""
    workflow, qid = task.split(".", 1)
    if workflow not in _QUESTION_CACHE:
        from datasets import load_dataset
        raw = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
        for r in raw:
            if r["workflow"] == workflow:
                _QUESTION_CACHE[workflow] = json.loads(r["questions"])
                break
    return _QUESTION_CACHE[workflow][qid]


def query_teacher(agent, texts: List[str], task: str, labels: List[str], batch_size: Optional[int] = None,
                  max_len: Optional[int] = None) -> np.ndarray:
    """Live soft labels from the laya teacher for a batch of (unlabelled) JSON states."""
    if not texts:
        return np.zeros((0, len(labels)), dtype=np.float32)
    workflow, qid = task.split(".", 1)
    q = get_question_def(task)
    states = [json.loads(t) for t in texts]
    results = agent.predict_batch(states, {qid: q}, batch_size=batch_size, max_len=max_len)
    out = np.zeros((len(texts), len(labels)), dtype=np.float32)
    for i, r in enumerate(results):
        ans = r["answers"][qid]
        if ans["type"] == "noul":
            pt = float(ans["noul"])
            probs = {"false": 1.0 - pt, "true": pt}
        else:
            probs = ans["probabilities"]
        out[i] = [probs[k] for k in labels]
    return out


@torch.no_grad()
def entropy_scores(branch: ProbeBranch, cache: FeatureCache, idx: List[int]) -> np.ndarray:
    if not idx:
        return np.zeros(0)
    logits = predict_logits(branch, cache, idx)
    p = torch.softmax(logits, -1).clamp_min(1e-9)
    return (-(p * p.log()).sum(-1)).numpy()


def sample_capped(rows: List[Example], cap: Optional[int], rng: np.random.Generator) -> List[Example]:
    if cap and len(rows) > cap:
        idx = sorted(rng.choice(len(rows), cap, replace=False).tolist())
        rows = [rows[i] for i in idx]
    return rows


def run_task(task: str, trunk: Trunk, agent, ds: Dataset, split: int, rounds: int, seed_n: int,
            k_per_round: int, pool_cap: Optional[int], val_cap: Optional[int], test_cap: Optional[int],
            epochs: int, seed: int, teacher_bs: Optional[int], log=print) -> Dict:
    rng = np.random.default_rng(seed)
    pool = sample_capped([e for e in ds.train if task in e.y], pool_cap, rng)
    val = sample_capped([e for e in ds.val if task in e.y], val_cap, rng)
    test = sample_capped([e for e in ds.test if task in e.y], test_cap, rng)
    labels = ds.tasks[task].labels
    n_labels = len(labels)
    log(f"[{task}] pool={len(pool)} val={len(val)} test={len(test)} labels={labels}")

    texts = [e.text for e in pool] + [e.text for e in val] + [e.text for e in test]
    t0 = time.time()
    cache = FeatureCache(trunk, texts, [split], ds.max_len)
    log(f"  cached trunk depth {split} for {len(texts)} messages in {cache.seconds:.1f}s")
    n_pool, n_val = len(pool), len(val)
    val_idx = list(range(n_pool, n_pool + n_val))
    test_idx = list(range(n_pool + n_val, len(texts)))
    y_val = torch.tensor([e.y[task] for e in val])
    y_test = torch.tensor([e.y[task] for e in test])
    soft_test = torch.tensor(np.stack([e.soft[task] for e in test]))

    t1 = time.time()
    teacher_probs = query_teacher(agent, [e.text for e in test], task, labels, batch_size=teacher_bs)
    teacher_acc = float((teacher_probs.argmax(-1) == y_test.numpy()).mean())
    log(f"  teacher test acc {teacher_acc:.3f} on {len(test)} held-out messages ({time.time() - t1:.1f}s)")

    seed_idx = sorted(rng.choice(n_pool, min(seed_n, n_pool), replace=False).tolist())
    t1 = time.time()
    seed_soft = query_teacher(agent, [pool[i].text for i in seed_idx], task, labels, batch_size=teacher_bs)
    log(f"  seed teacher query for {len(seed_idx)} messages ({time.time() - t1:.1f}s)")
    eye = np.eye(n_labels, dtype=np.float32)
    seed_hard = np.stack([eye[pool[i].y[task]] for i in seed_idx])

    methods = {
        "active_soft": dict(zip(seed_idx, seed_soft)),
        "random_soft": dict(zip(seed_idx, seed_soft)),
        "random_hard": dict(zip(seed_idx, seed_hard)),
    }
    remaining = {m: set(range(n_pool)) - set(seed_idx) for m in methods}
    curves: Dict[str, List[Dict]] = {m: [] for m in methods}
    active_branch: Optional[ProbeBranch] = None

    def train_and_eval(labeled: Dict[int, np.ndarray]) -> tuple:
        idx = list(labeled.keys())
        y = torch.tensor([int(np.argmax(labeled[i])) for i in idx])
        soft = torch.tensor(np.stack([labeled[i] for i in idx]), dtype=torch.float32)
        branch = ProbeBranch(split, labels, trunk.hidden)
        train_branch(branch, cache, idx, y, soft, val_idx, y_val, epochs=epochs,
                    bs=min(32, max(4, len(idx))), seed=seed, log=lambda s: None)
        z = predict_logits(branch, cache, test_idx)
        probs = torch.softmax(z, -1).numpy()
        ev = evaluate(probs, y_test.numpy(), soft_test.numpy())
        return branch, ev

    for m, labeled in methods.items():
        branch, ev = train_and_eval(labeled)
        if m == "active_soft":
            active_branch = branch
        curves[m].append({"n_labels": len(labeled), **ev})
        log(f"  [{m}] round 0 n={len(labeled)} acc={ev['acc']:.3f} f1={ev['macro_f1']:.3f} "
            f"brier={ev['brier_soft']:.3f}")

    for r in range(1, rounds + 1):
        avail_random = list(remaining["random_soft"])
        take_r = min(k_per_round, len(avail_random))
        random_new = [avail_random[i] for i in
                      sorted(rng.choice(len(avail_random), take_r, replace=False).tolist())] if take_r else []

        avail_active = list(remaining["active_soft"])
        if avail_active and active_branch is not None:
            ent = entropy_scores(active_branch, cache, avail_active)
            top = np.argsort(-ent)[:min(k_per_round, len(avail_active))]
            active_new = [avail_active[i] for i in top]
        else:
            active_new = []

        if random_new:
            t1 = time.time()
            soft_new = query_teacher(agent, [pool[i].text for i in random_new], task, labels,
                                    batch_size=teacher_bs)
            log(f"  round {r}: queried teacher for {len(random_new)} random messages "
                f"({time.time() - t1:.1f}s)")
            for i, s in zip(random_new, soft_new):
                methods["random_soft"][i] = s
                methods["random_hard"][i] = eye[pool[i].y[task]]
                remaining["random_soft"].discard(i)
                remaining["random_hard"].discard(i)
        if active_new:
            t1 = time.time()
            soft_new = query_teacher(agent, [pool[i].text for i in active_new], task, labels,
                                    batch_size=teacher_bs)
            log(f"  round {r}: queried teacher for {len(active_new)} high-entropy messages "
                f"({time.time() - t1:.1f}s)")
            for i, s in zip(active_new, soft_new):
                methods["active_soft"][i] = s
                remaining["active_soft"].discard(i)

        for m, labeled in methods.items():
            branch, ev = train_and_eval(labeled)
            if m == "active_soft":
                active_branch = branch
            curves[m].append({"n_labels": len(labeled), **ev})
            log(f"  [{m}] round {r} n={len(labeled)} acc={ev['acc']:.3f} f1={ev['macro_f1']:.3f} "
                f"brier={ev['brier_soft']:.3f}")

    return {"labels": labels, "n_test": len(test), "teacher_test_acc": teacher_acc,
           "elapsed_s": round(time.time() - t0, 1), "curves": curves}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed-n", type=int, default=24)
    ap.add_argument("--k-per-round", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="trunk device; teacher uses the same unless --smoke")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--pool-cap", type=int, default=None, help="None = use every available pool row")
    ap.add_argument("--val-cap", type=int, default=None)
    ap.add_argument("--test-cap", type=int, default=None)
    args = ap.parse_args()

    if args.smoke:
        device = "cpu"
        tasks = [args.tasks] if args.tasks else [DEFAULT_SMOKE_TASK]
        rounds, seed_n, k_per_round, epochs = 1, 8, 8, 3
        pool_cap, val_cap, test_cap, teacher_bs = 40, 12, 16, 8
    else:
        device = args.device
        tasks = args.tasks.split(",") if args.tasks else DEFAULT_FULL_TASKS
        rounds, seed_n, k_per_round, epochs = args.rounds, args.seed_n, args.k_per_round, args.epochs
        # 10 rounds x 20/round + a 24-message seed reaches ~224 labels per task -- squarely in the "a
        # few hundred labelled Slack messages" range the coordinator flagged as the realistic user
        # budget, without exhausting a ~270-message-per-workflow pool. Uncapped by default so the curve
        # can also show what happens if a user keeps going to the full pool.
        pool_cap, val_cap, test_cap, teacher_bs = args.pool_cap, args.val_cap, args.test_cap, None

    t_start = time.time()
    trunk = Trunk(device=device)
    print(f"loading laya teacher (device={device or 'auto'}) ...")
    agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions", device=device)
    ds = load_typed_decisions(seed=args.seed)
    print(f"loaded {ds.summary()}")

    results = {}
    for task in tasks:
        results[task] = run_task(task, trunk, agent, ds, args.split, rounds, seed_n, k_per_round,
                                 pool_cap, val_cap, test_cap, epochs, args.seed, teacher_bs)

    print("\n=== label-efficiency summary ===")
    for task, r in results.items():
        print(f"[{task}] teacher test acc={r['teacher_test_acc']:.3f} (n_test={r['n_test']})")
        for m, curve in r["curves"].items():
            last = curve[-1]
            print(f"  {m:12s} {last['n_labels']:4d} labels -> acc={last['acc']:.3f} "
                  f"f1={last['macro_f1']:.3f} brier={last['brier_soft']:.3f}")

    out_data = {"config": vars(args), "results": results, "elapsed_s": round(time.time() - t_start, 1)}
    out = args.out or (None if args.smoke else "results/tarski/explore_active_distillation.json")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(out_data, f, indent=1)
        print(f"wrote {out}")
    print(f"total elapsed: {out_data['elapsed_s']}s")


if __name__ == "__main__":
    main()
