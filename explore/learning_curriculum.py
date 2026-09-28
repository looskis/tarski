"""Exploration I: order branch training by a cheap, SHALLOWER probe's per-example difficulty.

`tarski/autosplit.py` already computes a layer-wise linear-probe accuracy curve per task, for free, to
pick each task's split depth. This script reuses the same idea at the level of individual examples: fit a
quick logistic-regression probe on much SHALLOWER cached features (depth `shallow_depth`, e.g. 3) than the
branch's own split depth, and use each training example's margin (top-1 minus top-2 probability under
that shallow probe) as a difficulty score. Train the branch with a "baby steps" curriculum (Bengio et al.,
2009): stage 1 sees only the easiest 25% of the training pool, stage 2 the easiest 50%, and so on to 100%
by the final stage, with the SAME total optimiser-step budget (`tarski.train.train_branch`'s own
min-steps schedule) as a plain random-order baseline, so the only thing that differs is which examples are
available early in training.

Because the coordinator's fix (train_branch now forces >=300 steps) already turned typed-decisions'
"collapse" into an under-training story rather than a hard optimisation problem, this experiment reports
mostly on CONVERGENCE SPEED (the validation-accuracy-per-epoch curve) rather than expecting a different
final accuracy -- if the underlying issue really was step count, a curriculum should not change the
ceiling, only (maybe) how many of those steps are needed to reach it.

Novelty note (see docs/research/notes/explore/learning.md, idea I): curriculum learning by difficulty is
decades old; using a shallow-depth probe on the SAME frozen trunk the eventual (deeper) branch will also
read, purely to order examples for that branch's training, is the specific combination we did not find in
prior curriculum-learning-for-transformers work (which mostly uses external difficulty proxies or the
model being trained itself, not a cheaper probe on an earlier layer of the very same frozen backbone).

Usage:
  .venv/bin/python explore/learning_curriculum.py --smoke
  .venv/bin/python explore/learning_curriculum.py --out results/tarski/explore_curriculum.json
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

from tarski.branches import ProbeBranch, mean_pool
from tarski.data import Example, load_typed_decisions
from tarski.train import FeatureCache, evaluate, predict_logits, train_branch
from tarski.trunk import Trunk

DEFAULT_TASK = "customer_service.action"


def scheduled_epochs(n_train: int, bs: int, epochs: int, min_steps: int) -> int:
    per_epoch = max(1, (n_train + bs - 1) // bs)
    return max(epochs, -(-min_steps // per_epoch))


def difficulty_margins(cache: FeatureCache, shallow_depth: int, idx: List[int], y: torch.Tensor,
                       n_labels: int, device, steps: int = 150, lr: float = 1e-2) -> np.ndarray:
    """A quick, deliberately cheap in-sample logistic probe at a SHALLOW depth (the same style as
    `tarski.autosplit.probe_accuracy`, but scored on the training set itself, since we only need a
    difficulty ranking, not a held-out estimate). Margin = P(top-1) - P(top-2): confident and (usually,
    though not necessarily) correct at a shallow depth is the "easy" end."""
    h, ctx = cache.batch(shallow_depth, idx)
    x = mean_pool(h, ctx.attention_mask).detach()
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-4)
    xn = (x - mu) / sd
    lin = torch.nn.Linear(xn.shape[1], n_labels).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-4)
    yt = y.to(device)
    for _ in range(steps):
        loss = F.cross_entropy(lin(xn), yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.softmax(lin(xn), -1)
        if n_labels > 1:
            top2 = p.topk(2, -1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = p[:, 0]
    return margin.cpu().numpy()


def train_curriculum(branch: ProbeBranch, cache: FeatureCache, order_easy_first: List[int],
                     y_map: Dict[int, int], soft_map: Dict[int, torch.Tensor], val_idx: List[int],
                     y_val: torch.Tensor, epochs: int, bs: int, lr: float, seed: int, stages: int = 4,
                     min_steps: int = 300, min_val: int = 50, patience: int = 3) -> Dict:
    n = len(order_easy_first)
    epochs = scheduled_epochs(n, bs, epochs, min_steps)
    select = len(val_idx) >= min_val
    min_epochs = max(1, (epochs + 1) // 2)
    torch.manual_seed(seed)
    dev = cache.trunk.device
    branch.to(dev).train()
    opt = torch.optim.AdamW(branch.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    fracs = np.linspace(1.0 / stages, 1.0, stages)
    best, best_state, bad, hist, ep = -1.0, None, 0, [], 0
    for ep in range(epochs):
        stage = min(int(ep / max(1, epochs) * stages), stages - 1)
        m = max(bs, int(round(n * fracs[stage])))
        avail = order_easy_first[:m]
        branch.train()
        local = list(range(len(avail)))
        rng.shuffle(local)
        for s in range(0, len(local), bs):
            sel = [avail[j] for j in local[s:s + bs]]
            h, ctx = cache.batch(branch.split, sel)
            z = branch.logits(h, ctx).float()
            soft = torch.stack([soft_map[i] for i in sel]).to(dev)
            loss = -(soft * F.log_softmax(z, -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
            opt.step()
        va = float((predict_logits(branch, cache, val_idx).argmax(-1) == y_val).float().mean())
        hist.append({"epoch": ep + 1, "stage_frac": float(fracs[stage]), "val_acc": va})
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
    return {"val_acc": best, "history": hist, "epochs_run": ep + 1, "epochs_scheduled": epochs}


def epochs_to_90pct_of_best(history: List[Dict]) -> Optional[int]:
    if not history:
        return None
    best = max(h["val_acc"] for h in history)
    if best <= 0:
        return None
    for h in history:
        if h["val_acc"] >= 0.9 * best:
            return h["epoch"]
    return None


def _cap(rows: List[Example], cap: Optional[int], rng: np.random.Generator) -> List[Example]:
    if cap and len(rows) > cap:
        idx = sorted(rng.choice(len(rows), cap, replace=False).tolist())
        rows = [rows[i] for i in idx]
    return rows


def run_task(task: str, trunk: Trunk, ds, shallow_depth: int, split: int, epochs: int, bs: int, lr: float,
            seed: int, min_steps: int, cap_train: Optional[int], cap_val: Optional[int],
            cap_test: Optional[int], log=print) -> Dict:
    rng = np.random.default_rng(seed)
    train = _cap([e for e in ds.train if task in e.y], cap_train, rng)
    val = _cap([e for e in ds.val if task in e.y], cap_val, rng)
    test = _cap([e for e in ds.test if task in e.y], cap_test, rng)
    labels = ds.tasks[task].labels
    log(f"[{task}] train={len(train)} val={len(val)} test={len(test)} labels={labels}")

    texts = [e.text for e in train] + [e.text for e in val] + [e.text for e in test]
    cache = FeatureCache(trunk, texts, sorted({shallow_depth, split}), ds.max_len)
    n_tr, n_va = len(train), len(val)
    train_idx = list(range(n_tr))
    val_idx = list(range(n_tr, n_tr + n_va))
    test_idx = list(range(n_tr + n_va, len(texts)))
    y_train = torch.tensor([e.y[task] for e in train])
    soft_train = torch.tensor(np.stack([e.soft[task] for e in train]))
    y_val = torch.tensor([e.y[task] for e in val])
    y_test = torch.tensor([e.y[task] for e in test])
    soft_test = torch.tensor(np.stack([e.soft[task] for e in test]))
    log(f"  cached trunk depths [{shallow_depth}, {split}] for {len(texts)} messages in {cache.seconds:.1f}s")

    margins = difficulty_margins(cache, shallow_depth, train_idx, y_train, len(labels), trunk.device)
    order_easy_first = [i for i, _ in sorted(zip(train_idx, margins), key=lambda p: -p[1])]
    y_map = {i: int(y_train[i]) for i in train_idx}
    soft_map = {i: soft_train[i] for i in train_idx}

    def eval_branch(branch) -> Dict:
        z = predict_logits(branch, cache, test_idx)
        probs = torch.softmax(z, -1).numpy()
        return evaluate(probs, y_test.numpy(), soft_test.numpy())

    baseline = ProbeBranch(split, labels, trunk.hidden)
    b_info = train_branch(baseline, cache, train_idx, y_train, soft_train, val_idx, y_val, epochs=epochs,
                          bs=bs, seed=seed, min_steps=min_steps)
    base_eval = eval_branch(baseline)
    log(f"  random-order baseline: acc={base_eval['acc']:.3f} f1={base_eval['macro_f1']:.3f} "
        f"epochs={len(b_info['history'])} 90%-of-best@epoch={epochs_to_90pct_of_best(b_info['history'])}")

    curric = ProbeBranch(split, labels, trunk.hidden)
    c_info = train_curriculum(curric, cache, order_easy_first, y_map, soft_map, val_idx, y_val,
                              epochs=epochs, bs=bs, lr=1e-3, seed=seed, min_steps=min_steps)
    curric_eval = eval_branch(curric)
    log(f"  easy-to-hard curriculum: acc={curric_eval['acc']:.3f} f1={curric_eval['macro_f1']:.3f} "
        f"epochs={c_info['epochs_run']} 90%-of-best@epoch={epochs_to_90pct_of_best(c_info['history'])}")

    return {"labels": labels, "n_train": n_tr, "n_test": len(test),
           "random_order": {"final": base_eval, "history": b_info["history"],
                            "epoch_to_90pct_of_best": epochs_to_90pct_of_best(b_info["history"])},
           "curriculum": {"final": curric_eval, "history": c_info["history"],
                         "epoch_to_90pct_of_best": epochs_to_90pct_of_best(c_info["history"])}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shallow-depth", type=int, default=3)
    ap.add_argument("--split", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tasks", default=None)
    args = ap.parse_args()

    if args.smoke:
        device = "cpu"
        tasks = [args.tasks] if args.tasks else [DEFAULT_TASK]
        cap_train, cap_val, cap_test, min_steps = 24, 10, 10, 8
    else:
        device = args.device
        tasks = args.tasks.split(",") if args.tasks else [DEFAULT_TASK, "invoice_processing.urgency"]
        cap_train, cap_val, cap_test, min_steps = None, None, None, 300

    t_start = time.time()
    trunk = Trunk(device=device)
    ds = load_typed_decisions(seed=args.seed)
    print(f"loaded {ds.summary()}")

    results = {}
    for task in tasks:
        results[task] = run_task(task, trunk, ds, args.shallow_depth, args.split, args.epochs, args.bs,
                                 args.lr, args.seed, min_steps, cap_train, cap_val, cap_test)

    print("\n=== curriculum summary ===")
    for task, r in results.items():
        ro, cu = r["random_order"], r["curriculum"]
        print(f"[{task}] random acc={ro['final']['acc']:.3f} (90%@ep{ro['epoch_to_90pct_of_best']})  "
              f"curriculum acc={cu['final']['acc']:.3f} (90%@ep{cu['epoch_to_90pct_of_best']})")

    out_data = {"config": vars(args), "results": results, "elapsed_s": round(time.time() - t_start, 1)}
    out = args.out or (None if args.smoke else "results/tarski/explore_curriculum.json")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(out_data, f, indent=1)
        print(f"wrote {out}")
    print(f"total elapsed: {out_data['elapsed_s']}s")


if __name__ == "__main__":
    main()
