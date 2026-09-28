"""Exploration H: add a new route to an already-trained branch without forgetting the old ones.

Today, adding a new label to a trained decision (a new team, a new category) means retraining that
branch's whole output head, risking regressing the routes it already handles. This script tests weight
imprinting (Qi, Brown & Lowe, "Low-Shot Learning with Imprinted Weights", CVPR 2018:
https://openaccess.thecvf.com/content_cvpr_2018/html/Qi_Low-Shot_Learning_With_CVPR_2018_paper.html) on a
`ProbeBranch`'s output layer: hide one CLINC150 domain entirely during initial training (the branch's
head still has all 11 rows, but the held-out row never sees a gradient), then set that row directly from
data -- the mean pooled+normed feature vector of a handful of held-out-domain examples, L2-normalised and
rescaled to match the other rows' typical norm -- with NO fine-tuning at all. Then compare a "surgical"
short fine-tune (only the new row's weights get gradient, everything else -- including the shared
LayerNorm -- is frozen) against a "naive" fine-tune (the whole head retrained on the same small replay
set, no freezing) and a from-scratch ceiling (all 11 domains, full training data).

Metrics reported for every configuration: accuracy on the 10 OLD domains (retention -- does the surgical
edit forget less than the naive one?) and on the NEW domain (how good is imprinting's few-shot answer
before any fine-tuning at all, and after the short one?).

Novelty note (see research_notes/explore/learning.md, idea H): weight imprinting is established for
few-shot class-incremental vision classifiers. Applying it to a frozen-trunk text ROUTING branch, framed
explicitly as "add a route without forgetting" for a local decision-routing system, and measuring
retention against a naive full-head fine-tune at the same replay budget, is a reasonable but incremental
combination -- the mechanism itself is not new, only the setting and the retention framing.

Usage:
  .venv/bin/python explore/learning_route_imprinting.py --smoke
  .venv/bin/python explore/learning_route_imprinting.py --out results/tarski/explore_route_imprinting.json
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
from tarski.data import load_clinc
from tarski.train import FeatureCache, evaluate, predict_logits, train_branch
from tarski.trunk import Trunk


def scheduled_epochs(n_train: int, bs: int, epochs: int, min_steps: int) -> int:
    per_epoch = max(1, (n_train + bs - 1) // bs)
    return max(epochs, -(-min_steps // per_epoch))


@torch.no_grad()
def pooled_features(branch: ProbeBranch, cache: FeatureCache, idx: List[int]) -> torch.Tensor:
    """The exact pre-linear representation `ProbeBranch.logits` computes: LayerNorm -> mean pool."""
    branch.eval()
    h, ctx = cache.batch(branch.split, idx)
    return mean_pool(branch.norm(h), ctx.attention_mask).cpu()


def imprint_row(branch: ProbeBranch, cache: FeatureCache, idx: List[int], class_idx: int) -> None:
    feats = pooled_features(branch, cache, idx)
    v = feats.mean(0)
    v = v / v.norm().clamp_min(1e-6)
    other_norms = branch.out.weight.data.norm(dim=1)
    scale = float(other_norms.mean()) if other_norms.numel() else 1.0
    with torch.no_grad():
        branch.out.weight.data[class_idx] = (v * scale).to(branch.out.weight.device)
        branch.out.bias.data[class_idx] = float(branch.out.bias.data.mean())


def fine_tune_new_row_only(branch: ProbeBranch, cache: FeatureCache, train_idx: List[int],
                           y_train: torch.Tensor, val_idx: List[int], y_val: torch.Tensor,
                           class_idx: int, epochs: int, bs: int, lr: float, seed: int,
                           min_steps: int = 300, min_val: int = 50, patience: int = 3) -> Dict:
    """Surgical edit: freeze the shared LayerNorm and every output row except `class_idx`, so retention
    on old routes can only be hurt by the (small) shared-representation drift this loop does NOT allow --
    only the new row moves. Mirrors train_branch's min-steps/last-epoch schedule (see
    learning_proper_score_objectives.py's `train_with_objective` for why)."""
    for p in branch.norm.parameters():
        p.requires_grad_(False)
    epochs = scheduled_epochs(len(train_idx), bs, epochs, min_steps)
    select = len(val_idx) >= min_val
    min_epochs = max(1, (epochs + 1) // 2)
    torch.manual_seed(seed)
    dev = cache.trunk.device
    branch.to(dev).train()
    opt = torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    K = branch.n_labels
    row_mask = torch.zeros(K, dtype=torch.bool)
    row_mask[class_idx] = True
    rng = np.random.default_rng(seed)
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
            opt.zero_grad(set_to_none=True)
            loss.backward()
            branch.out.weight.grad[~row_mask] = 0
            branch.out.bias.grad[~row_mask] = 0
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
    for p in branch.norm.parameters():
        p.requires_grad_(True)
    branch.eval()
    return {"epochs_run": ep + 1, "epochs_scheduled": epochs}


def eval_split(branch: ProbeBranch, cache: FeatureCache, test_idx: List[int], y_test: np.ndarray,
              held_idx: int) -> Dict:
    z = predict_logits(branch, cache, test_idx)
    pred = z.argmax(-1).numpy()
    old_mask, new_mask = y_test != held_idx, y_test == held_idx
    return {"old_domain_acc": float((pred[old_mask] == y_test[old_mask]).mean()) if old_mask.any() else None,
           "new_domain_acc": float((pred[new_mask] == y_test[new_mask]).mean()) if new_mask.any() else None,
           "overall_acc": float((pred == y_test).mean())}


def run(split: int, k_shots: int, replay_per_domain: int, epochs: int, bs: int, lr: float, seed: int,
       device: str, cap_train: Optional[int], cap_val: Optional[int], cap_test: Optional[int],
       min_steps: int, log=print) -> Dict:
    t0 = time.time()
    trunk = Trunk(device=device)
    ds = load_clinc(seed=seed)
    log(f"loaded {ds.summary()}")
    labels = ds.tasks["domain"].labels
    held_idx = 0
    held_name = labels[held_idx]
    log(f"held-out (new) domain: {held_name!r}; {len(labels) - 1} old domains")

    rng = np.random.default_rng(seed)

    def cap(rows, n):
        if n and len(rows) > n:
            idx = sorted(rng.choice(len(rows), n, replace=False).tolist())
            return [rows[i] for i in idx]
        return rows

    def cap_stratified(rows, n):
        """Cap total count to `n` but guarantee the held-out domain keeps at least `k_shots` rows,
        so a small smoke-sized cap can't silently draw zero held-domain examples by chance."""
        held = [e for e in rows if e.y["domain"] == held_idx]
        old = [e for e in rows if e.y["domain"] != held_idx]
        held_keep = max(k_shots, min(len(held), (n or len(rows)) // len(labels) * 2)) if n else len(held)
        held = cap(held, held_keep)
        old = cap(old, max(0, (n - len(held))) if n else None)
        return old + held

    train_all = cap_stratified(ds.train, cap_train)
    val_all = cap(ds.val, cap_val)
    test_all = cap_stratified(ds.test, cap_test)
    texts = [e.text for e in train_all] + [e.text for e in val_all] + [e.text for e in test_all]
    cache = FeatureCache(trunk, texts, [split], ds.max_len)
    n_tr, n_va = len(train_all), len(val_all)
    val_idx_all = list(range(n_tr, n_tr + n_va))
    test_idx_all = list(range(n_tr + n_va, len(texts)))
    y_val_all = torch.tensor([e.y["domain"] for e in val_all])
    y_test_all = np.array([e.y["domain"] for e in test_all])
    log(f"cached trunk depth {split} for {len(texts)} messages in {cache.seconds:.1f}s")

    # --- stage 0: train on the 10 OLD domains only (held domain's rows in `out` never see a gradient) ---
    old_train_idx = [i for i in range(n_tr) if train_all[i].y["domain"] != held_idx]
    y_old = torch.tensor([train_all[i].y["domain"] for i in old_train_idx])
    branch = ProbeBranch(split, labels, trunk.hidden)
    train_branch(branch, cache, old_train_idx, y_old, None, val_idx_all, y_val_all, epochs=epochs,
                seed=seed, min_steps=min_steps)
    before = eval_split(branch, cache, test_idx_all, y_test_all, held_idx)
    log(f"before imprinting: old_domain_acc={before['old_domain_acc']:.3f} "
        f"new_domain_acc={before['new_domain_acc']:.3f}")

    # --- imprint the new row from k examples, no fine-tuning at all ---
    new_domain_train_idx = [i for i in range(n_tr) if train_all[i].y["domain"] == held_idx]
    shot_idx = new_domain_train_idx[:k_shots]
    imprint_row(branch, cache, shot_idx, held_idx)
    after_imprint = eval_split(branch, cache, test_idx_all, y_test_all, held_idx)
    log(f"after imprinting (0 fine-tune steps, k={len(shot_idx)}): "
        f"old_domain_acc={after_imprint['old_domain_acc']:.3f} "
        f"new_domain_acc={after_imprint['new_domain_acc']:.3f}")

    # small mixed replay set for both fine-tune variants: the k new-domain shots + a few old-domain
    # examples per domain, so a naive fine-tune has SOME chance to avoid forgetting too
    replay_old = []
    by_domain: Dict[int, List[int]] = {}
    for i in old_train_idx:
        by_domain.setdefault(train_all[i].y["domain"], []).append(i)
    for d, idxs in by_domain.items():
        replay_old.extend(idxs[:replay_per_domain])
    replay_idx = shot_idx + replay_old
    y_replay = torch.tensor([train_all[i].y["domain"] for i in replay_idx])

    # --- surgical fine-tune: only the new row moves ---
    import copy
    branch_surgical = copy.deepcopy(branch)
    fine_tune_new_row_only(branch_surgical, cache, replay_idx, y_replay, val_idx_all, y_val_all,
                          held_idx, epochs, bs, lr, seed, min_steps=min_steps)
    after_surgical = eval_split(branch_surgical, cache, test_idx_all, y_test_all, held_idx)
    log(f"after surgical fine-tune (new row only, n_replay={len(replay_idx)}): "
        f"old_domain_acc={after_surgical['old_domain_acc']:.3f} "
        f"new_domain_acc={after_surgical['new_domain_acc']:.3f}")

    # --- naive fine-tune: the whole head retrained on the same tiny replay set ---
    branch_naive = copy.deepcopy(branch)
    train_branch(branch_naive, cache, replay_idx, y_replay, None, val_idx_all, y_val_all, epochs=epochs,
                seed=seed, min_steps=min_steps)
    after_naive = eval_split(branch_naive, cache, test_idx_all, y_test_all, held_idx)
    log(f"after naive fine-tune (whole head, n_replay={len(replay_idx)}): "
        f"old_domain_acc={after_naive['old_domain_acc']:.3f} "
        f"new_domain_acc={after_naive['new_domain_acc']:.3f}")

    # --- ceiling: from-scratch retrain on ALL 11 domains' full training data ---
    y_all = torch.tensor([e.y["domain"] for e in train_all])
    branch_ceiling = ProbeBranch(split, labels, trunk.hidden)
    train_branch(branch_ceiling, cache, list(range(n_tr)), y_all, None, val_idx_all, y_val_all,
                epochs=epochs, seed=seed, min_steps=min_steps)
    ceiling = eval_split(branch_ceiling, cache, test_idx_all, y_test_all, held_idx)
    log(f"ceiling (from-scratch, all 11 domains, n={n_tr}): old_domain_acc={ceiling['old_domain_acc']:.3f} "
        f"new_domain_acc={ceiling['new_domain_acc']:.3f}")

    return {"held_out_domain": held_name, "n_shots": len(shot_idx), "n_replay": len(replay_idx),
           "before_imprinting": before, "after_imprinting_zero_shot_finetune": after_imprint,
           "after_surgical_finetune": after_surgical, "after_naive_finetune": after_naive,
           "ceiling_full_retrain": ceiling, "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--k-shots", type=int, default=10)
    ap.add_argument("--replay-per-domain", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.smoke:
        device = "cpu"
        cap_train, cap_val, cap_test = 60, 20, 30
        k_shots, replay_per_domain, min_steps = 3, 1, 8
    else:
        device = args.device
        cap_train, cap_val, cap_test = 3000, 500, 1500
        k_shots, replay_per_domain, min_steps = args.k_shots, args.replay_per_domain, 300

    result = run(args.split, k_shots, replay_per_domain, args.epochs, args.bs, args.lr, args.seed,
                device, cap_train, cap_val, cap_test, min_steps)

    print("\n=== route-imprinting summary ===")
    for name in ["before_imprinting", "after_imprinting_zero_shot_finetune", "after_surgical_finetune",
                "after_naive_finetune", "ceiling_full_retrain"]:
        m = result[name]
        print(f"  {name:35s} old={m['old_domain_acc']:.3f}  new={m['new_domain_acc']:.3f}")

    out = args.out or (None if args.smoke else "results/tarski/explore_route_imprinting.json")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {out}")
    print(f"total elapsed: {result['elapsed_s']}s")


if __name__ == "__main__":
    main()
