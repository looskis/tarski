"""Exploration: does the OBJECTIVE, not the architecture, explain why typed-decisions probes collapse?

`research_notes/explore/BRIEF.md` reports that on `typed-decisions`, probes predict the majority class
at every depth (mean acc 51-55% vs laya's full fine-tune at 76.6%), with soft teacher labels that are
"near-uniform". The default tarski objective is soft cross-entropy: `-(soft * log_softmax(z)).sum(-1)`.
Under near-uniform soft targets and imbalanced hard labels, that objective's gradient is dominated by the
majority class and a tiny linear head takes the cheapest way out.

This script trains the SAME frozen-trunk features with five objectives and two branch sizes, holding
everything else fixed, to see whether the objective alone fixes (or worsens) the collapse:

  ce         tarski's current objective: soft cross-entropy.
  cb_ce      class-balanced soft cross-entropy (Cui et al., CVPR 2019: reweight by the effective number
             of samples 1/(1-beta^n), computed from the soft-label mass per class).
  logit_adj  logit-adjusted softmax (Menon et al., ICLR 2021): add tau*log(prior) to the TRAINING logits
             only, so the loss stops rewarding the model for defaulting to the frequent class; evaluated
             on raw (unadjusted) logits.
  proper     laya's own strictly-proper reward (`laya.common.proper_reward`: log score + spherical score,
             + ranked-probability score for ordinal "score" questions) used directly as a differentiable
             loss (-reward), i.e. the "RLCD" objective without sampling.
  reinforce  the same proper_reward used as an actual REINFORCE policy-gradient target: sample a label
             from the branch's own categorical output, score that one-hot decision against the soft
             target with proper_reward, and push the sampled log-probability by the (baselined) reward.
             This is the "policy-gradient-on-proper-scores" reading of laya's objective, as opposed to
             `proper`'s direct/differentiable reading -- recent work explicitly distinguishes the two uses
             of the same scoring rule (see NOTES below); nobody seems to have tried the RL reading on a
             frozen-trunk branch this small.

Two branch sizes:
  probe  tarski's own `ProbeBranch` (LayerNorm + mean pool + linear): ~(hidden+1)*K params.
  bias   a new tiny branch: fixed random projection (hidden->K, untrained) after a parameter-free
         LayerNorm, plus a trainable per-class bias and one trainable scale: K+1 params. This is the
         "TinyLoRA" shape (tiny trainable vector through a frozen random tensor) applied to a routing
         head instead of a generation LoRA -- a routing decision needs log2(K) bits, so most of a
         standard linear probe's Hidden*K parameters may be spent overfitting spurious pooled-mean
         correlations rather than carrying task signal.

Prior art / novelty notes (see research_notes/explore/learning.md for citations):
  - Class-balanced loss, logit adjustment and proper scoring rules for classifiers are each established.
  - "Proper Scoring Rules for Agentic Uncertainty Quantification" (2026) argues explicitly that using a
    scoring rule as a *direct calibration loss* and using it as an *RL/policy-optimization reward* are NOT
    the same thing, even though they share a mathematical object. We could not find a prior empirical
    A/B of the two readings against each other on the same tiny classification head; that comparison
    (`proper` vs `reinforce` below) is the genuinely new part of this script.
  - TinyLoRA (2026) shows 13 RL-trained parameters can suffice for math reasoning via a frozen random
    projection; applying that shape to a frozen-trunk routing/typed-decision branch (a few classes, not
    open-ended generation) is a new combination as far as we found.

Usage:
  .venv/bin/python explore/learning_proper_score_objectives.py --smoke
  .venv/bin/python explore/learning_proper_score_objectives.py --out results/tarski/explore_proper_score_objectives.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `tarski`

from tarski.branches import Branch, ProbeBranch, mean_pool
from tarski.data import Dataset, Example, load_typed_decisions
from tarski.train import FeatureCache, autocast, evaluate, fit_temperature, predict_logits
from tarski.trunk import Trunk

from laya.common import QTYPES, proper_reward

DEFAULT_FULL_TASKS = [
    "invoice_processing.urgency",
    "agent_trace_observability.action",
    "agent_trace_observability.urgency",
    "customer_service.action",
    "customer_service.urgency",
    "security_incidents.severity",
    "customer_service.needs_human",
    "invoice_processing.duplicate",
]
DEFAULT_SMOKE_TASKS = ["agent_trace_observability.action", "customer_service.urgency"]
OBJECTIVES = ["ce", "cb_ce", "logit_adj", "proper", "reinforce"]
BRANCH_TYPES = ["probe", "bias"]


# ---------------------------------------------------------------------------------------------------
# The tiny branch
# ---------------------------------------------------------------------------------------------------

class BiasOnlyBranch(Branch):
    """K+1 trainable parameters: a fixed random projection, a per-class bias and one scale.

    The projection and the parameter-free LayerNorm carry no trainable weight -- everything the
    optimiser touches is `bias` (K numbers) and `scale` (1 number), the TinyLoRA regime applied to a
    classification head instead of a generation adapter.
    """

    kind = "bias_probe"

    def __init__(self, split: int, labels: List[str], hidden: int, seed: int = 0):
        super().__init__(split, labels, hidden)
        g = torch.Generator().manual_seed(seed)
        proj = torch.randn(hidden, len(labels), generator=g) / (hidden ** 0.5)
        self.register_buffer("proj", proj)
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)  # 0 params: just standardises
        self.bias = nn.Parameter(torch.zeros(len(labels)))
        self.scale = nn.Parameter(torch.ones(()))

    def logits(self, h, ctx):
        pooled = mean_pool(self.norm(h), ctx.attention_mask)
        return (pooled @ self.proj) * self.scale + self.bias


def make_branch(kind: str, split: int, labels: List[str], hidden: int, seed: int) -> Branch:
    if kind == "probe":
        return ProbeBranch(split, labels, hidden)
    if kind == "bias":
        return BiasOnlyBranch(split, labels, hidden, seed=seed)
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------------------------------

def infer_qtype(labels: Sequence[str]) -> int:
    """typed-decisions stringifies "score" question options as "0".."K-1" (tarski.data._option_keys);
    everything else ("choice", "noul") keeps its own key names. Treating that pattern as ordinal lets
    the proper-score reward's ranked-probability-score term engage for the ordinal tasks (severity,
    urgency, ...), where being off by one level should cost less than being off by three."""
    return QTYPES["score"] if all(l.isdigit() for l in labels) and \
        [int(l) for l in labels] == list(range(len(labels))) else QTYPES["choice"]


def class_balanced_weights(soft_train: torch.Tensor, beta: float = 0.99) -> torch.Tensor:
    """Cui, Jia, Lin, Song & Belongie, "Class-Balanced Loss Based on Effective Number of Samples",
    CVPR 2019 (arXiv:1901.05555): weight class c by 1/effective_number(c), effective_number(c) =
    (1-beta^n_c)/(1-beta). n_c is the soft-label mass per class rather than a hard count, since these
    targets are themselves soft."""
    n_c = soft_train.sum(0).clamp_min(1e-6)
    eff = (1.0 - beta ** n_c) / (1.0 - beta)
    w = (1.0 / eff)
    return w / w.mean()


def logit_adjust_bias(soft_train: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Menon, Jayasumana, Rawat, Jain, Veit & Kumar, "Long-Tail Learning via Logit Adjustment",
    ICLR 2021 (arXiv:2007.07314): add tau*log(prior) to the TRAINING logits so cross-entropy stops
    rewarding the shortcut of reporting the frequent class; evaluation logits stay unadjusted."""
    prior = soft_train.sum(0)
    prior = prior / prior.sum()
    return tau * torch.log(prior.clamp_min(1e-6))


def reinforce_step(probs: torch.Tensor, soft: torch.Tensor, qtype: torch.Tensor, mask: torch.Tensor,
                   baseline: Dict[str, float]) -> torch.Tensor:
    """REINFORCE on laya's proper_reward: sample a hard decision from the branch's own distribution,
    score that decision (as a one-hot report) against the soft target with the strictly-proper reward,
    and push the sampled action's log-probability by the reward above a running-mean baseline. This is
    the actual policy-gradient reading -- gradients flow only through log pi(a), never through the
    sampled outcome itself -- as opposed to `proper_direct_loss` below, which backpropagates through the
    full distribution."""
    with torch.no_grad():
        actions = torch.multinomial(probs.clamp_min(1e-8), 1).squeeze(-1)
        one_hot = F.one_hot(actions, probs.shape[-1]).float()
        reward = proper_reward(one_hot, soft, qtype, mask)
        b = baseline.get("mean", float(reward.mean()))
        advantage = reward - b
        baseline["mean"] = 0.9 * b + 0.1 * float(reward.mean())
    logp = torch.log(probs.gather(1, actions[:, None]).squeeze(1).clamp_min(1e-9))
    return -(advantage * logp).mean()


def proper_direct_loss(probs: torch.Tensor, soft: torch.Tensor, qtype: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    return -proper_reward(probs, soft, qtype, mask).mean()


def soft_ce(logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    return -(soft * F.log_softmax(logits, -1)).sum(-1).mean()


# ---------------------------------------------------------------------------------------------------
# Training loop (mirrors tarski.train.train_branch, with a pluggable objective)
# ---------------------------------------------------------------------------------------------------

def train_with_objective(branch: Branch, cache: FeatureCache, train_idx: List[int], y_train: torch.Tensor,
                         soft_train: torch.Tensor, val_idx: List[int], y_val: torch.Tensor,
                         objective: str, epochs: int, bs: int, lr: float, seed: int = 0,
                         patience: int = 3, min_val: int = 50, min_steps: int = 300) -> Dict:
    """Mirrors `tarski.train.train_branch`'s schedule exactly (that function was updated after an
    earlier run of this script under-trained every objective at typed-decisions' scale: a few hundred
    messages at batch 32 is only ~10 steps/epoch, so the old default of 8-10 epochs with patience=4 was
    stopping at 30-40 steps total, nowhere near enough for a linear head to separate a near-uniform soft
    target from noise -- most of the "collapse" the brief reported turned out to be under-training, not
    an objective problem). `min_steps` forces enough epochs regardless of dataset size; with fewer than
    `min_val` validation rows, best-epoch selection is too noisy to trust, so the full (longer) schedule
    runs and the final annealed weights are kept instead of an early, possibly-lucky epoch; with enough
    validation rows, early stopping is allowed only after half the (now-longer) schedule has run."""
    per_epoch = max(1, (len(train_idx) + bs - 1) // bs)
    epochs = max(epochs, -(-min_steps // per_epoch))       # ceil(min_steps / per_epoch)
    select = len(val_idx) >= min_val
    min_epochs = max(1, (epochs + 1) // 2)
    torch.manual_seed(seed)
    dev = cache.trunk.device
    branch.to(dev).train()
    opt = torch.optim.AdamW(branch.parameters(), lr=lr, weight_decay=0.01)
    labels = branch.labels
    qtype_val = infer_qtype(labels)
    class_w = class_balanced_weights(soft_train).to(dev) if objective == "cb_ce" else None
    adj_bias = logit_adjust_bias(soft_train).to(dev) if objective == "logit_adj" else None
    baseline: Dict[str, float] = {}
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        branch.train()
        order = list(range(len(train_idx)))
        rng.shuffle(order)
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            idx = [train_idx[j] for j in sel]
            h, ctx = cache.batch(branch.split, idx)
            logits = branch.logits(h, ctx).float()
            y = y_train[sel].to(dev)
            soft = soft_train[sel].to(dev)
            mask = torch.ones_like(logits)
            qtype = torch.full((logits.shape[0],), qtype_val, device=dev)
            if objective == "ce":
                loss = soft_ce(logits, soft)
            elif objective == "cb_ce":
                per = -(soft * F.log_softmax(logits, -1)).sum(-1)
                loss = (per * class_w[y]).mean()
            elif objective == "logit_adj":
                loss = soft_ce(logits + adj_bias[None, :], soft)
            elif objective == "proper":
                loss = proper_direct_loss(F.softmax(logits, -1), soft, qtype, mask)
            elif objective == "reinforce":
                loss = reinforce_step(F.softmax(logits, -1), soft, qtype, mask, baseline)
            else:
                raise ValueError(objective)
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
    return {"val_acc": best, "epochs_run": ep + 1, "epochs_scheduled": epochs,
           "selection": "best_val_acc" if select else "last_epoch"}


# ---------------------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------------------

def targets(examples: List[Example], task: str):
    y = torch.tensor([e.y[task] for e in examples])
    soft = torch.tensor(np.stack([e.soft[task] for e in examples])) if all(task in e.soft for e in examples) \
        else F.one_hot(y).float()
    return y, soft


def run(tasks: List[str], split: int, epochs: int, bs: int, lr: float, seed: int, device: str,
       cap_train: Optional[int], cap_val: Optional[int], cap_test: Optional[int],
       branch_types: List[str], objectives: List[str], min_val: int = 50, min_steps: int = 300,
       log=print) -> Dict:
    t_start = time.time()
    trunk = Trunk(device=device)
    ds = load_typed_decisions(seed=seed)
    log(f"loaded typed-decisions: {ds.summary()}")

    def pool(split_name):
        rows = [e for e in ds.split(split_name) if any(t in e.y for t in tasks)]
        cap = {"train": cap_train, "val": cap_val, "test": cap_test}[split_name]
        if cap and len(rows) > cap:
            # typed-decisions' HF splits are laid out in contiguous per-workflow blocks, so a plain
            # rows[:cap] can silently zero out every task but the first; sample instead so a small
            # smoke pool still covers every requested task.
            rng_local = np.random.default_rng(seed)
            keep = sorted(rng_local.choice(len(rows), size=cap, replace=False).tolist())
            rows = [rows[i] for i in keep]
        return rows

    train_pool, val_pool, test_pool = pool("train"), pool("val"), pool("test")
    all_rows = train_pool + val_pool + test_pool
    n_tr, n_va = len(train_pool), len(val_pool)
    log(f"pooled rows for {len(tasks)} tasks: train={n_tr} val={n_va} test={len(test_pool)}")

    t0 = time.time()
    cache = FeatureCache(trunk, [e.text for e in all_rows], [split], ds.max_len)
    log(f"cached trunk depth {split} for {len(all_rows)} messages in {cache.seconds:.1f}s")

    per_task: Dict[str, Dict] = {}
    baseline_majority: Dict[str, float] = {}
    n_runs = 0
    for task in tasks:
        tr = [i for i in range(0, n_tr) if task in all_rows[i].y]
        va = [i for i in range(n_tr, n_tr + n_va) if task in all_rows[i].y]
        te = [i for i in range(n_tr + n_va, len(all_rows)) if task in all_rows[i].y]
        if len(tr) < 8 or len(va) < 4 or len(te) < 4:
            log(f"  [{task}] skipped: too few rows after pooling (train={len(tr)} val={len(va)} test={len(te)})")
            continue
        y_tr, soft_tr = targets([all_rows[i] for i in tr], task)
        y_va, _ = targets([all_rows[i] for i in va], task)
        y_te, soft_te = targets([all_rows[i] for i in te], task)
        labels = ds.tasks[task].labels
        majority = int(soft_tr.sum(0).argmax())
        maj_acc = float((y_te == majority).float().mean())
        baseline_majority[task] = maj_acc
        per_task[task] = {}
        for bt in branch_types:
            per_task[task][bt] = {}
            for obj in objectives:
                n_runs += 1
                branch = make_branch(bt, split, labels, trunk.hidden, seed)
                t1 = time.time()
                info = train_with_objective(branch, cache, tr, y_tr, soft_tr, va, y_va, obj, epochs,
                                            min(bs, max(4, len(tr))), lr, seed,
                                            min_val=min_val, min_steps=min_steps)
                t = fit_temperature(predict_logits(branch, cache, va), y_va) if len(va) >= 30 else 1.0
                z = predict_logits(branch, cache, te)
                probs = torch.softmax(z / t, -1).numpy()
                m = evaluate(probs, y_te.numpy(), soft_te.numpy())
                pred = probs.argmax(-1)
                m["frac_pred_majority"] = float((pred == majority).mean())
                m["distinct_pred_ratio"] = len(set(pred.tolist())) / len(labels)
                m["params"] = sum(p.numel() for p in branch.parameters())
                m["train_s"] = round(time.time() - t1, 2)
                m["epochs_run"], m["epochs_scheduled"], m["selection"] = \
                    info["epochs_run"], info["epochs_scheduled"], info["selection"]
                per_task[task][bt][obj] = m
                log(f"  [{task}][{bt}][{obj}] acc={m['acc']:.3f} f1={m['macro_f1']:.3f} "
                    f"brier={m.get('brier_soft', float('nan')):.3f} maj_frac={m['frac_pred_majority']:.2f} "
                    f"params={m['params']} epochs={m['epochs_run']}/{m['epochs_scheduled']} ({m['train_s']}s)")

    summary: Dict[str, Dict[str, Dict]] = {}
    for bt in branch_types:
        summary[bt] = {}
        for obj in objectives:
            rows = [per_task[t][bt][obj] for t in per_task if obj in per_task[t].get(bt, {})]
            if not rows:
                continue
            summary[bt][obj] = {
                "mean_acc": float(np.mean([r["acc"] for r in rows])),
                "mean_macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
                "mean_brier_soft": float(np.mean([r["brier_soft"] for r in rows])),
                "mean_ece": float(np.mean([r["ece"] for r in rows])),
                "mean_frac_pred_majority": float(np.mean([r["frac_pred_majority"] for r in rows])),
                "mean_distinct_pred_ratio": float(np.mean([r["distinct_pred_ratio"] for r in rows])),
                "mean_params": float(np.mean([r["params"] for r in rows])),
            }
    mean_majority = float(np.mean(list(baseline_majority.values()))) if baseline_majority else float("nan")
    result = {
        "config": {"tasks": tasks, "split": split, "epochs": epochs, "bs": bs, "lr": lr, "seed": seed,
                  "device": device, "n_runs": n_runs,
                  "cap_train": cap_train, "cap_val": cap_val, "cap_test": cap_test},
        "baseline_majority": baseline_majority,
        "reference": {"mean_majority_baseline_acc": mean_majority,
                     "laya_full_finetune_mean_acc_all_20_tasks": 0.766,
                     "note": "laya's 0.766 is a full fine-tune of a much larger cross-encoder over all "
                             "20 tasks; it is an anchor, not an apples-to-apples baseline for these tiny "
                             "frozen-trunk branches trained on a subset of tasks."},
        "per_task": per_task,
        "summary": summary,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny subset, finishes in under ~2 minutes")
    ap.add_argument("--out", default=None, help="write JSON results here")
    ap.add_argument("--split", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tasks", default=None, help="comma-separated task names; default depends on --smoke")
    ap.add_argument("--branch-types", default=",".join(BRANCH_TYPES))
    ap.add_argument("--objectives", default=",".join(OBJECTIVES))
    ap.add_argument("--min-steps", type=int, default=300,
                    help="force at least this many optimiser steps regardless of dataset size "
                         "(matches tarski.train.train_branch's default; a few hundred typed-decisions "
                         "messages at batch 32 is only ~10 steps/epoch, so this is what actually fixed "
                         "the majority-class 'collapse' the brief reported -- it was under-training)")
    ap.add_argument("--min-val", type=int, default=50,
                    help="need at least this many validation rows to trust best-epoch selection; "
                         "below it, run the full schedule and keep the last (annealed) epoch")
    args = ap.parse_args()

    if args.smoke:
        device = "cpu"
        tasks = (args.tasks or ",".join(DEFAULT_SMOKE_TASKS)).split(",")
        epochs, cap_train, cap_val, cap_test = 3, 48, 20, 30
    else:
        device = args.device
        tasks = (args.tasks or ",".join(DEFAULT_FULL_TASKS)).split(",")
        epochs, cap_train, cap_val, cap_test = args.epochs, None, None, None

    result = run(tasks, args.split, epochs, args.bs, args.lr, args.seed, device,
                cap_train, cap_val, cap_test, args.branch_types.split(","), args.objectives.split(","),
                min_val=args.min_val, min_steps=args.min_steps)

    print("\n=== summary (mean over tasks) ===")
    for bt, objs in result["summary"].items():
        for obj, m in objs.items():
            print(f"{bt:10s} {obj:10s} acc={m['mean_acc']:.3f} f1={m['mean_macro_f1']:.3f} "
                  f"brier={m['mean_brier_soft']:.3f} maj_frac={m['mean_frac_pred_majority']:.2f} "
                  f"distinct={m['mean_distinct_pred_ratio']:.2f} params={m['mean_params']:.0f}")
    print(f"mean majority-class baseline acc: {result['reference']['mean_majority_baseline_acc']:.3f}")
    print(f"elapsed: {result['elapsed_s']}s")

    out = args.out or (None if args.smoke else "results/tarski/explore_proper_score_objectives.json")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
