"""Far-field idea 10: sleep-like consolidation of independently trained branches (hippocampal replay).

During sleep, replay consolidates many episodic traces into cortical weights without the original stimuli.
Here N independently trained blocks branches at the same (split, depth) are consolidated into ONE body of d
layers with N heads by replaying cached trunk states through the teachers: every training message, labelled
or not, gets each teacher's soft output, and the student minimises the sum of KL divergences. No labels are
used in consolidation. A second, incremental round adds one more task to an already consolidated body by
replaying the body's own outputs for the old tasks (self-distillation) plus the new teacher, and measures
forgetting.

Compared per task: independent branches (N x d layers, the baseline), consolidated body (d layers),
a jointly supervised multi-task body (d layers, the usual multi-task upper reference), and the weight-merging
results of explore/merge_branches.py when its JSON is present (mean / TIES + refit head).

Testable prediction: replay consolidation lands within 1 point of the independent branches and above both
weight merging and joint training; the incremental round costs the old tasks under 0.5 points.

Usage:
  .venv/bin/python explore/farfield_consolidate.py --smoke
  .venv/bin/python explore/farfield_consolidate.py --dataset clinc150 --split 11 --out results/tarski/explore_consolidate_clinc150.json
  .venv/bin/python explore/farfield_consolidate.py --dataset typed_cs --split 14 --out results/tarski/explore_consolidate_typed_cs.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from explore.farfield_common import (LAMBDA_DIR, Logger, dump, labels_of, out_paths, soft_of, split_indices, subsample, task_rows,
                                     train_and_eval)
from tarski import data
from tarski.branches import BlockBranch, mean_pool
from tarski.train import FeatureCache, evaluate, fit_temperature
from tarski.trunk import Trunk


class MultiHeadBody(nn.Module):
    """d copied layers + final norm shared by all tasks, one linear head per task."""

    def __init__(self, trunk: Trunk, split: int, depth: int, heads: Dict[str, int]):
        super().__init__()
        self.split, self.depth = split, depth
        self.layers = trunk.copy_layers(split, split + depth)
        self.norm = trunk.copy_final_norm()
        self.heads = nn.ModuleDict({t.replace(".", "__"): nn.Linear(trunk.hidden, c) for t, c in heads.items()})

    def pooled(self, h, ctx):
        return mean_pool(self.norm(ctx.run(self.layers, h)), ctx.attention_mask)

    def forward(self, h, ctx) -> Dict[str, torch.Tensor]:
        p = self.pooled(h, ctx)
        return {t.replace("__", "."): head(p) for t, head in self.heads.items()}


@torch.no_grad()
def predict_body(body: MultiHeadBody, cache: FeatureCache, rows: List[int], bs: int = 128) -> Dict[str, torch.Tensor]:
    body.eval()
    order = sorted(range(len(rows)), key=lambda j: cache.lengths[rows[j]])
    out = {t: torch.zeros(len(rows), head.out_features) for t, head in zip([k.replace("__", ".") for k in body.heads], body.heads.values())}
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        h, ctx = cache.batch(body.split, [rows[j] for j in sel])
        z = body(h, ctx)
        for t in z:
            out[t][sel] = z[t].float().cpu()
    return out


@torch.no_grad()
def teacher_probs(branch: BlockBranch, cache: FeatureCache, rows: List[int], bs: int = 128) -> torch.Tensor:
    from tarski.train import predict_logits
    return torch.softmax(predict_logits(branch, cache, rows, bs) / branch.temperature.float().cpu(), -1)


def train_body(body: MultiHeadBody, cache: FeatureCache, rows: List[int], targets: Dict[str, torch.Tensor],
               mask: Dict[str, torch.Tensor], epochs: int, min_steps: int, seed: int, log, bs: int = 32,
               lr_layers: float = 1e-4, lr_head: float = 1e-3) -> Dict:
    """Minimise sum_t mean over rows where mask[t] of -(targets[t] * log_softmax(student_t)).
    `targets[t]` are soft distributions aligned with `rows`; teacher outputs for replay, labels for joint."""
    torch.manual_seed(seed)
    dev = cache.trunk.device
    body.to(dev).train()
    per_epoch = (len(rows) + bs - 1) // bs
    epochs = max(epochs, -(-min_steps // per_epoch))
    groups = [{"params": [p for n, p in body.named_parameters() if n.startswith("heads.")], "lr": lr_head},
              {"params": [p for n, p in body.named_parameters() if not n.startswith("heads.")], "lr": lr_layers}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * per_epoch,
                                                pct_start=0.1, anneal_strategy="cos")
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        order = sorted(range(len(rows)), key=lambda j: cache.lengths[rows[j]] + rng.random() * 8)
        chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
        rng.shuffle(chunks)
        tot = 0.0
        for sel in chunks:
            h, ctx = cache.batch(body.split, [rows[j] for j in sel])
            z = body(h, ctx)
            loss = 0.0
            for t in z:
                m = mask[t][sel]
                if m.any():
                    tgt = targets[t][sel][m].to(dev)
                    loss = loss + (-(tgt * F.log_softmax(z[t].float()[m.to(dev)], -1)).sum(-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(body.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(sel)
        hist.append(tot / len(rows))
        log(f"    epoch {ep + 1}/{epochs} loss {tot / len(rows):.4f}")
    body.eval()
    return {"epochs": epochs, "steps": epochs * per_epoch, "loss": hist}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dataset", default="clinc150", help="clinc150 or typed_cs (typed-decisions customer_service tasks)")
    ap.add_argument("--split", type=int, default=None)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6, help="teacher epochs (min_steps applies)")
    ap.add_argument("--replay-epochs", type=int, default=3)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.device = args.device or "cpu"
        args.epochs, args.replay_epochs, args.min_steps = 1, 1, 6
    if args.split is None:
        args.split = 14 if args.dataset == "typed_cs" else 11
    out, logp = out_paths(args, f"consolidate_{args.dataset}")
    L = Logger(logp)
    t_start = time.time()
    trunk = Trunk(device=args.device)

    if args.dataset == "typed_cs":
        ds = data.load_typed_decisions()
        tasks = [t for t in ds.tasks if t.startswith("customer_service.")]
        keep = lambda e: any(t in e.y for t in tasks)
        ds = copy.copy(ds)
        ds.train, ds.val, ds.test = [e for e in ds.train if keep(e)], [e for e in ds.val if keep(e)], [e for e in ds.test if keep(e)]
        if args.smoke:
            ds.train, ds.val, ds.test = subsample(ds.train, 100, args.seed), subsample(ds.val, 20, args.seed + 1), subsample(ds.test, 50, args.seed + 2)
    else:
        ds = data.load(args.dataset)
        tasks = list(ds.tasks)
        if args.smoke:
            key = lambda e: e.y["intent"]
            ds = copy.copy(ds)
            ds.train, ds.val, ds.test = (subsample(ds.train, 600, args.seed, key), subsample(ds.val, 151, args.seed + 1, key),
                                         subsample(ds.test, 300, args.seed + 2, key))
    allx = ds.train + ds.val + ds.test
    idx = split_indices(len(ds.train), len(ds.val), len(allx))
    L(f"== consolidation on {ds.summary()} | tasks {tasks} | blocks@{args.split}+{args.depth} | device {trunk.device}")
    cache = FeatureCache(trunk, [e.text for e in allx], [args.split], ds.max_len)
    L(f"  cached depth {args.split}: {cache.seconds:.1f}s ({cache.bytes() / 1e6:.0f} MB)")
    results = {"config": vars(args), "device": str(trunk.device), "tasks": tasks, "independent": {}, "consolidated": {},
               "joint": {}, "incremental": {}}

    # --- teachers: independent branches ------------------------------------------------------------------
    teachers: Dict[str, BlockBranch] = {}
    t0 = time.time()
    for t in tasks:
        br = BlockBranch(args.split, ds.tasks[t].labels, trunk.hidden, args.depth, trunk, init="next")
        r = train_and_eval(br, cache, allx, idx, t, epochs=args.epochs, min_steps=args.min_steps, seed=args.seed)
        teachers[t] = br
        results["independent"][t] = r["metrics"]
        L(f"  [teacher {t}] test acc {r['metrics']['acc']:.4f} macro-F1 {r['metrics']['macro_f1']:.4f} ({r['metrics']['train_s']}s)")
    results["independent_train_s"] = round(time.time() - t0, 1)

    train_rows = list(idx["train"])
    test_rows = {t: task_rows(allx, t, idx["test"]) for t in tasks}
    val_rows = {t: task_rows(allx, t, idx["val"]) for t in tasks}

    def eval_body(body: MultiHeadBody, tag: str, subset: List[str]) -> Dict[str, Dict]:
        res = {}
        for t in subset:
            zv = predict_body(body, cache, val_rows[t])[t]
            yv, sv = labels_of(allx, t, val_rows[t]), soft_of(allx, t, val_rows[t])
            temp = fit_temperature(zv, yv, sv) if len(val_rows[t]) >= 30 else 1.0
            zt = predict_body(body, cache, test_rows[t])[t]
            yt, st = labels_of(allx, t, test_rows[t]), soft_of(allx, t, test_rows[t])
            res[t] = evaluate(torch.softmax(zt / temp, -1).numpy(), yt.numpy(), None if st is None else st.numpy())
            L(f"  [{tag} {t}] test acc {res[t]['acc']:.4f} macro-F1 {res[t]['macro_f1']:.4f}")
        return res

    def replay_targets(sources: Dict[str, object]) -> Dict[str, torch.Tensor]:
        """Soft targets for every training row from a teacher branch or a consolidated body, per task."""
        out = {}
        for t, src in sources.items():
            if isinstance(src, BlockBranch):
                out[t] = teacher_probs(src, cache, train_rows)
            else:
                out[t] = torch.softmax(predict_body(src, cache, train_rows)[t], -1)
        return out

    all_mask = {t: torch.ones(len(train_rows), dtype=torch.bool) for t in tasks}

    # --- one-shot consolidation by replay (no labels) ----------------------------------------------------
    t0 = time.time()
    body = MultiHeadBody(trunk, args.split, args.depth, {t: len(ds.tasks[t].labels) for t in tasks})
    for t in tasks:                                                   # heads start from the teachers' heads
        body.heads[t.replace(".", "__")].load_state_dict(teachers[t].out.state_dict())
    info = train_body(body, cache, train_rows, replay_targets(teachers), all_mask, args.replay_epochs, args.min_steps, args.seed, L)
    results["consolidated"] = eval_body(body, "consolidated", tasks)
    results["consolidated_train"] = {"seconds": round(time.time() - t0, 1), **{k: v for k, v in info.items() if k != "loss"}}

    # --- joint supervised multi-task body (labels only, the usual upper reference) ------------------------
    t0 = time.time()
    joint = MultiHeadBody(trunk, args.split, args.depth, {t: len(ds.tasks[t].labels) for t in tasks})
    lab_targets, lab_mask = {}, {}
    for t in tasks:
        m = torch.tensor([t in allx[i].y for i in train_rows])
        tgt = torch.zeros(len(train_rows), len(ds.tasks[t].labels))
        rows_t = [i for i in train_rows if t in allx[i].y]
        s = soft_of(allx, t, rows_t)
        tgt[m] = s if s is not None else F.one_hot(labels_of(allx, t, rows_t), len(ds.tasks[t].labels)).float()
        lab_targets[t], lab_mask[t] = tgt, m
    info = train_body(joint, cache, train_rows, lab_targets, lab_mask, args.replay_epochs, args.min_steps, args.seed, L)
    results["joint"] = eval_body(joint, "joint", tasks)
    results["joint_train"] = {"seconds": round(time.time() - t0, 1), **{k: v for k, v in info.items() if k != "loss"}}

    # --- incremental: consolidate tasks[:-1], then add tasks[-1] with self-replay for the old ones --------
    if len(tasks) >= 2:
        old, new = tasks[:-1], tasks[-1]
        t0 = time.time()
        body_a = MultiHeadBody(trunk, args.split, args.depth, {t: len(ds.tasks[t].labels) for t in old})
        for t in old:
            body_a.heads[t.replace(".", "__")].load_state_dict(teachers[t].out.state_dict())
        train_body(body_a, cache, train_rows, replay_targets({t: teachers[t] for t in old}), {t: all_mask[t] for t in old},
                   args.replay_epochs, args.min_steps, args.seed, L)
        stage_a = eval_body(body_a, "stage A", old)
        body_b = MultiHeadBody(trunk, args.split, args.depth, {t: len(ds.tasks[t].labels) for t in tasks})
        body_b.layers.load_state_dict(body_a.layers.state_dict())
        body_b.norm.load_state_dict(body_a.norm.state_dict())
        for t in old:
            body_b.heads[t.replace(".", "__")].load_state_dict(body_a.heads[t.replace(".", "__")].state_dict())
        body_b.heads[new.replace(".", "__")].load_state_dict(teachers[new].out.state_dict())
        sources = {t: body_a for t in old}
        sources[new] = teachers[new]
        train_body(body_b, cache, train_rows, replay_targets(sources), all_mask, args.replay_epochs, args.min_steps, args.seed + 1, L)
        stage_b = eval_body(body_b, "stage B", tasks)
        results["incremental"] = {"old_tasks": old, "new_task": new, "stage_a": stage_a, "stage_b": stage_b,
                                  "forgetting_acc": {t: stage_b[t]["acc"] - stage_a[t]["acc"] for t in old},
                                  "seconds": round(time.time() - t0, 1)}
        L(f"  incremental: forgetting on old tasks {results['incremental']['forgetting_acc']}")

    # --- sibling weight-merging results, if present ---------------------------------------------------------
    merge_path = os.path.join(LAMBDA_DIR, f"explore_merge_{'typed_cs' if args.dataset == 'typed_cs' else args.dataset}.json")
    if os.path.exists(merge_path):
        mj = json.load(open(merge_path))
        results["weight_merge_reference"] = {k: v for k, v in mj.get("mean", {}).items()}

    def mean_acc(d):
        return float(np.mean([d[t]["acc"] for t in d])) if d else None

    results["mean_acc"] = {"independent": mean_acc(results["independent"]), "consolidated": mean_acc(results["consolidated"]),
                           "joint": mean_acc(results["joint"])}
    results["wall_seconds"] = round(time.time() - t_start, 1)
    dump(results, out)
    L(f"== done in {results['wall_seconds']}s; wrote {out}")
    L(f"   mean acc: independent {results['mean_acc']['independent']:.4f} | consolidated (replay, no labels) "
      f"{results['mean_acc']['consolidated']:.4f} | joint (labels) {results['mean_acc']['joint']:.4f}"
      + (f" | weight-merge reference {json.dumps({k: round(v, 4) for k, v in results['weight_merge_reference'].items()})}"
         if "weight_merge_reference" in results else ""))


if __name__ == "__main__":
    main()
