"""Idea 9: shared decision subspaces. Do decisions that use the same trunk directions share branch compute?

Three tests on the frozen trunk at split depth 11 (typed-decisions: 20 decisions in 4 workflows of 5;
CLINC150: intent, domain, oos on the same messages):

  affinity    Per-task logistic probes on standardized mean-pooled states. Affinity(s, t) is the mean
              squared cosine of the principal angles between the two probes' class-contrast subspaces
              (column spaces of the centred weight matrices), against random subspaces of the same ranks.
              Reported: within-workflow vs across-workflow mean, the full matrix, and a same-messages null
              (affinity to a probe fitted on the same messages with shuffled labels), since probes fitted
              on the same few hundred messages share noise directions.
  bottleneck  A group of decisions shares one rank-r projection U (768 x r) of the pooled state, with its own
              heads on top (trained jointly, each task on its own labelled messages). Groups: the 4
              workflows, 3 random partitions into groups of 5 that mix workflows, groups built greedily from
              the affinity matrix, and singletons. If decisions share directions, affinity/workflow groups
              lose less than random groups at small r.
  fused       Real branch compute: one blocks@11+2 stack with one head per decision in a group (a workflow's
              5 decisions; CLINC's 3) vs one blocks@11+2 per decision (tarski.train.train_branch). Fused costs
              2 layers per message instead of 2 x n_decisions.

Prediction (geometry.md idea 9): within-workflow affinity > its shuffled-label null > across ~ random; at
r = 8 workflow/affinity
groups beat random groups; fused blocks within ~1-2 points of separate blocks at 1/5 (typed) or 1/3 (CLINC)
of the branch compute.

Usage:
  .venv/bin/python explore/geometry_subspaces.py --smoke
  .venv/bin/python explore/geometry_subspaces.py --out results/tarski/explore_subspaces.json   # ~15-20 min on an A10
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_common import (Logger, acc, device, fit_logreg, labels, load_dataset, message_groups, pick_by_val,
                             pooled, splits, standardize, targets, task_split, warmup_cosine)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tarski import train as ttrain
from tarski.branches import mean_pool
from tarski.trunk import Trunk


# ---------------------------------------------------------------------------------------------------
# Affinity between probe subspaces
# ---------------------------------------------------------------------------------------------------

def probe_subspace(W: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of the class-contrast directions of a (D, C) probe weight matrix."""
    Wc = (W - W.mean(1, keepdim=True)).double()
    U, S, _ = torch.linalg.svd(Wc, full_matrices=False)
    return U[:, : int((S > 1e-6 * S.max()).sum())]


def affinity(Qa, Qb) -> float:
    return float((Qa.T @ Qb).pow(2).sum() / min(Qa.shape[1], Qb.shape[1]))


def random_affinity(ra, rb, D, n=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n):
        Qa, _ = torch.linalg.qr(torch.randn(D, ra, generator=g, dtype=torch.float64))
        Qb, _ = torch.linalg.qr(torch.randn(D, rb, generator=g, dtype=torch.float64))
        vals.append(affinity(Qa, Qb))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------------------------------
# Shared rank-r bottleneck for a group of decisions
# ---------------------------------------------------------------------------------------------------

def fit_bottleneck(X, allx, group, ds, n_tr, n_va, r, dev, steps, l2=1e-3, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    D = X.shape[1]
    data = []
    for t in group:
        tr, va, te = task_split(allx, t, n_tr, n_va)
        C = len(ds.tasks[t].labels)
        data.append((t, X[tr].to(dev), targets(allx, tr, t, C).to(dev), X[te].to(dev), labels(allx, te, t)))
    U = nn.Parameter(torch.randn(D, r, device=dev) / D ** 0.5)
    heads = nn.ModuleList(nn.Linear(r, len(ds.tasks[t].labels)) for t in group).to(dev)
    opt = torch.optim.Adam([U] + list(heads.parameters()), lr=lr)
    for _ in range(steps):
        loss = 0.0
        for (t, Xtr, Ttr, _, _), h in zip(data, heads):
            z = h(Xtr @ U)
            loss = loss + (-(Ttr * F.log_softmax(z, -1)).sum(-1).mean())
        loss = loss + l2 * (U.pow(2).sum() + sum(h.weight.pow(2).sum() for h in heads))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return {t: acc(h(Xte @ U).cpu(), yte) for (t, _, _, Xte, yte), h in zip(data, heads)}


def greedy_groups(tasks, A, size):
    """Partition tasks into groups of `size`, each seeded by the least-assigned task and filled with the
    tasks of highest mean affinity to the group."""
    left, groups = list(tasks), []
    while left:
        g = [left.pop(0)]
        while len(g) < size and left:
            nxt = max(left, key=lambda t: np.mean([A[t][u] for u in g]))
            g.append(nxt)
            left.remove(nxt)
        groups.append(g)
    return groups


# ---------------------------------------------------------------------------------------------------
# Fused blocks branch: one stack, one head per decision
# ---------------------------------------------------------------------------------------------------

class FusedBlocks(nn.Module):
    def __init__(self, trunk: Trunk, split: int, depth: int, n_labels, dropout=0.1):
        super().__init__()
        self.split = split
        self.layers = trunk.copy_layers(split, split + depth)
        self.norm = trunk.copy_final_norm()
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList(nn.Linear(trunk.hidden, n) for n in n_labels)

    def forward(self, h, ctx):
        x = self.drop(mean_pool(self.norm(ctx.run(self.layers, h)), ctx.attention_mask))
        return [hd(x) for hd in self.heads]


def train_fused(trunk, fc, allx, group, ds, tr, va, te, epochs=6, bs=32, lr_layers=1e-4, lr_head=1e-3,
                min_steps=300, min_val=50, seed=0, log=print):
    torch.manual_seed(seed)
    dev = trunk.device
    model = FusedBlocks(trunk, fc.depths[0], 2, [len(ds.tasks[t].labels) for t in group]).to(dev)
    lp = [p for n, p in model.named_parameters() if n.startswith("layers.")]
    hp = [p for n, p in model.named_parameters() if not n.startswith("layers.")]
    opt = torch.optim.AdamW([{"params": hp, "lr": lr_head}, {"params": lp, "lr": lr_layers}], weight_decay=0.01)
    per_epoch = -(-len(tr) // bs)
    epochs = max(epochs, -(-min_steps // per_epoch))
    sched = warmup_cosine(opt, epochs * per_epoch)
    T = {t: targets(allx, tr, t, len(ds.tasks[t].labels)) for t in group}
    pos = {i: j for j, i in enumerate(tr)}
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def predict(idx):
        model.eval()
        out = [torch.zeros(len(idx), len(ds.tasks[t].labels)) for t in group]
        order = sorted(range(len(idx)), key=lambda j: fc.lengths[idx[j]])
        for s in range(0, len(order), 128):
            sel = order[s:s + 128]
            h, ctx = fc.batch(fc.depths[0], [idx[j] for j in sel])
            for k, z in enumerate(model(h, ctx)):
                out[k][sel] = z.float().cpu()
        return out

    select = len(va) >= min_val
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        order = sorted(tr, key=lambda i: fc.lengths[i] + rng.random() * 8)
        chunks = [order[k:k + bs] for k in range(0, len(order), bs)]
        rng.shuffle(chunks)
        for c in chunks:
            h, ctx = fc.batch(fc.depths[0], c)
            zs = model(h, ctx)
            rows = torch.tensor([pos[i] for i in c])
            loss = sum(-(T[t][rows].to(dev) * F.log_softmax(z.float(), -1)).sum(-1).mean() for t, z in zip(group, zs)) / len(group)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if select:
            zv = predict(va)
            v = float(np.mean([acc(z, labels(allx, va, t)) for t, z in zip(group, zv)]))
            if v > best:
                best, best_state = v, copy.deepcopy(model.state_dict())
    if select:
        model.load_state_dict(best_state)
    zt = predict(te)
    return {t: acc(z, labels(allx, te, t)) for t, z in zip(group, zt)}, epochs * per_epoch


def run_dataset(trunk, ds, dev, args, log):
    allx, n_tr, n_va = splits(ds)
    tasks = list(ds.tasks)
    wf = message_groups(allx, tasks)                 # typed-decisions: workflows; CLINC150: one group
    out = {}
    feats = pooled(trunk, ds, [args.split])
    X = standardize(feats[args.split], list(range(n_tr)))
    # --- affinity -------------------------------------------------------------------------------
    Q, Qnull, probe_acc = {}, {}, {}
    perm_rng = np.random.default_rng(1)
    for t in tasks:
        tr, va, te = task_split(allx, t, n_tr, n_va)
        C = len(ds.tasks[t].labels)
        T = targets(allx, tr, t, C)
        r = fit_logreg(X[tr], T, [X[va], X[te]], dev, steps=args.steps)
        g = pick_by_val(r["logits"][0], labels(allx, va, t))
        probe_acc[t] = acc(r["logits"][1][g], labels(allx, te, t))
        Q[t] = probe_subspace(r["W"][g])
        # same-messages null: the same task's targets shuffled across its messages. Probes fitted on the same
        # few hundred messages share noise directions; this measures how much affinity that alone produces.
        rn = fit_logreg(X[tr], T[torch.tensor(perm_rng.permutation(len(tr)))], [X[va]], dev, l2s=[(1e-4, 1e-3, 1e-2)[g]],
                        steps=args.steps)
        Qnull[t] = probe_subspace(rn["W"][0])
    A = {s: {t: affinity(Q[s], Q[t]) for t in tasks} for s in tasks}
    D = X.shape[1]
    rnd = {s: {t: random_affinity(Q[s].shape[1], Q[t].shape[1], D, n=10) for t in tasks} for s in tasks}
    pairs = [(s, t) for i, s in enumerate(tasks) for t in tasks[i + 1:]]
    within = [A[s][t] for s, t in pairs if wf[s] == wf[t]]
    across = [A[s][t] for s, t in pairs if wf[s] != wf[t]]
    within_null = [0.5 * (affinity(Q[s], Qnull[t]) + affinity(Qnull[s], Q[t])) for s, t in pairs if wf[s] == wf[t]]
    out["affinity"] = {"matrix": A, "within_workflow_mean": float(np.mean(within)) if within else None,
                       "within_workflow_shuffled_label_null": float(np.mean(within_null)) if within_null else None,
                       "across_workflow_mean": float(np.mean(across)) if across else None,
                       "random_mean": float(np.mean([rnd[s][t] for s, t in pairs])) if pairs else None,
                       "ranks": {t: int(Q[t].shape[1]) for t in tasks}, "probe_acc": probe_acc}
    log(f"   affinity: within-group {out['affinity']['within_workflow_mean']} (shuffled-label null "
        f"{out['affinity']['within_workflow_shuffled_label_null']}) across {out['affinity']['across_workflow_mean']} "
        f"random {out['affinity']['random_mean']}")
    # --- shared bottleneck ------------------------------------------------------------------------
    if len(set(wf.values())) > 1:
        size = max(len([t for t in tasks if wf[t] == w]) for w in set(wf.values()))
        rng = np.random.default_rng(0)
        partitions = {"workflow": [[t for t in tasks if wf[t] == w] for w in sorted(set(wf.values()))],
                      "affinity_greedy": greedy_groups(tasks, A, size), "singleton": [[t] for t in tasks]}
        for k in range(3):
            perm = list(rng.permutation(tasks))
            partitions[f"random{k}"] = [perm[i:i + size] for i in range(0, len(perm), size)]
    else:
        partitions = {"all": [tasks], "singleton": [[t] for t in tasks]}
    out["bottleneck"] = {}
    for name, parts in partitions.items():
        for r in args.ranks:
            accs = {}
            for g in parts:
                accs.update(fit_bottleneck(X, allx, g, ds, n_tr, n_va, r, dev, args.steps))
            out["bottleneck"].setdefault(name, {})[r] = {"mean_acc": float(np.mean(list(accs.values()))), "tasks": accs}
        log(f"   bottleneck {name:>16}: " + " ".join(f"r={r}:{out['bottleneck'][name][r]['mean_acc']:.4f}" for r in args.ranks))
    # --- fused vs separate blocks -------------------------------------------------------------------
    if args.no_blocks:
        return out
    t0 = time.time()
    fc = ttrain.FeatureCache(trunk, [e.text for e in allx], [args.split], ds.max_len)
    groups = partitions.get("workflow") or partitions["all"]
    sep, fused, steps_used = {}, {}, {}
    for g in groups:
        tr, va, te = task_split(allx, g[0], n_tr, n_va)
        f, n_steps = train_fused(trunk, fc, allx, g, ds, tr, va, te, epochs=args.block_epochs,
                                 min_steps=args.min_steps, log=log)
        fused.update(f)
        steps_used["|".join(g)] = n_steps
        for t in g:
            y = {s: torch.tensor(labels(allx, idx, t)) for s, idx in (("tr", tr), ("va", va))}
            soft = targets(allx, tr, t, len(ds.tasks[t].labels)) if all(t in allx[i].soft for i in tr) else None
            br = ttrain.make_branch("blocks", args.split, ds.tasks[t].labels, trunk, 2)
            ttrain.train_branch(br, fc, tr, y["tr"], soft, va, y["va"], epochs=args.block_epochs, lr_layers=1e-4,
                                lr_head=1e-3, min_steps=args.min_steps)
            sep[t] = acc(ttrain.predict_logits(br, fc, te), labels(allx, te, t))
            del br
    med = float(np.median(fc.lengths))
    out["fused_blocks"] = {"separate_mean_acc": float(np.mean(list(sep.values()))),
                           "fused_mean_acc": float(np.mean(list(fused.values()))), "separate": sep, "fused": fused,
                           "fused_steps": steps_used, "median_tokens": med,
                           "branch_token_layers_per_message": {"separate": 2 * med * np.mean([len(g) for g in groups]),
                                                               "fused": 2 * med},
                           "s": round(time.time() - t0, 1)}
    log(f"   fused blocks@{args.split}+2: separate {out['fused_blocks']['separate_mean_acc']:.4f} vs fused "
        f"{out['fused_blocks']['fused_mean_acc']:.4f} ({time.time() - t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=["typed-decisions", "clinc150"])
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--ranks", type=int, nargs="*", default=[2, 4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--block-epochs", type=int, default=6)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--no-blocks", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.min_steps, args.block_epochs, args.ranks = 30, 12, 1, [2, 8]
    out_path = args.out or ("results/tarski/explore_subspaces_smoke.json" if args.smoke
                            else "results/tarski/explore_subspaces.json")
    log = Logger(out_path)
    dev = device(args.smoke)
    trunk = Trunk(device="cpu" if args.smoke else None)
    log(f"== shared decision subspaces | trunk {trunk.device}, fits {dev} | split {args.split} | ranks {args.ranks}")
    res = {"args": vars(args), "datasets": {}}
    for name in args.datasets:
        t0 = time.time()
        ds = load_dataset(name, args.smoke)
        log(f"-- {ds.summary()[:160]}")
        res["datasets"][name] = run_dataset(trunk, ds, dev, args, log)
        res["datasets"][name]["wall_s"] = round(time.time() - t0, 1)
        json.dump(res, open(out_path, "w"), indent=1, default=float)
    log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
