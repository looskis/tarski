"""Merge independently trained block branches into one shared body.

Every blocks branch at split k starts as a copy of the same base layers [k, k+d), so after training each
one is the base plus a task vector. Merging those vectors (mean, or TIES: trim, elect sign, disjoint
mean) gives ONE body that all tasks run through, so N decisions cost d layers instead of N x d.
Each task then keeps a linear head: either its own trained head as is, or one refit on the merged body
(seconds; no layer is retrained).

Compared per task: independent branches (the baseline, N x d layers), merged + own head, merged + refit
head, and a body trained jointly on all tasks (d layers, but tasks are no longer independent: this is
the multi-task upper reference that loses isolation).

Usage:
  python explore/merge_branches.py --smoke
  python explore/merge_branches.py --dataset clinc150 --split 11 --out results/tarski/explore_merge_clinc150.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tarski import data, train
from tarski.branches import Branch, mean_pool
from tarski.trunk import Trunk


def body_state(br) -> dict:
    return {k: v.detach().float().cpu() for k, v in br.state_dict().items() if k.startswith(("layers.", "norm."))}


def merge(base: dict, tuned: list, method: str, lam: float = 1.0, keep: float = 0.2) -> dict:
    out = {}
    for k in base:
        tv = torch.stack([t[k] - base[k] for t in tuned])                     # (N, ...)
        if method == "mean":
            m = tv.mean(0)
        elif method == "ties":
            flat = tv.flatten(1).abs()
            n_keep = max(1, int(keep * flat.shape[1]))
            thr = flat.kthvalue(flat.shape[1] - n_keep + 1, dim=1).values       # per-task magnitude cut
            trimmed = tv * (tv.abs() >= thr.view(-1, *([1] * (tv.dim() - 1))))
            sign = torch.sign(trimmed.sum(0))
            agree = (torch.sign(trimmed) == sign) & (trimmed != 0)
            m = (trimmed * agree).sum(0) / agree.sum(0).clamp_min(1)
        else:
            raise ValueError(method)
        out[k] = base[k] + lam * m
    return out


@torch.no_grad()
def pooled_features(body_layers, body_norm, fc: train.FeatureCache, split: int, idx, bs: int = 128):
    feats = torch.zeros(len(idx), fc.trunk.hidden)
    order = sorted(range(len(idx)), key=lambda j: fc.lengths[idx[j]])
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        h, ctx = fc.batch(split, [idx[j] for j in sel])
        h = body_norm(ctx.run(body_layers, h))
        feats[sel] = mean_pool(h, ctx.attention_mask).float().cpu()
    return feats


def fit_head(xtr, ytr, xva, yva, n_labels, device, steps=400, lr=1e-2, wd=1e-4, init=None):
    lin = nn.Linear(xtr.shape[1], n_labels)
    if init is not None:
        lin.load_state_dict(init)
    lin = lin.to(device)
    xtr, ytr = xtr.to(device), ytr.to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        loss = F.cross_entropy(lin(xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lin.cpu()


def evaluate_head(lin, xva, yva, xte, yte):
    with torch.no_grad():
        zva, zte = lin(xva), lin(xte)
    t = train.fit_temperature(zva, yva) if len(yva) >= 30 else 1.0
    return train.evaluate(torch.softmax(zte / t, -1).numpy(), yte.numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="clinc150")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or ("results/tarski/explore_merge_smoke.json" if a.smoke else "results/tarski/explore_merge.json")
    if a.smoke:
        a.device, a.epochs = "cpu", 1

    trunk = Trunk(device=a.device)
    ds = data.load(a.dataset)
    if a.smoke:
        ds.train, ds.val, ds.test = ds.train[:600], ds.val[:200], ds.test[:300]
    tasks = a.tasks or list(ds.tasks)
    log = lambda s: print(s, flush=True)
    k, d, dev = a.split, a.depth, trunk.device
    res = {"dataset": a.dataset, "split": k, "depth": d, "tasks": tasks, "per_task": {}}

    # 1. independent branches, saved so the task vectors can be read back
    cache = {}
    with tempfile.TemporaryDirectory() as store:
        t0 = time.time()
        indep = train.fit(trunk, ds, tasks, kind="blocks", split=k, depth=d, epochs=a.epochs, seed=a.seed,
                          store=store, cache=cache, log=lambda s: None)
        res["independent_train_s"] = time.time() - t0
        branches = {t: Branch.load(os.path.join(store, t), trunk) for t in tasks}
    fc = cache[(ds.name, k)]
    base_layers = trunk.copy_layers(k, k + d)
    base = {**{f"layers.{n}": p.detach().float().cpu() for n, p in base_layers.state_dict().items()},
            **{f"norm.{n}": p.detach().float().cpu() for n, p in trunk.copy_final_norm().state_dict().items()}}
    tuned = [body_state(branches[t]) for t in tasks]
    for t in tasks:
        res["per_task"][t] = {"independent": indep[t]["acc"]}
        log(f"[{t}] independent acc {indep[t]['acc']:.4f}")

    n_tr, n_va = len(ds.train), len(ds.val)
    allx = ds.train + ds.val + ds.test
    splits = {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va), "test": range(n_tr + n_va, len(allx))}

    def labelled(task, s):
        idx = [i for i in splits[s] if task in allx[i].y]
        return idx, torch.tensor([allx[i].y[task] for i in idx])

    def run_body(sd, tag):
        layers, norm = trunk.copy_layers(k, k + d), trunk.copy_final_norm()
        layers.load_state_dict({n[len("layers."):]: v for n, v in sd.items() if n.startswith("layers.")})
        norm.load_state_dict({n[len("norm."):]: v for n, v in sd.items() if n.startswith("norm.")})
        layers.to(dev).eval(), norm.to(dev).eval()
        allidx = list(range(len(allx)))
        feats = pooled_features(layers, norm, fc, k, allidx)
        for t in tasks:
            (tr, ytr), (va, yva), (te, yte) = labelled(t, "train"), labelled(t, "val"), labelled(t, "test")
            own = {"weight": branches[t].out.weight.detach().float().cpu(), "bias": branches[t].out.bias.detach().float().cpu()}
            lin_own = nn.Linear(trunk.hidden, len(ds.tasks[t].labels))
            lin_own.load_state_dict(own)
            m_own = evaluate_head(lin_own, feats[va], yva, feats[te], yte)
            lin_fit = fit_head(feats[tr], ytr, feats[va], yva, len(ds.tasks[t].labels), dev, init=own)
            m_fit = evaluate_head(lin_fit, feats[va], yva, feats[te], yte)
            res["per_task"][t][f"{tag}+own_head"] = m_own["acc"]
            res["per_task"][t][f"{tag}+refit_head"] = m_fit["acc"]
            log(f"[{t}] {tag}: own head {m_own['acc']:.4f}, refit head {m_fit['acc']:.4f}")

    # 2. merged bodies
    for method, lam in (("mean", 1.0), ("ties", 1.0), ("ties", 0.5)):
        run_body(merge(base, tuned, method, lam), f"{method}@{lam}")
    run_body(base, "base_copy")          # the unmerged starting point: a probe on base layers k..k+d

    # 3. joint multi-task body (loses isolation; upper reference for one shared body)
    torch.manual_seed(a.seed)
    layers, norm = trunk.copy_layers(k, k + d).to(dev), trunk.copy_final_norm().to(dev)
    heads = nn.ModuleDict({t.replace(".", "_"): nn.Linear(trunk.hidden, len(ds.tasks[t].labels)) for t in tasks}).to(dev)
    params = [{"params": list(layers.parameters()), "lr": 1e-4},
              {"params": list(norm.parameters()) + list(heads.parameters()), "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    tr_idx = [i for i in splits["train"] if any(t in allx[i].y for t in tasks)]
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    for ep in range(a.epochs):
        order = sorted(tr_idx, key=lambda i: fc.lengths[i] + rng.random() * 8)
        chunks = [order[i:i + 32] for i in range(0, len(order), 32)]
        rng.shuffle(chunks)
        for sel in chunks:
            h, ctx = fc.batch(k, sel)
            pooled = mean_pool(norm(ctx.run(layers, h)), ctx.attention_mask).float()
            loss = 0.0
            for t in tasks:
                m = torch.tensor([t in allx[i].y for i in sel], device=dev)
                if m.any():
                    yt = torch.tensor([allx[i].y.get(t, 0) for i in sel], device=dev)
                    loss = loss + F.cross_entropy(heads[t.replace(".", "_")](pooled[m]), yt[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
    res["joint_train_s"] = time.time() - t0
    layers.eval(), norm.eval()
    feats = pooled_features(layers, norm, fc, k, list(range(len(allx))))
    for t in tasks:
        (va, yva), (te, yte) = labelled(t, "val"), labelled(t, "test")
        m = evaluate_head(heads[t.replace(".", "_")].cpu(), feats[va], yva, feats[te], yte)
        res["per_task"][t]["joint"] = m["acc"]
        log(f"[{t}] joint multi-task body {m['acc']:.4f}")

    cols = list(next(iter(res["per_task"].values())))
    res["mean"] = {c: float(np.mean([v[c] for v in res["per_task"].values()])) for c in cols}
    log("mean: " + json.dumps({c: round(v, 4) for c, v in res["mean"].items()}))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
