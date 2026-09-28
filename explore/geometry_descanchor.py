"""Idea 12: description-anchored routes. Probe weights from label descriptions, read off the shared trunk.

Each typed-decisions option ships a description (for example `"stop": "Halt the agent now."`, or
`"Critical; requires action within the same day."`). The option text "<question> Answer: <description>"
goes through the same frozen trunk once, and its mean-pooled state e_c at depth d is the option's anchor.
One scorer is shared by every decision:

    score(x, c) = (U^T x) . (V^T e_c) + w . e_c        U, V: 768 x r,  w: 768

Here x is the message's mean-pooled state at depth d, and all states are standardized. A new route
costs one encoding per option description and one dot product per option on the shared trunk pass.

Settings:
  in-distribution          train the shared scorer on all 20 tasks (train split) and test on the test
                           split. One scorer for all routes, against per-task logistic probes on the same
                           states.
  leave-one-workflow-out   train on the 15 tasks of three workflows; score the 5 tasks of the held-out
                           workflow zero-shot (no labels, new schema, new questions).
Baselines: uniform guessing (1/C), the held-out task's majority label (an oracle prior that uses its
labels), untrained cosine similarity between x and e_c, and per-task logistic probes (in-distribution).

Prediction (geometry.md idea 12): in-distribution within ~3 points of per-task probes; zero-shot above
uniform on score and yes/no questions, weak on choice questions.

Usage:
  .venv/bin/python explore/geometry_descanchor.py --smoke
  .venv/bin/python explore/geometry_descanchor.py --out results/tarski/explore_descanchor.json   # ~3 min on an A10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_common import (Logger, acc, device, fit_logreg, labels, load_dataset, pick_by_val, pooled, splits,
                             standardize, targets, task_split)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tarski.autosplit import pooled_by_depth
from tarski.trunk import Trunk


def option_texts() -> dict:
    """task -> list of option texts, in tarski's label order (tarski.data._option_keys)."""
    from datasets import load_dataset as hf_load
    seen = {}
    for r in hf_load("LocalLLaMA/typed-decisions", "all", split="train"):
        if r["workflow"] not in seen:
            seen[r["workflow"]] = json.loads(r["questions"])
    out = {}
    for w, qs in seen.items():
        for qid, q in qs.items():
            ins, crit = q.get("instructions", ""), q.get("criteria")
            if q["type"] == "choice":
                opts = [f"{ins} Answer: {k.replace('_', ' ')}: {v}" for k, v in crit.items()]
            elif q["type"] == "score":
                opts = [f"{ins} Answer: {v}" for v in crit]
            else:
                crit = crit or {"false": f"No. It is not the case that: {ins}", "true": f"Yes. {ins}"}
                opts = [f"{ins} Answer: {'no' if k == 'false' else 'yes'}: {crit[k]}" for k in ("false", "true")]
            out[f"{w}.{qid}"] = opts
    return out


def qtype(ds, t):
    n = len(ds.tasks[t].labels)
    keys = ds.tasks[t].labels
    return "noul" if keys == ["false", "true"] else ("score" if keys[0] == "0" else "choice")


class Scorer(nn.Module):
    def __init__(self, D, r):
        super().__init__()
        self.U = nn.Parameter(torch.randn(D, r) / D ** 0.5)
        self.V = nn.Parameter(torch.randn(D, r) / D ** 0.5)
        self.w = nn.Parameter(torch.zeros(D))

    def forward(self, X, E):                       # X (n, D), E (C, D) -> (n, C)
        return (X @ self.U) @ (E @ self.V).T + (E @ self.w)[None]


def train_scorer(X, E, allx, ds, tasks, n_tr, n_va, dev, r, steps, l2=1e-3, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    model = Scorer(X.shape[1], r).to(dev)
    data = []
    for t in tasks:
        tr, _, _ = task_split(allx, t, n_tr, n_va)
        data.append((X[tr].to(dev), E[t].to(dev), targets(allx, tr, t, len(ds.tasks[t].labels)).to(dev)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = sum(-(T * F.log_softmax(model(Xt, Et), -1)).sum(-1).mean() for Xt, Et, T in data) / len(data)
        loss = loss + l2 * sum(p.pow(2).sum() for p in model.parameters())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, X, E, allx, ds, tasks, n_tr, n_va, dev, split="test"):
    out = {}
    for t in tasks:
        tr, va, te = task_split(allx, t, n_tr, n_va)
        idx = te if split == "test" else va
        z = model(X[idx].to(dev), E[t].to(dev)).cpu()
        out[t] = acc(z, labels(allx, idx, t))
    return out


def by_type(ds, per):
    res = {}
    for t, a in per.items():
        res.setdefault(qtype(ds, t), []).append(a)
    return {k: float(np.mean(v)) for k, v in res.items()} | {"all": float(np.mean(list(per.values())))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--depths", type=int, nargs="*", default=[11, 22])
    ap.add_argument("--ranks", type=int, nargs="*", default=[8, 32, 128])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.depths, args.ranks, args.steps = [11], [8], 40
    out_path = args.out or ("results/tarski/explore_descanchor_smoke.json" if args.smoke
                            else "results/tarski/explore_descanchor.json")
    log = Logger(out_path)
    dev = device(args.smoke)
    trunk = Trunk(device="cpu" if args.smoke else None)
    ds = load_dataset("typed-decisions", args.smoke)
    allx, n_tr, n_va = splits(ds)
    tasks = list(ds.tasks)
    wf = {t: t.split(".")[0] for t in tasks}
    opts = option_texts()
    assert all(len(opts[t]) == len(ds.tasks[t].labels) for t in tasks)
    log(f"== description-anchored routes | trunk {trunk.device}, fits {dev} | {ds.summary()[:90]} | "
        f"{sum(len(v) for v in opts.values())} option descriptions")
    feats = pooled(trunk, ds, args.depths)
    flat = [(t, j) for t in tasks for j in range(len(opts[t]))]
    efeats = pooled_by_depth(trunk, [opts[t][j] for t, j in flat], args.depths, 128)
    res = {"args": vars(args), "option_texts": {t: opts[t] for t in tasks}, "depths": {}}
    for d in args.depths:
        X = standardize(feats[d], list(range(n_tr)))
        Eall = standardize(efeats[d], list(range(len(flat))))
        E = {t: Eall[[k for k, (tt, _) in enumerate(flat) if tt == t]] for t in tasks}
        r_d = {}
        # baselines
        uniform = {t: 1.0 / len(ds.tasks[t].labels) for t in tasks}
        major, cosine, probe = {}, {}, {}
        for t in tasks:
            tr, va, te = task_split(allx, t, n_tr, n_va)
            yte = labels(allx, te, t)
            major[t] = float((yte == np.bincount(labels(allx, tr, t), minlength=len(ds.tasks[t].labels)).argmax()).mean())
            cosine[t] = acc(F.normalize(X[te], dim=-1) @ F.normalize(E[t], dim=-1).T, yte)
            lr_ = fit_logreg(X[tr], targets(allx, tr, t, len(ds.tasks[t].labels)), [X[va], X[te]], dev, steps=300)
            probe[t] = acc(lr_["logits"][1][pick_by_val(lr_["logits"][0], labels(allx, va, t))], yte)
        r_d["baselines"] = {"uniform": by_type(ds, uniform), "majority_oracle_prior": by_type(ds, major),
                            "untrained_cosine": by_type(ds, cosine), "per_task_logreg": by_type(ds, probe)}
        log(f"-- depth {d} baselines: " + json.dumps({k: round(v["all"], 4) for k, v in r_d["baselines"].items()}))
        # in-distribution: one shared scorer for all 20 tasks
        r_d["in_distribution"] = {}
        for r in args.ranks:
            m = train_scorer(X, E, allx, ds, tasks, n_tr, n_va, dev, r, args.steps)
            per = evaluate(m, X, E, allx, ds, tasks, n_tr, n_va, dev)
            r_d["in_distribution"][r] = {"by_type": by_type(ds, per), "tasks": per}
            log(f"   in-distribution shared scorer r={r}: " + json.dumps({k: round(v, 4) for k, v in by_type(ds, per).items()}))
        # leave one workflow out: zero-shot routes on a new schema
        r_d["lowo"] = {}
        for r in args.ranks:
            per, per_major, per_cos, per_uni = {}, {}, {}, {}
            for w in sorted(set(wf.values())):
                train_tasks = [t for t in tasks if wf[t] != w]
                held = [t for t in tasks if wf[t] == w]
                m = train_scorer(X, E, allx, ds, train_tasks, n_tr, n_va, dev, r, args.steps)
                per.update(evaluate(m, X, E, allx, ds, held, n_tr, n_va, dev))
            r_d["lowo"][r] = {"by_type": by_type(ds, per), "tasks": per}
            log(f"   zero-shot (leave one workflow out) r={r}: " + json.dumps({k: round(v, 4) for k, v in by_type(ds, per).items()}))
        res["depths"][d] = r_d
        json.dump(res, open(out_path, "w"), indent=1, default=float)
    log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
