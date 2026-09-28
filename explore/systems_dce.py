"""Idea 3: dead-code elimination in the frozen trunk, per decision set.

For a fixed set S of requested decisions (their branches all read the trunk at split k), score every
attention head and every group of 64 MLP channels in trunk layers [0, k) by how much the S-branches'
outputs depend on it. The score is the gradient magnitude of a gate at 1 (Michel et al. 2019), on a
label-free loss: cross-entropy of each branch against its own ungated prediction, over unlabelled
calibration messages. Zero the lowest-scoring fraction (zeroing a head's output or an MLP group is
exactly removing its weight rows/columns, since ModernBERT has no biases), keep branches untouched, and
cache the pruned trunk as the "plan" for S.

Tested prediction: at equal sparsity, a plan computed for S keeps S's decisions better than a plan
computed for a different decision set (other branches, other traffic), than weight-magnitude pruning,
and than random pruning. Two decision sets: CLINC150 (intent, domain, oos) and Banking77 (intent);
each is evaluated under its own plan and under the other's.

Usage:
  python explore/systems_dce.py --smoke
  python explore/systems_dce.py --out results/tarski/explore_dce.json
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explore.systems_common import (Log, all_texts, layer_flops, save, split_index, subset, threads, train_tasks)

import numpy as np
import torch
import torch.nn.functional as F

from tarski import data
from tarski.train import FeatureCache, autocast
from tarski.trunk import Trunk


class Gates:
    """Multiplicative gates on each head's output and each MLP channel group, for layers [0, k)."""

    def __init__(self, trunk: Trunk, k: int, group: int = 64):
        cfg = trunk.cfg
        self.k, self.H, self.hd = k, cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        self.G, self.gs = cfg.intermediate_size // group, group
        dev = trunk.device
        self.head = torch.ones(k, self.H, device=dev)
        self.mlp = torch.ones(k, self.G, device=dev)
        self.trunk = trunk

    @contextlib.contextmanager
    def active(self):
        hs = []
        for j in range(self.k):
            layer = self.trunk.model.layers[j]

            def attn_hook(mod, args, j=j):
                x = args[0]
                g = self.head[j].to(x.dtype).repeat_interleave(self.hd)
                return (x * g,)

            def mlp_hook(mod, args, j=j):
                x = args[0]
                g = self.mlp[j].to(x.dtype).repeat_interleave(self.gs)
                return (x * g,)
            hs.append(layer.attn.Wo.register_forward_pre_hook(attn_hook))
            hs.append(layer.mlp.Wo.register_forward_pre_hook(mlp_hook))
        try:
            yield self
        finally:
            for h in hs:
                h.remove()


def trunk_to(trunk: Trunk, ids, att, k: int):
    """Trunk states after k layers, differentiable w.r.t. the gates (the stock `taps` is no_grad)."""
    h = trunk.model.embeddings(input_ids=ids)
    ctx = trunk.context(h, att)
    for j in range(k):
        layer = trunk.model.layers[j]
        h = layer(h, attention_mask=ctx.masks[layer.attention_type], position_embeddings=ctx.rope[layer.attention_type])
    return h, ctx


def batches(trunk: Trunk, texts: List[str], max_len: int, bs: int):
    ids = trunk.token_ids(texts, max_len)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(ids[i]) for i in idx)
        x = torch.full((len(idx), L), trunk.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            x[j, : len(ids[i])] = torch.tensor(ids[i])
            att[j, : len(ids[i])] = 1
        yield idx, x.to(trunk.device), att.to(trunk.device)


def importance(trunk: Trunk, gates: Gates, branches: Dict, texts: List[str], max_len: int, k: int, bs: int):
    """Accumulated |d loss / d gate| with pseudo-labels from the ungated trunk."""
    gh, gm = torch.zeros_like(gates.head), torch.zeros_like(gates.mlp)
    for br in branches.values():
        br.eval()
    for _, x, att in batches(trunk, texts, max_len, bs):
        with torch.no_grad():
            h0, ctx0 = trunk_to(trunk, x, att, k)
            pseudo = {t: br.logits(h0, ctx0).argmax(-1) for t, br in branches.items()}
        gates.head.requires_grad_(True)
        gates.mlp.requires_grad_(True)
        with gates.active():
            h, ctx = trunk_to(trunk, x, att, k)
            loss = sum(F.cross_entropy(br.logits(h, ctx).float(), pseudo[t]) for t, br in branches.items())
        g_head, g_mlp = torch.autograd.grad(loss, [gates.head, gates.mlp])
        gh += g_head.abs()
        gm += g_mlp.abs()
        gates.head.requires_grad_(False)
        gates.mlp.requires_grad_(False)
    return gh, gm


def magnitude(trunk: Trunk, k: int, gates: Gates):
    H, hd, G, gs = gates.H, gates.hd, gates.G, gates.gs
    sh, sm = torch.zeros(k, H), torch.zeros(k, G)
    for j in range(k):
        l = trunk.model.layers[j]
        Wqkv = l.attn.Wqkv.weight.detach().float().cpu().view(3, H, hd, -1)
        Wo = l.attn.Wo.weight.detach().float().cpu().view(-1, H, hd)
        sh[j] = Wqkv.pow(2).sum((0, 2, 3)).sqrt() * Wo.pow(2).sum((0, 2)).sqrt()
        Wi = l.mlp.Wi.weight.detach().float().cpu()
        I = Wi.shape[0] // 2
        wi = (Wi[:I].pow(2).sum(1) + Wi[I:].pow(2).sum(1)).view(G, gs).sum(1).sqrt()
        wo = l.mlp.Wo.weight.detach().float().cpu().pow(2).sum(0).view(G, gs).sum(1).sqrt()
        sm[j] = wi * wo
    return sh, sm


def masks_from(score_h: torch.Tensor, score_m: torch.Tensor, sparsity: float):
    """Zero the lowest `sparsity` fraction of heads and of MLP groups (scores normalised per layer),
    keeping at least one head and one MLP group per layer."""
    def one(s):
        s = s.float().cpu()
        s = s / s.norm(dim=1, keepdim=True).clamp_min(1e-12)
        flat = s.flatten()
        n = int(round(sparsity * flat.numel()))
        m = torch.ones_like(flat)
        if n > 0:
            m[flat.argsort()[:n]] = 0
        m = m.view_as(s)
        for j in range(m.shape[0]):
            if m[j].sum() == 0:
                m[j, s[j].argmax()] = 1
        return m
    return one(score_h), one(score_m)


@torch.no_grad()
def eval_plan(trunk: Trunk, gates: Gates, mh, mm, branches: Dict, texts: List[str], ys: Dict, ref: Dict,
              max_len: int, k: int, bs: int):
    gates.head.copy_(mh.to(gates.head.device))
    gates.mlp.copy_(mm.to(gates.mlp.device))
    preds = {t: np.zeros(len(texts), int) for t in branches}
    with gates.active(), autocast(trunk.device):
        for idx, x, att in batches(trunk, texts, max_len, bs):
            h, ctx = trunk_to(trunk, x, att, k)
            for t, br in branches.items():
                preds[t][idx] = br.logits(h, ctx).float().argmax(-1).cpu().numpy()
    gates.head.fill_(1.0)
    gates.mlp.fill_(1.0)
    out = {t: {"acc": float((preds[t] == ys[t]).mean()),
               "agree": float((preds[t] == ref[t]).mean()) if ref else 1.0} for t in branches}
    return preds, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs=2, default=["clinc150", "banking77"])
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--kind", default="probe", help="probe | blocks:D")
    ap.add_argument("--sparsities", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--calib", type=int, default=1024, help="unlabelled calibration messages per decision set")
    ap.add_argument("--max-test", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.split, a.sparsities, a.calib, a.max_test, a.epochs, a.min_steps, a.bs = \
            "cpu", 4, [0.2, 0.5], 32, 40, 1, 5, 16
    if not a.smoke and not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    trunk = Trunk(device=a.device)
    k = a.split
    gates = Gates(trunk, k)
    res = {"split": k, "kind": a.kind, "sets": {}, "device": str(trunk.device)}
    D, I = trunk.hidden, trunk.cfg.intermediate_size

    sets = {}
    for name in a.datasets:
        ds = data.load(name)
        if a.smoke:
            ds = subset(ds, list(ds.tasks), 120, 40, 40, stride=37)
        cache = FeatureCache(trunk, all_texts(ds), [k], ds.max_len)
        log(f"== {ds.summary()} | cached depth {k} in {cache.seconds:.1f}s")
        br = train_tasks(trunk, ds, cache, list(ds.tasks), a.kind, k, seed=a.seed, epochs=a.epochs,
                         min_steps=a.min_steps, log=log)
        rng = np.random.default_rng(a.seed)
        calib = [ds.train[i].text for i in rng.permutation(len(ds.train))[: a.calib]]
        test = ds.test[: a.max_test] if not a.smoke else ds.test
        sets[name] = {"ds": ds, "branches": {t: v["branch"].to(trunk.device).eval() for t, v in br.items()},
                      "calib": calib, "test_texts": [e.text for e in test],
                      "ys": {t: np.array([e.y[t] for e in test]) for t in ds.tasks},
                      "full_acc": {t: v["metrics"]["acc"] for t, v in br.items()}}
        del cache
    L_med = int(np.median([len(x) for s in sets.values() for x in trunk.token_ids(s["test_texts"][:500])]))

    # plans: importance per decision set, magnitude, random
    plans = {}
    for name, s in sets.items():
        t0 = time.time()
        plans[f"importance[{name}]"] = importance(trunk, gates, s["branches"], s["calib"], s["ds"].max_len, k, a.bs)
        log(f"  importance for decision set {name}: {time.time() - t0:.1f}s")
    plans["magnitude"] = magnitude(trunk, k, gates)
    g = torch.Generator().manual_seed(a.seed)
    plans["random"] = (torch.rand(k, gates.H, generator=g), torch.rand(k, gates.G, generator=g))
    # how different are the two sets' plans? (overlap of the pruned heads at 30%)
    names = list(sets)
    mh0, _ = masks_from(*plans[f"importance[{names[0]}]"], 0.3)
    mh1, _ = masks_from(*plans[f"importance[{names[1]}]"], 0.3)
    pruned0, pruned1 = (mh0 == 0), (mh1 == 0)
    res["plan_overlap_heads_at_30pct"] = float((pruned0 & pruned1).sum() / max(1, pruned0.sum()))
    log(f"  overlap of pruned heads between the two sets' plans at 30%: {res['plan_overlap_heads_at_30pct']:.2f}")

    for name, s in sets.items():
        ones_h, ones_m = torch.ones_like(gates.head).cpu(), torch.ones_like(gates.mlp).cpu()
        ref, base = eval_plan(trunk, gates, ones_h, ones_m, s["branches"], s["test_texts"], s["ys"], None,
                              s["ds"].max_len, k, a.bs)
        entry = {"unpruned": {t: v["acc"] for t, v in base.items()}, "plans": {}}
        log(f"-- decision set {name}: unpruned acc " + " ".join(f"{t}={v['acc']:.4f}" for t, v in base.items()))
        for pname, (sh, sm) in plans.items():
            label = "own" if pname == f"importance[{name}]" else ("other_set" if pname.startswith("importance") else pname)
            rows = {}
            for sp in a.sparsities:
                mh, mm = masks_from(sh, sm, sp)
                _, r = eval_plan(trunk, gates, mh, mm, s["branches"], s["test_texts"], s["ys"], ref,
                                 s["ds"].max_len, k, a.bs)
                # FLOPs kept in trunk layers [0, k): heads share QKV/O/attention, groups share the MLP
                att = (1 - mh.mean()).item() * (2 * L_med * 4 * D * D + 4 * L_med * L_med * D)
                mlp = (1 - mm.mean()).item() * (2 * L_med * 3 * D * I)
                full = 2 * L_med * 4 * D * D + 4 * L_med * L_med * D + 2 * L_med * 3 * D * I
                kept = 1 - (att + mlp) / full
                rows[sp] = {"mean_acc": float(np.mean([v["acc"] for v in r.values()])),
                            "mean_agree": float(np.mean([v["agree"] for v in r.values()])),
                            "per_task": r, "trunk_flops_kept": kept}
                log(f"   plan {label:10s} sparsity {sp:.1f}: acc {rows[sp]['mean_acc']:.4f} agree "
                    f"{rows[sp]['mean_agree']:.4f} trunk FLOPs kept {kept:.2f}")
            entry["plans"][label] = rows
        res["sets"][name] = entry
        save(res, a.out)

    log("== summary: mean decision agreement with the unpruned trunk (acc) by sparsity")
    for name, e in res["sets"].items():
        for label, rows in e["plans"].items():
            log(f"  {name:10s} {label:10s} " + " ".join(f"{sp:.1f}:{r['mean_agree']:.3f}({r['mean_acc']:.3f})"
                                                     for sp, r in rows.items()))
    save(res, a.out)
    log("done")


if __name__ == "__main__":
    main()
