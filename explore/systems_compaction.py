"""Idea 7: projection pushdown at the fork — compact the tokens once, before every branch.

At split k each message's L token states are compacted to R tokens once; all N block branches then run
on R tokens, so each decision's branch costs ~R/L of the uncompacted one. Compactions (all training-free,
computed once per message and cached):

  full        no compaction (the blocks:2 baseline)
  tome<R>     ToMe-style bipartite soft matching on the depth-k states: merge the most similar pairs
              (size-weighted averages) until R tokens remain; CLS is never merged. Branches use
              proportional attention (log token size added to the attention logits), size-weighted mean
              pooling, and each merged token's size-weighted mean position for RoPE and the local window.
  values      drop the JSON structure tokens (keys and punctuation; see idea 4) and keep CLS, SEP and values
  stride<R>   keep CLS plus R-1 evenly spaced tokens (a control for "fewer tokens" without choosing well)

Branches are trained and evaluated on the compacted states (the compaction is part of the plan).

Tested prediction: tome64 stays within ~1 point of full blocks:2 at ~4x lower branch FLOPs; values beats
stride at a similar token count.

Usage:
  python explore/systems_compaction.py --smoke
  python explore/systems_compaction.py --out results/tarski/explore_compaction.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explore.systems_common import (Log, all_texts, json_structure, layer_flops, save, subset, threads,
                                    train_tasks)

import numpy as np
import torch
import torch.nn.functional as F

from tarski import data
from tarski.train import FeatureCache
from tarski.trunk import Context, Trunk


def tome(h: torch.Tensor, r: int):
    """Bipartite soft matching down to r tokens. h: (L, D). Returns states, sizes, positions."""
    x = h.float()
    size = torch.ones(x.shape[0], device=x.device)
    pos = torch.arange(x.shape[0], device=x.device, dtype=torch.float)
    while x.shape[0] > r:
        n = x.shape[0]
        idx = torch.arange(1, n, device=x.device)                             # CLS (0) is never merged
        A, B = idx[0::2], idx[1::2]
        if len(B) == 0:
            break
        m = min(n - r, len(A))
        xn = F.normalize(x, dim=-1)
        best, bj = (xn[A] @ xn[B].T).max(-1)
        top = best.argsort(descending=True)[:m]
        src, dst = A[top], B[bj[top]]
        xw, pw = x * size[:, None], pos * size
        xw.index_add_(0, dst, xw[src])
        pw.index_add_(0, dst, pw[src])
        new_size = size.clone()
        new_size.index_add_(0, dst, size[src])
        keep = torch.ones(n, dtype=torch.bool, device=x.device)
        keep[src] = False
        x, pos, size = (xw / new_size[:, None])[keep], (pw / new_size)[keep], new_size[keep]
    return x, size, pos.round().long()


class CompactCache:
    """FeatureCache interface over compacted states: h (R_i, D), sizes, positions per message."""

    def __init__(self, trunk: Trunk, depth: int, hs, sizes, poss, window: int):
        self.trunk, self.depth, self.window = trunk, depth, window
        self.h = {depth: [t.to("cpu", torch.float16) for t in hs]}
        self.sizes, self.poss = [s.float().cpu() for s in sizes], [p.long().cpu() for p in poss]
        self.lengths = [t.shape[0] for t in hs]

    def batch(self, depth: int, idx: Sequence[int], dtype=torch.float32):
        dev, D = self.trunk.device, self.trunk.hidden
        R = max(self.lengths[i] for i in idx)
        h = torch.zeros(len(idx), R, D, dtype=torch.float16)
        size = torch.zeros(len(idx), R)
        pos = torch.zeros(len(idx), R, dtype=torch.long)
        for j, i in enumerate(idx):
            n = self.lengths[i]
            h[j, :n], size[j, :n], pos[j, :n] = self.h[depth][i], self.sizes[i], self.poss[i]
            if n < R:
                pos[j, n:] = pos[j, n - 1]
        h, size, pos = h.to(dev, dtype), size.to(dev), pos.to(dev)
        valid = size > 0
        bias = torch.log(size.clamp_min(1.0)).to(dtype)
        neg = torch.finfo(dtype).min
        allow_full = valid[:, None, :].expand(-1, R, -1)
        allow_slide = allow_full & ((pos[:, :, None] - pos[:, None, :]).abs() <= self.window)
        masks = {}
        for name, allow in (("full_attention", allow_full), ("sliding_attention", allow_slide)):
            allow = allow.clone()
            allow[~valid] = False
            allow[..., 0] |= ~valid                                           # pad queries see CLS (no NaN)
            m = torch.where(allow, bias[:, None, :].expand(-1, R, -1), torch.full_like(allow, neg, dtype=dtype))
            masks[name] = m[:, None]
        rope = {t: self.trunk.model.rotary_emb(h, pos, t) for t in set(self.trunk.cfg.layer_types)}
        return h, Context(masks, rope, size.to(dtype))                        # size-weighted mean pooling

    def bytes(self):
        return sum(t.numel() * 2 for t in self.h[self.depth])


def build(trunk: Trunk, fc: FeatureCache, k: int, how: str, struct: List[List[bool]]):
    hs, sizes, poss = [], [], []
    dev = trunk.device
    for i, h in enumerate(fc.h[k]):
        L = h.shape[0]
        if how.startswith("tome"):
            r = int(how[4:])
            x, s, p = tome(h.to(dev), r) if L > r else (h.float(), torch.ones(L), torch.arange(L))
        elif how == "values":
            keep = torch.tensor([not s_ for s_ in struct[i][:L]])
            keep[0] = True
            p = torch.nonzero(keep).squeeze(1)
            x, s = h[p].float(), torch.ones(len(p))
        elif how.startswith("stride"):
            r = int(how[6:])
            p = torch.unique(torch.linspace(0, L - 1, min(r, L)).round().long()) if L > r else torch.arange(L)
            x, s = h[p].float(), torch.ones(len(p))
        else:
            raise ValueError(how)
        hs.append(x.cpu())
        sizes.append(s.cpu())
        poss.append(p.cpu())
    return CompactCache(trunk, k, hs, sizes, poss, trunk.cfg.sliding_window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--configs", nargs="*", default=["full", "tome128", "tome64", "tome32", "values", "stride64"])
    ap.add_argument("--workflows", nargs="*", default=["customer_service", "security_incidents"])
    ap.add_argument("--kind", default="blocks:2")
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.split, a.min_steps, a.configs = "cpu", 4, 5, ["full", "tome32", "values", "stride32"]
    if not a.smoke and not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    trunk = Trunk(device=a.device)
    ds = data.load("typed-decisions")
    tasks = [t for t in ds.tasks if t.split(".")[0] in a.workflows]
    if a.smoke:
        ds = subset(ds, tasks[:2], 40, 30, 24, max_len=256)
        tasks = list(ds.tasks)
    else:                                            # messages of the chosen workflows (any of their tasks)
        keep = lambda xs: [e for e in xs if any(t in e.y for t in tasks)]
        ds = data.Dataset(ds.name, {t: ds.tasks[t] for t in tasks}, keep(ds.train), keep(ds.val), keep(ds.test),
                          ds.max_len)
    k, texts = a.split, all_texts(ds)
    wf = [next(iter(e.y)).split(".")[0] for e in ds.train + ds.val + ds.test]
    struct = [json_structure(trunk.tok, t, ds.max_len, w)[1] for t, w in zip(texts, wf)]
    fc = FeatureCache(trunk, texts, [k], ds.max_len)
    L_mean = float(np.mean(fc.lengths))
    log(f"== {ds.summary()} | split {k} | {a.kind} | mean tokens {L_mean:.0f} | cached in {fc.seconds:.1f}s")
    # sanity: the compact path with no compaction (all sizes 1, true positions) matches the plain batch
    probe_idx = list(range(min(4, len(texts))))
    cc = CompactCache(trunk, k, [fc.h[k][i].float() for i in probe_idx], [torch.ones(fc.lengths[i]) for i in probe_idx],
                      [torch.arange(fc.lengths[i]) for i in probe_idx], trunk.cfg.sliding_window)
    from explore.systems_common import make
    br = make(a.kind, k, ["a", "b"], trunk).to(trunk.device).eval()
    with torch.no_grad():
        z1 = br.logits(*fc.batch(k, probe_idx))
        z2 = br.logits(*cc.batch(k, probe_idx))
    err = float((z1 - z2).abs().max())
    log(f"  sanity: uncompacted CompactCache vs FeatureCache logits max abs diff {err:.2e}")
    assert err < 1e-3, "compact path does not reproduce the plain branch"
    res = {"split": k, "kind": a.kind, "tasks": tasks, "mean_tokens": L_mean, "sanity_abs_err": err, "configs": {}}
    D, I, d = trunk.hidden, trunk.cfg.intermediate_size, int(a.kind.split(":")[1]) if ":" in a.kind else 0
    for how in a.configs:
        t0 = time.time()
        cache = fc if how == "full" else build(trunk, fc, k, how, struct)
        R = float(np.mean(cache.lengths))
        r = train_tasks(trunk, ds, cache, tasks, a.kind, k, seed=a.seed, min_steps=a.min_steps, log=lambda s: None)
        acc = float(np.mean([v["metrics"]["acc"] for v in r.values()]))
        f1 = float(np.mean([v["metrics"]["macro_f1"] for v in r.values()]))
        gfl = d * float(np.mean([layer_flops(D, I, n) for n in cache.lengths])) / 1e9
        res["configs"][how] = {"mean_acc": acc, "mean_macro_f1": f1, "mean_tokens": R,
                               "branch_gflops_per_decision": gfl,
                               "per_task": {t: v["metrics"]["acc"] for t, v in r.items()}, "wall_s": round(time.time() - t0, 1)}
        log(f"-- {how:9s}: acc {acc:.4f} F1 {f1:.4f} | tokens {R:.0f} | branch GFLOP/decision {gfl:.2f} | "
            f"{time.time() - t0:.0f}s")
        save(res, a.out)
    log("done")


if __name__ == "__main__":
    main()
