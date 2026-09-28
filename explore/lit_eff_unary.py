"""Lit-scan idea 8: unary-score token reduction instead of pairwise ToMe merging, at the fork and inside
the trunk (a separate arm next to explore/systems_compaction.py; its ToMe/CompactCache code is copied here,
that file is not modified).

CATIS (arXiv 2604.16745, vision) argues that training-free pairwise-similarity reduction (ToMe and
relatives) collapses at high compression because pairwise rankings become unstable in deep layers, and
that per-token (unary) scores do not. On typed-decisions JSON states (split 11, blocks:2, the two
workflows systems_compaction uses) we compare, keeping R tokens per message (CLS and the last token always
kept, original order and positions preserved):
  full          no reduction
  tome<R>       ToMe bipartite merging (copied from systems_compaction.py)
  stride<R>     evenly spaced tokens (control)
  cls<R>        attention the next base layer's CLS query pays each token (mean over heads)
  norm<R>       L2 norm of the depth-k state, massive channels excluded
  idf<R>        token-id rarity: log(N / document frequency) over the training messages (text only)
and an in-trunk variant that cuts trunk compute for every decision, not only branch compute:
  trunk<g>:<score><pct>   score tokens at depth g, keep pct% of them, run trunk layers [g, k) on the kept
                          tokens only (true positions for RoPE and the sliding window), then the branch.
Branches are trained and evaluated on the reduced states. Checks: the compact path with nothing removed
reproduces the plain branch; the in-trunk path with nothing removed reproduces the trunk's depth-k states.

Prediction (lit_efficiency.md entry 8): unary >= ToMe at R = 32 by several points; the in-trunk 50% cut
stays within ~1 point for probes, more for blocks.

Usage:
  .venv/bin/python explore/lit_eff_unary.py --smoke
  .venv/bin/python explore/lit_eff_unary.py --out results/tarski/explore_lit_unary.json
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_eff_common import Log, index, load_ds, make, save, task_rows, texts, threads, train_eval

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from tarski.train import FeatureCache, predict_logits
from tarski.trunk import Context, Trunk


# ---------------------------------------------------------------------------------------------------
# Copied from explore/systems_compaction.py (ToMe and the compacted-state cache)
# ---------------------------------------------------------------------------------------------------

def tome(h: torch.Tensor, r: int):
    """Bipartite soft matching down to r tokens. h: (L, D). Returns states, sizes, positions."""
    x = h.float()
    size = torch.ones(x.shape[0], device=x.device)
    pos = torch.arange(x.shape[0], device=x.device, dtype=torch.float)
    while x.shape[0] > r:
        n = x.shape[0]
        idx = torch.arange(1, n, device=x.device)
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
            allow[..., 0] |= ~valid
            m = torch.where(allow, bias[:, None, :].expand(-1, R, -1), torch.full_like(allow, neg, dtype=dtype))
            masks[name] = m[:, None]
        rope = {t: self.trunk.model.rotary_emb(h, pos, t) for t in set(self.trunk.cfg.layer_types)}
        return h, Context(masks, rope, size.to(dtype))


# ---------------------------------------------------------------------------------------------------
# Unary scores
# ---------------------------------------------------------------------------------------------------

@torch.no_grad()
def cls_attention(trunk: Trunk, h: torch.Tensor, layer: int) -> torch.Tensor:
    """Attention the CLS query of base layer `layer` pays each token (mean over heads). h: (L, D)."""
    lay = trunk.model.layers[layer]
    x = h.float().to(trunk.device)[None]
    a = lay.attn_norm(x)
    qkv = lay.attn.Wqkv(a)
    H = trunk.cfg.num_attention_heads
    hd = trunk.hidden // H
    L = x.shape[1]
    qkv = qkv.view(1, L, 3, H, hd)
    q, k = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2)
    pos = torch.arange(L, device=trunk.device)[None]
    cos, sin = trunk.model.rotary_emb(x, pos, lay.attention_type)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    s = (q[:, :, :1] @ k.transpose(-1, -2)) / math.sqrt(hd)        # (1, H, 1, L)
    return torch.softmax(s.float(), -1).mean(1)[0, 0].cpu()


def keep_top(scores: torch.Tensor, R: int) -> torch.Tensor:
    L = len(scores)
    if L <= R:
        return torch.arange(L)
    s = scores.clone().float()
    s[0] = s[-1] = float("inf")                                       # CLS and the final [SEP]
    return torch.sort(torch.topk(s, R).indices).values


def scores_for(how: str, trunk, h, ids, idf, keep_ch, layer):
    if how == "cls":
        return cls_attention(trunk, h, layer)
    if how == "norm":
        return h.float()[:, keep_ch].norm(dim=-1)
    if how == "idf":
        return torch.tensor([idf.get(t, max(idf.values())) for t in ids], dtype=torch.float)
    raise ValueError(how)


def build(trunk, fc, k, how, ids, idf, keep_ch):
    hs, sizes, poss = [], [], []
    for i, h in enumerate(fc.h[k]):
        L = h.shape[0]
        if how.startswith("tome"):
            r = int(how[4:])
            x, s, p = tome(h.to(trunk.device), r) if L > r else (h.float(), torch.ones(L), torch.arange(L))
        elif how.startswith("stride"):
            r = int(how[6:])
            p = torch.unique(torch.linspace(0, L - 1, min(r, L)).round().long()) if L > r else torch.arange(L)
            x, s = h[p].float(), torch.ones(len(p))
        else:
            score = "".join(c for c in how if c.isalpha())
            r = int("".join(c for c in how if c.isdigit()))
            p = keep_top(scores_for(score, trunk, h, ids[i], idf, keep_ch, k), r)
            x, s = h[p].float(), torch.ones(len(p))
        hs.append(x.cpu())
        sizes.append(s.cpu())
        poss.append(p.cpu())
    return CompactCache(trunk, k, hs, sizes, poss, trunk.cfg.sliding_window)


@torch.no_grad()
def in_trunk(trunk, fc_g, g, k, how, ids, idf, keep_ch, bs: int = 32, frac: float = None):
    """Score at depth g, keep a fraction of tokens, run trunk layers [g, k) on the kept tokens only."""
    hs, sizes, poss = [], [], []
    for i, h in enumerate(fc_g.h[g]):
        L = h.shape[0]
        R = max(2, int(round(frac * L)))
        p = keep_top(scores_for(how, trunk, h, ids[i], idf, keep_ch, g), R) if frac < 1 else torch.arange(L)
        hs.append(h[p].float())
        sizes.append(torch.ones(len(p)))
        poss.append(p)
    cc = CompactCache(trunk, g, hs, sizes, poss, trunk.cfg.sliding_window)
    out = [None] * len(hs)
    order = sorted(range(len(hs)), key=lambda j: cc.lengths[j])
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        h, ctx = cc.batch(g, idx)
        h = ctx.run(trunk.model.layers[g:k], h)
        for j, i in enumerate(idx):
            out[i] = h[j, : cc.lengths[i]].cpu()
    return CompactCache(trunk, k, out, sizes, poss, trunk.cfg.sliding_window)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--configs", nargs="*", default=["full", "tome64", "tome32", "stride32", "cls64", "cls32",
                                                      "norm32", "idf32", "trunk7:cls50", "trunk4:idf50"])
    ap.add_argument("--workflows", nargs="*", default=["customer_service", "security_incidents"])
    ap.add_argument("--kinds", nargs="*", default=["blocks:2"])
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.split, a.min_steps = "cpu", 5, 20
        a.configs = ["full", "tome32", "cls32", "norm32", "idf32", "trunk4:cls50"]
        a.kinds = ["probe", "blocks:2"]
        a.out = a.out or "results/tarski/explore_lit_unary_smoke.json"
    if not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    trunk = Trunk(device=a.device)
    ds = load_ds("typed-decisions", a.smoke, workflows=a.workflows, n_smoke=(40, 30, 24))
    allx, rng = index(ds)
    tasks = list(ds.tasks)[: (2 if a.smoke else None)]
    k = a.split
    gs = sorted({int(c.split(":")[0][5:]) for c in a.configs if c.startswith("trunk")})
    fc = FeatureCache(trunk, texts(ds), [k] + gs, ds.max_len)
    ids = trunk.token_ids(texts(ds), ds.max_len)
    N = len(rng["train"])
    df: Dict[int, int] = {}
    for i in rng["train"]:
        for t in set(ids[i]):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(N / c) for t, c in df.items()}
    chmax = torch.stack([fc.h[k][i].float().abs().amax(0) for i in rng["train"][:256]]).amax(0)
    keep_ch = torch.ones(trunk.hidden, dtype=torch.bool)
    keep_ch[torch.nonzero(chmax > 8 * chmax.median()).squeeze(1)] = False
    L_mean = float(np.mean(fc.lengths))
    log(f"== typed-decisions {a.workflows}: {len(allx)} messages, {len(tasks)} tasks, split {k}, mean tokens "
        f"{L_mean:.0f}, cached in {fc.seconds:.1f}s")
    # sanity 1: compact path with no compaction == plain branch
    probe_idx = list(range(min(4, len(allx))))
    cc = CompactCache(trunk, k, [fc.h[k][i].float() for i in probe_idx], [torch.ones(fc.lengths[i]) for i in probe_idx],
                      [torch.arange(fc.lengths[i]) for i in probe_idx], trunk.cfg.sliding_window)
    br = make("blocks:2", k, ["a", "b"], trunk).to(trunk.device).eval()
    with torch.no_grad():
        e1 = float((br.logits(*fc.batch(k, probe_idx)) - br.logits(*cc.batch(k, probe_idx))).abs().max())
    res = {"split": k, "tasks": tasks, "mean_tokens": L_mean, "checks": {"compact_vs_plain_abs_err": e1}, "configs": {}}
    if gs:
        from lit_eff_common import cache_from_states
        sub = cache_from_states(trunk, {gs[0]: [fc.h[gs[0]][i] for i in probe_idx]})
        full_path = in_trunk(trunk, sub, gs[0], k, "cls", [ids[i] for i in probe_idx], idf, keep_ch, frac=1.0)
        e2 = max(float((full_path.h[k][j].float() - fc.h[k][i].float()).abs().max() /
                       fc.h[k][i].float().abs().max()) for j, i in enumerate(probe_idx))
        res["checks"]["in_trunk_keep_all_rel_err"] = e2
    log(f"  checks: {res['checks']}")
    assert e1 < 1e-3, "compact path does not reproduce the plain branch"
    D, I = trunk.hidden, trunk.cfg.intermediate_size

    def lf(Lq):
        return 2 * Lq * (4 * D * D + 3 * D * I) + 4 * Lq * Lq * D
    for how in a.configs:
        t0 = time.time()
        if how == "full":
            cache, trunk_gfl = fc, k * float(np.mean([lf(n) for n in fc.lengths])) / 1e9
        elif how.startswith("trunk"):
            g = int(how.split(":")[0][5:])
            spec = how.split(":")[1]
            score = "".join(c for c in spec if c.isalpha())
            frac = int("".join(c for c in spec if c.isdigit())) / 100
            cache = in_trunk(trunk, fc, g, k, score, ids, idf, keep_ch, frac=frac)
            trunk_gfl = float(np.mean([g * lf(n) + (k - g) * lf(m) for n, m in zip(fc.lengths, cache.lengths)])) / 1e9
        else:
            cache = build(trunk, fc, k, how, ids, idf, keep_ch)
            trunk_gfl = k * float(np.mean([lf(n) for n in fc.lengths])) / 1e9
        R = float(np.mean(cache.lengths))
        row = {"mean_tokens_at_split": R, "trunk_gflops": trunk_gfl,
               "branch_gflops_per_decision_blocks2": 2 * float(np.mean([lf(n) for n in cache.lengths])) / 1e9}
        for kind in a.kinds:
            per = {}
            for task in tasks:
                sel = task_rows(allx, rng, task)
                b = make(kind, k, ds.tasks[task].labels, trunk)
                r = train_eval(b, cache, allx, sel, task, kind.split(":")[0], min_steps=a.min_steps, seed=a.seed,
                               epochs=8 if kind == "probe" else None)
                per[task] = r["test"]["acc"]
            row[kind] = {"mean_acc": float(np.mean(list(per.values()))), "per_task": per}
        row["wall_s"] = round(time.time() - t0, 1)
        res["configs"][how] = row
        log(f"-- {how:13s}: tokens {R:5.1f} | trunk GFLOP {trunk_gfl:5.2f} | " +
            " | ".join(f"{kind} {row[kind]['mean_acc']:.4f}" for kind in a.kinds) + f" | {row['wall_s']:.0f}s")
        save(res, a.out)
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
