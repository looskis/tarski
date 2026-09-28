"""Far-field idea 9: shared attention, private MLP (cortical columns / task-level MoE).

Every blocks branch at (k, d) starts from the same base layers, and `tarski/fused.py` runs N branches as one
stacked computation that recomputes attention per branch. If the attention sub-blocks (Wqkv, Wo, attn_norm)
stay frozen at the base's weights and only the MLPs, norms and head are trained, then (a) the branch file
shrinks by the attention parameters (2.36M of 5.02M per layer) and (b) the first branch layer's attention can
be computed once for all N branches, because its input (the trunk state) and weights are identical. From the
second layer on the residual streams differ per branch, so only weights, not computation, are shared there.

Tests:
  accuracy   blocks@k+d fully trained vs attention frozen, Banking77 and CLINC150 (all three tasks)
  latency    FusedBlocks vs SharedAttnFused (first-layer attention shared) vs a loop, N in {1, 4, 10, 20}
             branches, batch 1 and 16, with an equality check between the two fused modules.

Testable prediction: freezing attention costs 0.3-0.8 accuracy points; shared first-layer attention saves
about a quarter of the per-branch compute at d=2 (attention is ~47% of a layer's MACs at 64 tokens), and
about half at d=1; branch files shrink ~47%.

Usage:
  .venv/bin/python explore/farfield_sharedattn.py --smoke
  .venv/bin/python explore/farfield_sharedattn.py --out results/tarski/explore_sharedattn.json
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from explore.farfield_common import Logger, dump, out_paths, smoke_subset, split_indices, train_and_eval
from tarski import data
from tarski.branches import BlockBranch
from tarski.fused import FusedBlocks, _norm
from tarski.train import FeatureCache, autocast
from tarski.trunk import Trunk


def freeze_attention(branch: BlockBranch) -> int:
    n = 0
    for name, p in branch.named_parameters():
        if ".attn." in name or ".attn_norm." in name:
            p.requires_grad_(False)
            n += p.numel()
    return n


class SharedAttnFused(FusedBlocks):
    """FusedBlocks for branches whose attention weights are identical: layer 0's attention runs once on the
    shared trunk state and is broadcast; later layers run per branch (their inputs differ)."""

    def forward(self, h: torch.Tensor, ctx) -> torch.Tensor:
        n = self.norm_w.shape[0]
        B, L, D = h.shape
        hd = D // self.heads
        x = None
        for i, (p, lt, has_norm) in enumerate(zip(self.params, self.layer_types, self.has_attn_norm)):
            cos, sin = ctx.rope[lt]
            mask = ctx.masks[lt]
            if i == 0:
                a = F.layer_norm(h, (D,), p["attn_norm_w"][0], p["attn_norm_b"][0] if p["attn_norm_b"] is not None else None,
                                 self.eps) if has_norm else h
                qkv = F.linear(a, p["Wqkv"][0], p["Wqkv_b"][0] if p["Wqkv_b"] is not None else None).view(B, L, 3, self.heads, hd)
                q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
                q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=hd ** -0.5).transpose(1, 2).reshape(B, L, D)
                o = F.linear(o, p["Wo"][0], p["Wo_b"][0] if p["Wo_b"] is not None else None)
                x = (h + o).unsqueeze(0).expand(n, B, L, D)
            else:
                a = _norm(x, p["attn_norm_w"], p["attn_norm_b"], self.eps) if has_norm else x
                qkv = self._lin(a, p["Wqkv"], p["Wqkv_b"]).view(n * B, L, 3, self.heads, hd)
                q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
                q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
                m = mask.repeat(n, *([1] * (mask.dim() - 1))) if mask is not None else None
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=hd ** -0.5).transpose(1, 2).reshape(n, B, L, D)
                x = x + self._lin(o, p["Wo"], p["Wo_b"])
            m2 = _norm(x, p["mlp_norm_w"], p["mlp_norm_b"], self.eps)
            inp, gate = self._lin(m2, p["Wi"], p["Wi_b"]).chunk(2, dim=-1)
            x = x + self._lin(self.act(inp) * gate, p["Wo2"], p["Wo2_b"])
        x = _norm(x, self.norm_w, self.norm_b, self.eps)
        att = ctx.attention_mask.to(x.dtype)[None, :, :, None]
        pooled = (x * att).sum(2) / att.sum(2).clamp_min(1.0)
        return torch.einsum("nbd,ncd->nbc", pooled, self.out_w) + self.out_b[:, None, :]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def time_fn(fn, dev, reps: int) -> float:
    for _ in range(3):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / reps * 1000


@torch.no_grad()
def latency_study(trunk: Trunk, branch: BlockBranch, split: int, ns: List[int], batch_sizes: List[int], reps: int, L) -> Dict:
    """Replicate one frozen-attention branch N times (as independent branch objects) and time the modules."""
    dev = trunk.device
    text = "hi there, could you tell me whether my card payment to the electricity company went through last night"
    out = {}
    for bsz in batch_sizes:
        enc = trunk.tokenize([text] * bsz, 64)
        with autocast(dev):
            taps, ctx = trunk.taps(enc["input_ids"], enc["attention_mask"], [split])
        h = taps[split].float()
        ctx.masks = {k: (v.float() if v is not None and v.dtype != torch.bool else v) for k, v in ctx.masks.items()}
        for n in ns:
            branches = [copy.deepcopy(branch).eval() for _ in range(n)]
            fused, shared = FusedBlocks(branches), SharedAttnFused(branches)
            z1, z2 = fused(h, ctx), shared(h, ctx)
            z0 = torch.stack([b.logits(h, ctx) for b in branches])
            diff = float((z1 - z2).abs().max())
            diff_loop = float((z0 - z2).abs().max())
            row = {"max_abs_diff_fused_vs_shared": diff, "max_abs_diff_loop_vs_shared": diff_loop,
                   "loop_ms": time_fn(lambda: [b.logits(h, ctx) for b in branches], dev, reps),
                   "fused_ms": time_fn(lambda: fused(h, ctx), dev, reps),
                   "shared_attn_ms": time_fn(lambda: shared(h, ctx), dev, reps)}
            row["shared_vs_fused_speedup"] = row["fused_ms"] / max(row["shared_attn_ms"], 1e-9)
            out[f"bs{bsz}_n{n}"] = row
            L(f"  latency bs={bsz} N={n:>2}: loop {row['loop_ms']:.1f} ms | fused {row['fused_ms']:.1f} ms | shared-attn "
              f"{row['shared_attn_ms']:.1f} ms (x{row['shared_vs_fused_speedup']:.2f}); max|diff| vs fused {diff:.1e}, vs loop {diff_loop:.1e}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150"])
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.device = args.device or "cpu"
        args.datasets = ["banking77"] if args.datasets == ["banking77", "clinc150"] else args.datasets
        args.epochs, args.min_steps, args.reps = 1, 8, 3
    out, logp = out_paths(args, "sharedattn")
    L = Logger(logp)
    t_start = time.time()
    trunk = Trunk(device=args.device)
    results = {"config": vars(args), "device": str(trunk.device), "datasets": {}, "params": {}}
    frozen_example = None
    for name in args.datasets:
        ds = data.load(name)
        if args.smoke:
            ds = smoke_subset(ds, 500, 150, 300, args.seed, key=lambda e: e.y["intent"])
        allx = ds.train + ds.val + ds.test
        idx = split_indices(len(ds.train), len(ds.val), len(allx))
        L(f"== {ds.summary()} | blocks@{args.split}+{args.depth}")
        cache = FeatureCache(trunk, [e.text for e in allx], [args.split], ds.max_len)
        L(f"  cached depth {args.split}: {cache.seconds:.1f}s ({cache.bytes() / 1e6:.0f} MB)")
        dres = {}
        for task in ds.tasks:
            labels = ds.tasks[task].labels
            tres = {}
            for cond in ("full", "frozen_attn"):
                branch = BlockBranch(args.split, labels, trunk.hidden, args.depth, trunk, init="next")
                frozen = freeze_attention(branch) if cond == "frozen_attn" else 0
                r = train_and_eval(branch, cache, allx, idx, task, epochs=args.epochs, min_steps=args.min_steps, seed=args.seed)
                m = r["metrics"]
                tres[cond] = {**m, "frozen_params": frozen}
                results["params"][cond] = {"total": m["params"], "trainable": m["trainable_params"]}
                L(f"  [{task}] {cond:>11}: test acc {m['acc']:.4f} macro-F1 {m['macro_f1']:.4f} NLL {m['nll']:.3f} "
                  f"(trainable {m['trainable_params'] / 1e6:.2f}M of {m['params'] / 1e6:.2f}M, {m['steps_run']} steps, {m['train_s']}s)")
                if cond == "frozen_attn" and frozen_example is None:
                    frozen_example = branch
                else:
                    del branch
            tres["acc_delta_frozen_minus_full"] = tres["frozen_attn"]["acc"] - tres["full"]["acc"]
            dres[task] = tres
        results["datasets"][name] = dres
        dump(results, out)
        del cache
    ns = [1, 4, 10, 20] if not args.smoke else [1, 4]
    bss = [1, 16] if not args.smoke else [1]
    results["latency"] = latency_study(trunk, frozen_example, args.split, ns, bss, args.reps, L)
    D, Li, Ltok = trunk.hidden, trunk.cfg.intermediate_size, 64
    attn_macs, mlp_macs = 4 * D * D + 2 * Ltok * D, 3 * D * Li
    results["analytic_attention_share_of_layer_at_64_tokens"] = attn_macs / (attn_macs + mlp_macs)
    results["wall_seconds"] = round(time.time() - t_start, 1)
    dump(results, out)
    deltas = {f"{n}/{t}": round(v["acc_delta_frozen_minus_full"], 4) for n, d in results["datasets"].items() for t, v in d.items()}
    L(f"== done in {results['wall_seconds']}s; wrote {out}")
    L(f"   frozen-attention accuracy deltas: {deltas}; analytic attention share of a layer at 64 tokens "
      f"{results['analytic_attention_share_of_layer_at_64_tokens']:.2f}")


if __name__ == "__main__":
    main()
