"""Idea 9: delta-coded branches, and branches that ride on the trunk's own matmuls.

A blocks@k+d branch initialised from base layers [k, k+d) is "base + delta". Two tests:

1. Storage. Train a full-copy blocks@8+2 branch on Banking77, then replace each weight's delta from
   the base by a compressed version, with no retraining: BitDelta (sign x one scale per matrix, and
   per output row), truncated SVD (rank 8/32/128), and top-magnitude sparsification (1%/10%). Report
   test accuracy and bytes against the fp16 branch file.

2. Shared base product. A LoRA branch (rank 16 on Wqkv/Wo/Wi/Wo2, base weights and LayerNorms frozen)
   has a first-layer input LN_k(h_k) identical to the trunk's own layer k, so its QKV projection is
   `trunk product + low-rank correction`. When the trunk runs past k anyway (another decision needs a
   deeper split), that product is free. Measured: LoRA vs full-copy accuracy, and the latency of N
   branches with (a) dense fused weights (tarski FusedBlocks), (b) shared product paid once, (c) shared
   product free (already computed by the trunk).

Tested prediction: <= 0.5 point loss for LoRA and for 1-bit deltas, ~10x smaller files, and 10-30%
lower branch latency at N=20 when the product is shared.

Usage:
  python explore/systems_deltabranch.py --smoke
  python explore/systems_deltabranch.py --out results/tarski/explore_deltabranch.json
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explore.systems_common import Log, all_texts, save, subset, sync, threads, timeit, train_tasks

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from tarski import data
from tarski.branches import BlockBranch
from tarski.fused import FusedBlocks
from tarski.train import FeatureCache, autocast, evaluate, predict_logits
from tarski.trunk import Trunk

LINEARS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged(self) -> torch.Tensor:
        return self.base.weight + self.scale * self.B @ self.A


def to_lora(br: BlockBranch, r: int) -> BlockBranch:
    for layer in br.layers:
        for name in LINEARS:
            parent, attr = name.split(".")
            mod = getattr(layer, parent)
            setattr(mod, attr, LoRALinear(getattr(mod, attr), r))
        for norm in (layer.attn_norm, layer.mlp_norm):
            for p in norm.parameters():
                p.requires_grad_(False)
    return br


def compress(delta: torch.Tensor, how: str):
    """Returns (approximation, bytes)."""
    m, n = delta.shape
    if how == "bit":
        s = delta.abs().mean()
        return torch.sign(delta) * s, m * n / 8 + 2
    if how == "bit-row":
        s = delta.abs().mean(1, keepdim=True)
        return torch.sign(delta) * s, m * n / 8 + 2 * m
    if how.startswith("svd"):
        r = int(how[3:])
        U, S, V = torch.linalg.svd(delta.float(), full_matrices=False)
        return (U[:, :r] * S[:r]) @ V[:r], 2 * r * (m + n)
    if how.startswith("top"):
        frac = float(how[3:]) / 100
        kk = max(1, int(frac * delta.numel()))
        thr = delta.abs().flatten().kthvalue(delta.numel() - kk + 1).values
        keep = delta.abs() >= thr
        return delta * keep, kk * (2 + 4)
    raise ValueError(how)


def delta_variant(br: BlockBranch, trunk: Trunk, how: str):
    out = copy.deepcopy(br)
    total = 0.0
    with torch.no_grad():
        for i, layer in enumerate(out.layers):
            base = trunk.model.layers[br.split + i]
            for name in LINEARS:
                w = layer.get_submodule(name).weight
                b = base.get_submodule(name).weight
                approx, nbytes = compress((w - b).float().cpu(), how)
                w.copy_((b.float().cpu() + approx).to(w.device, w.dtype))
                total += nbytes
    other = sum(p.numel() * 2 for n_, p in br.named_parameters() if not any(x in n_ for x in LINEARS))
    return out, total + other


class SharedLoRAFused:
    """N LoRA branches (same split, depth 2): the first layer's QKV projection is the base product
    (computed once, or supplied by the trunk) plus per-branch low-rank corrections; the rest runs as
    stacked dense weights, as in FusedBlocks."""

    def __init__(self, branches: List[BlockBranch], trunk: Trunk):
        self.trunk, self.n = trunk, len(branches)
        l0 = branches[0].layers[0]
        self.base_norm = l0.attn_norm
        self.base_Wqkv = l0.attn.Wqkv.base.weight.detach()
        self.A = torch.stack([b.layers[0].attn.Wqkv.A.detach() for b in branches])             # (N, r, D)
        self.B = torch.stack([b.layers[0].attn.Wqkv.B.detach() * b.layers[0].attn.Wqkv.scale for b in branches])
        # the rest: merged dense weights through tarski's fused path, starting after the first QKV
        merged = [merge_lora(b) for b in branches]
        self.fused = FusedBlocks(merged)

    def qkv_base(self, h):
        return self.base_norm(h) @ self.base_Wqkv.T

    def forward(self, h, ctx, qkv_base=None):
        """Same math as FusedBlocks.forward, except layer 0's QKV = shared base product + LoRA."""
        fb = self.fused
        n, (B, L, D) = self.n, h.shape
        hd = D // fb.heads
        x = h.unsqueeze(0).expand(n, B, L, D)
        a0 = self.base_norm(h)
        base = qkv_base if qkv_base is not None else a0 @ self.base_Wqkv.T                        # (B, L, 3D)
        low = torch.einsum("bld,nrd->nblr", a0, self.A.to(a0.dtype))
        qkv0 = base.unsqueeze(0) + torch.einsum("nblr,ner->nble", low, self.B.to(a0.dtype))
        for li, (p, lt, has_norm) in enumerate(zip(fb.params, fb.layer_types, fb.has_attn_norm)):
            if li == 0:
                qkv = qkv0
            else:
                a = F.layer_norm(x, (D,), eps=fb.eps) * p["attn_norm_w"][:, None, None, :] if has_norm else x
                qkv = torch.einsum("nbld,ned->nble", a, p["Wqkv"])
            qkv = qkv.reshape(n * B, L, 3, fb.heads, hd)
            q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
            cos, sin = ctx.rope[lt]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            mask = ctx.masks[lt]
            if mask is not None:
                mask = mask.repeat(n, *([1] * (mask.dim() - 1)))
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=hd ** -0.5)
            o = o.transpose(1, 2).reshape(n, B, L, D)
            x = x + torch.einsum("nbld,ned->nble", o, p["Wo"])
            m = F.layer_norm(x, (D,), eps=fb.eps) * p["mlp_norm_w"][:, None, None, :]
            inp, gate = torch.einsum("nbld,ned->nble", m, p["Wi"]).chunk(2, dim=-1)
            x = x + torch.einsum("nbld,ned->nble", fb.act(inp) * gate, p["Wo2"])
        x = F.layer_norm(x, (D,), eps=fb.eps) * fb.norm_w[:, None, None, :]
        att = ctx.attention_mask.to(x.dtype)[None, :, :, None]
        pooled = (x * att).sum(2) / att.sum(2).clamp_min(1.0)
        return torch.einsum("nbd,ncd->nbc", pooled, fb.out_w) + fb.out_b[:, None, :]


def merge_lora(br: BlockBranch) -> BlockBranch:
    out = copy.deepcopy(br)
    with torch.no_grad():
        for layer in out.layers:
            for name in LINEARS:
                parent, attr = name.split(".")
                mod = getattr(layer, parent)
                lo = getattr(mod, attr)
                if isinstance(lo, LoRALinear):
                    lin = nn.Linear(lo.base.in_features, lo.base.out_features, bias=False).to(lo.base.weight.device)
                    lin.weight.copy_(lo.merged())
                    setattr(mod, attr, lin)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77")
    ap.add_argument("--split", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--variants", nargs="*", default=["bit", "bit-row", "svd8", "svd32", "svd128", "top1", "top10"])
    ap.add_argument("--latency-ns", type=int, nargs="*", default=[1, 5, 20])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.split, a.min_steps, a.reps, a.latency_ns = "cpu", 4, 5, 2, [1, 3]
        a.variants = ["bit", "svd8", "top10"]
    if not a.smoke and not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    trunk = Trunk(device=a.device)
    ds = data.load(a.dataset)
    if a.smoke:
        ds = subset(ds, list(ds.tasks), 200, 40, 60, stride=17)
    task = list(ds.tasks)[0]
    k = a.split
    cache = FeatureCache(trunk, all_texts(ds), [k], ds.max_len)
    log(f"== {ds.summary()} | blocks@{k}+2 | task {task}")
    res = {"dataset": ds.name, "split": k, "task": task}

    full = train_tasks(trunk, ds, cache, [task], "blocks:2", k, seed=a.seed, min_steps=a.min_steps, log=log)[task]
    fbytes = sum(p.numel() * 2 for p in full["branch"].parameters())
    res["full_copy"] = {"acc": full["metrics"]["acc"], "bytes_fp16": fbytes}
    sel, yte, T = full["sel"], full["y"]["test"].numpy(), full["T"]

    def acc_of(br):
        br.to(trunk.device).eval()
        p = torch.softmax(predict_logits(br, cache, sel["test"]) / T, -1).numpy()
        return evaluate(p, yte)["acc"]

    res["delta"] = {}
    for how in a.variants:
        br, nbytes = delta_variant(full["branch"], trunk, how)
        acc = acc_of(br)
        res["delta"][how] = {"acc": acc, "bytes": nbytes, "ratio_vs_fp16": fbytes / nbytes}
        log(f"   delta {how:8s}: acc {acc:.4f} (full-copy {full['metrics']['acc']:.4f}) | {nbytes / 1e6:.2f} MB "
            f"({fbytes / nbytes:.1f}x smaller)")
        del br
    save(res, a.out)

    lora = train_tasks(trunk, ds, cache, [task], "blocks:2", k, seed=a.seed, min_steps=a.min_steps, lr_layers=1e-3,
                       build=lambda t, labels: to_lora(BlockBranch(k, labels, trunk.hidden, 2, trunk), a.rank),
                       log=log)[task]
    lbytes = sum(p.numel() * 2 for p in lora["branch"].parameters() if p.requires_grad)
    res["lora"] = {"rank": a.rank, "acc": lora["metrics"]["acc"], "bytes_trainable_fp16": lbytes}
    log(f"   LoRA r={a.rank} (frozen norms): acc {lora['metrics']['acc']:.4f} vs full-copy "
        f"{full['metrics']['acc']:.4f} | {lbytes / 1e6:.2f} MB of trainable weights")
    save(res, a.out)

    # correctness of the shared-product path, then latency
    dev = trunk.device
    brs = []
    for _ in range(max(a.latency_ns)):
        b = copy.deepcopy(lora["branch"]).to(dev).eval()
        with torch.no_grad():
            for n_, p in b.named_parameters():
                if n_.endswith(".A") or n_.endswith(".B"):
                    p.add_(torch.randn_like(p) * 0.01)
        brs.append(b)
    texts = {"short": ds.test[0].text, "long": ("Subject: production outage after the deploy. " * 30)[:1500]}
    batch = trunk.tokenize([texts["short"], texts["long"]], 512)
    with torch.no_grad():
        taps, ctx = trunk.taps(batch["input_ids"], batch["attention_mask"], [k])
        sh = SharedLoRAFused(brs[:3], trunk)
        z_shared = sh.forward(taps[k], ctx)
        z_loop = torch.stack([b.logits(taps[k], ctx) for b in brs[:3]])
    err = float((z_shared - z_loop).abs().max())
    res["shared_path_abs_err"] = err
    log(f"  shared-product fused vs looped LoRA branches: max abs diff {err:.2e}")
    assert err < 1e-2, "shared-product path disagrees with the LoRA branches"

    lat = {}
    for name, text in texts.items():
        b = trunk.tokenize([text], 512)
        ids, att = b["input_ids"], b["attention_mask"]
        rows = {}
        for n in a.latency_ns:
            sh = SharedLoRAFused(brs[:n], trunk)
            dense = FusedBlocks([merge_lora(x) for x in brs[:n]])
            with torch.no_grad(), autocast(dev):
                taps, ctx = trunk.taps(ids, att, [k])
                h = taps[k]
                qb = sh.qkv_base(h)
                rows[n] = {"dense_fused": timeit(lambda: dense(h, ctx), dev, a.reps),
                           "shared_paid": timeit(lambda: sh.forward(h, ctx), dev, a.reps),
                           "shared_free": timeit(lambda: sh.forward(h, ctx, qkv_base=qb), dev, a.reps)}
            rows[n] = {kk: round(v, 3) for kk, v in rows[n].items()}
            log(f"  latency {name} ({int(att.sum())} tok) N={n}: {rows[n]} ms (branches only)")
        lat[name] = rows
    res["latency_ms"] = lat
    save(res, a.out)
    log("done")


if __name__ == "__main__":
    main()
