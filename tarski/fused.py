"""Run many block branches as one batched computation.

Branches with the same split and depth share their input (the trunk state at that split), so their
layers can run as one stacked computation: weights are stacked along a new branch axis, the input is
broadcast to it, and attention runs over (branches x messages) as one batch. The math is ModernBERT's
encoder layer (pre-norm attention with RoPE, GeGLU MLP), matched to the looped branches in
`tests/test_fused.py`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from tarski.branches import BlockBranch, mean_pool
from tarski.trunk import Context


def _stack(ts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([t.detach() for t in ts])


def _norm(x: torch.Tensor, w: torch.Tensor, b, eps: float) -> torch.Tensor:
    """LayerNorm over the last dim with per-branch affine. x: (N, B, L, D), w: (N, D)."""
    y = F.layer_norm(x, x.shape[-1:], eps=eps)
    y = y * w[:, None, None, :]
    return y + b[:, None, None, :] if b is not None else y


class FusedBlocks(nn.Module):
    """N block branches with the same (split, depth), evaluated together."""

    def __init__(self, branches: Sequence[BlockBranch]):
        super().__init__()
        first = branches[0]
        if any(b.split != first.split or b.depth != first.depth for b in branches):
            raise ValueError("fused branches must share split and depth")
        self.names = [getattr(b, "task", str(i)) for i, b in enumerate(branches)]
        self.split, self.depth = first.split, first.depth
        cfg = first.layers[0].config
        self.eps = cfg.norm_eps
        self.heads = cfg.num_attention_heads
        self.layer_types = [l.attention_type for l in first.layers]
        self.has_attn_norm = [not isinstance(l.attn_norm, nn.Identity) for l in first.layers]
        self.act = first.layers[0].mlp.act
        self.params: List[Dict[str, torch.Tensor]] = []
        for i in range(self.depth):
            ls = [b.layers[i] for b in branches]
            p = {"Wqkv": _stack([l.attn.Wqkv.weight for l in ls]), "Wo": _stack([l.attn.Wo.weight for l in ls]),
                 "Wi": _stack([l.mlp.Wi.weight for l in ls]), "Wo2": _stack([l.mlp.Wo.weight for l in ls]),
                 "mlp_norm_w": _stack([l.mlp_norm.weight for l in ls]),
                 "mlp_norm_b": _stack([l.mlp_norm.bias for l in ls]) if ls[0].mlp_norm.bias is not None else None}
            if self.has_attn_norm[i]:
                p["attn_norm_w"] = _stack([l.attn_norm.weight for l in ls])
                p["attn_norm_b"] = _stack([l.attn_norm.bias for l in ls]) if ls[0].attn_norm.bias is not None else None
            for bias_name, mod in (("Wqkv_b", "attn.Wqkv"), ("Wo_b", "attn.Wo"), ("Wi_b", "mlp.Wi"), ("Wo2_b", "mlp.Wo")):
                bs = [l.get_submodule(mod).bias for l in ls]
                p[bias_name] = _stack(bs) if bs[0] is not None else None
            self.params.append(p)
        self.norm_w = _stack([b.norm.weight for b in branches])
        self.norm_b = _stack([b.norm.bias for b in branches]) if branches[0].norm.bias is not None else None
        self.out_w = _stack([b.out.weight for b in branches])            # (N, C, D); group by label count
        self.out_b = _stack([b.out.bias for b in branches])
        self.temperature = torch.stack([b.temperature.detach().float() for b in branches])

    @staticmethod
    def _lin(x, w, b):
        # x: (N, B, L, Din), w: (N, Dout, Din)
        y = torch.einsum("nbld,ned->nble", x, w)
        return y + b[:, None, None, :] if b is not None else y

    def forward(self, h: torch.Tensor, ctx: Context) -> torch.Tensor:
        """h: (B, L, D) trunk state at the split. Returns logits (N, B, C)."""
        n = self.norm_w.shape[0]
        B, L, D = h.shape
        x = h.unsqueeze(0).expand(n, B, L, D)
        hd = D // self.heads
        for p, lt, has_norm in zip(self.params, self.layer_types, self.has_attn_norm):
            a = _norm(x, p["attn_norm_w"], p["attn_norm_b"], self.eps) if has_norm else x
            qkv = self._lin(a, p["Wqkv"], p["Wqkv_b"]).view(n * B, L, 3, self.heads, hd)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)       # (nB, H, L, hd)
            cos, sin = ctx.rope[lt]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            mask = ctx.masks[lt]
            if mask is not None:
                mask = mask.repeat(n, *([1] * (mask.dim() - 1)))
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=hd ** -0.5)
            o = o.transpose(1, 2).reshape(n, B, L, D)
            x = x + self._lin(o, p["Wo"], p["Wo_b"])
            m = _norm(x, p["mlp_norm_w"], p["mlp_norm_b"], self.eps)
            inp, gate = self._lin(m, p["Wi"], p["Wi_b"]).chunk(2, dim=-1)
            x = x + self._lin(self.act(inp) * gate, p["Wo2"], p["Wo2_b"])
        x = _norm(x, self.norm_w, self.norm_b, self.eps)
        att = ctx.attention_mask.to(x.dtype)[None, :, :, None]
        pooled = (x * att).sum(2) / att.sum(2).clamp_min(1.0)                    # (N, B, D)
        return torch.einsum("nbd,ncd->nbc", pooled, self.out_w) + self.out_b[:, None, :]

    def probs(self, h: torch.Tensor, ctx: Context) -> torch.Tensor:
        z = self.forward(h, ctx).float()
        return torch.softmax(z / self.temperature.to(z.device)[:, None, None], -1)


def group_branches(branches: Dict[str, BlockBranch]) -> Dict[Tuple[int, int, int], List[str]]:
    """Group fusable branches by (split, depth, label count)."""
    groups: Dict[Tuple[int, int, int], List[str]] = {}
    for name, b in branches.items():
        if isinstance(b, BlockBranch):
            groups.setdefault((b.split, b.depth, b.n_labels), []).append(name)
    return groups
