"""Lit-scan idea 7: singular-value-only (SVF) branches. Probe-sized branch files, and N decisions that
share one set of weight matrices.

A blocks@k+d branch copies base layers [k, k+d). An SVF branch keeps each copied weight frozen as its SVD
W = U diag(sigma) V^T and trains only a vector z per matrix: W' = U diag(sigma * z) V^T (Transformer-squared,
Sun, Cetin, Tang, ICLR 2025). Arms:
  probe      tarski ProbeBranch at k
  full       tarski BlockBranch (every copied weight trainable; ~10.1M params for d = 2)
  svf        z only (+ the head); LayerNorms frozen
  svf_norm   z + LayerNorm weights (+ the head)
  lora8      LoRA rank 8 on the four linears (+ the head), LayerNorms frozen
Stored bytes = trainable parameters x 2 (the base weights are shared with the trunk).

Serving: all SVF branches at one split read the same U and V, so N branches are two shared-weight GEMMs
per linear with a per-branch diagonal in between, instead of N different weight matrices (tarski's
FusedBlocks, which did not beat a loop on the Mac GPU). --latency times, per message at batch 1:
loop of N full branches | FusedBlocks | SharedSVF, for N = 1, 5, 20 (random weights; latency does not
depend on training). Checks: an SVF linear with z = 1 reproduces the base linear, and SharedSVF equals the
looped SVF branches.

Prediction (lit_efficiency.md entry 7): svf/svf_norm between probe and full (~90.5-91.5 vs 92.0 on
Banking77 8+2), ~150x smaller branch files, and lower N = 20 latency than FusedBlocks.

Usage:
  .venv/bin/python explore/lit_eff_svf.py --smoke
  .venv/bin/python explore/lit_eff_svf.py --out results/tarski/explore_lit_svf.json
  .venv/bin/python explore/lit_eff_svf.py --latency-only --device mps --out results/tarski/explore_lit_svf_latency_mps.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_eff_common import Log, index, load_ds, make, save, task_rows, texts, threads, timeit, train_eval

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from tarski.branches import BlockBranch, ProbeBranch
from tarski.fused import FusedBlocks, _norm
from tarski.train import FeatureCache, autocast
from tarski.trunk import Trunk

LINEARS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")


class SVFLinear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        U, S, Vh = torch.linalg.svd(lin.weight.data.float(), full_matrices=False)
        self.register_buffer("U", U.contiguous())
        self.register_buffer("S", S.contiguous())
        self.register_buffer("Vh", Vh.contiguous())
        self.z = nn.Parameter(torch.ones_like(S))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.clone(), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        y = ((x @ self.Vh.T.to(x.dtype)) * (self.S * self.z).to(x.dtype)) @ self.U.T.to(x.dtype)
        return y if self.bias is None else y + self.bias.to(x.dtype)

    def merged(self) -> torch.Tensor:
        return (self.U * (self.S * self.z)[None]) @ self.Vh


class LoRALinear(nn.Module):
    def __init__(self, lin: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = lin
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, lin.in_features) / lin.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def _swap(br: BlockBranch, fn, freeze_norms: bool) -> BlockBranch:
    for layer in br.layers:
        for name in LINEARS:
            parent, attr = name.split(".")
            mod = getattr(layer, parent)
            setattr(mod, attr, fn(getattr(mod, attr)))
        if freeze_norms:
            for norm in (layer.attn_norm, layer.mlp_norm):
                for p in norm.parameters():
                    p.requires_grad_(False)
    return br


def build(arm: str, k: int, labels: List[str], trunk: Trunk):
    if arm == "probe":
        return ProbeBranch(k, labels, trunk.hidden)
    br = BlockBranch(k, labels, trunk.hidden, 2, trunk)
    if arm == "full":
        return br
    if arm in ("svf", "svf_norm"):
        return _swap(br, SVFLinear, freeze_norms=(arm == "svf"))
    if arm == "lora8":
        return _swap(br, LoRALinear, freeze_norms=True)
    raise ValueError(arm)


def trainable(br) -> int:
    return sum(p.numel() for p in br.parameters() if p.requires_grad)


LR = {"probe": 1e-4, "full": 1e-4, "svf": 2e-3, "svf_norm": 2e-3, "lora8": 5e-4}


# ---------------------------------------------------------------------------------------------------
# Shared-weight serving of N SVF branches
# ---------------------------------------------------------------------------------------------------

class SharedSVF(nn.Module):
    """N SVF block branches with the same split/depth and the same base layers, run together: every
    linear is x @ V^T (shared) -> per-branch diagonal -> @ U^T (shared)."""

    def __init__(self, branches: List[BlockBranch]):
        super().__init__()
        first = branches[0]
        cfg = first.layers[0].config
        self.eps, self.heads, self.act = cfg.norm_eps, cfg.num_attention_heads, first.layers[0].mlp.act
        self.layer_types = [l.attention_type for l in first.layers]
        self.has_attn_norm = [not isinstance(l.attn_norm, nn.Identity) for l in first.layers]
        self.params = []
        st = lambda ts: torch.stack([t.detach() for t in ts])
        for i in range(len(first.layers)):
            ls = [b.layers[i] for b in branches]
            p = {}
            for key, mod in (("qkv", "attn.Wqkv"), ("o", "attn.Wo"), ("i", "mlp.Wi"), ("o2", "mlp.Wo")):
                m0 = ls[0].get_submodule(mod)
                p[key] = (m0.U, m0.Vh, st([l.get_submodule(mod).S * l.get_submodule(mod).z for l in ls]))
            p["mlp_norm_w"] = st([l.mlp_norm.weight for l in ls])
            p["mlp_norm_b"] = st([l.mlp_norm.bias for l in ls]) if ls[0].mlp_norm.bias is not None else None
            if self.has_attn_norm[i]:
                p["attn_norm_w"] = st([l.attn_norm.weight for l in ls])
                p["attn_norm_b"] = st([l.attn_norm.bias for l in ls]) if ls[0].attn_norm.bias is not None else None
            self.params.append(p)
        self.norm_w = st([b.norm.weight for b in branches])
        self.norm_b = st([b.norm.bias for b in branches]) if branches[0].norm.bias is not None else None
        self.out_w, self.out_b = st([b.out.weight for b in branches]), st([b.out.bias for b in branches])

    @staticmethod
    def _lin(x, U, Vh, s):
        N, B, L, Din = x.shape
        y = x.reshape(-1, Din) @ Vh.T.to(x.dtype)
        y = y.view(N, B * L, -1) * s[:, None, :].to(x.dtype)
        return (y.reshape(-1, s.shape[1]) @ U.T.to(x.dtype)).view(N, B, L, -1)

    def forward(self, h, ctx):
        n = self.norm_w.shape[0]
        B, L, D = h.shape
        x = h.unsqueeze(0).expand(n, B, L, D)
        hd = D // self.heads
        for p, lt, has_norm in zip(self.params, self.layer_types, self.has_attn_norm):
            a = _norm(x, p["attn_norm_w"], p["attn_norm_b"], self.eps) if has_norm else x
            qkv = self._lin(a, *p["qkv"]).view(n * B, L, 3, self.heads, hd)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            cos, sin = ctx.rope[lt]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            mask = ctx.masks[lt]
            if mask is not None:
                mask = mask.repeat(n, *([1] * (mask.dim() - 1)))
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=hd ** -0.5)
            x = x + self._lin(o.transpose(1, 2).reshape(n, B, L, D), *p["o"])
            m = _norm(x, p["mlp_norm_w"], p["mlp_norm_b"], self.eps)
            inp, gate = self._lin(m, *p["i"]).chunk(2, dim=-1)
            x = x + self._lin(self.act(inp) * gate, *p["o2"])
        x = _norm(x, self.norm_w, self.norm_b, self.eps)
        att = ctx.attention_mask.to(x.dtype)[None, :, :, None]
        pooled = (x * att).sum(2) / att.sum(2).clamp_min(1.0)
        return torch.einsum("nbd,ncd->nbc", pooled, self.out_w.to(x.dtype)) + self.out_b[:, None, :].to(x.dtype)


@torch.no_grad()
def latency(trunk: Trunk, k: int, ns=(1, 5, 20), reps: int = 20) -> Dict:
    dev = trunk.device
    text = "hey can someone on billing look at the double charge for acme, they're pretty upset"
    b = trunk.tokenize([text], 64)
    labels = [f"l{i}" for i in range(20)]
    out = {"device": str(dev), "split": k, "rows": []}
    with autocast(dev):
        taps, ctx = trunk.taps(b["input_ids"], b["attention_mask"], [k])
        h = taps[k]
        out["trunk_ms"] = timeit(lambda: trunk.taps(b["input_ids"], b["attention_mask"], [k]), dev, reps=reps)
        for n in ns:
            full = [BlockBranch(k, labels, trunk.hidden, 2, trunk).to(dev).eval() for _ in range(n)]
            svf = [build("svf_norm", k, labels, trunk).to(dev).eval() for _ in range(n)]
            for br in svf:
                for m in br.modules():
                    if isinstance(m, SVFLinear):
                        m.z.data.uniform_(0.9, 1.1)
            fused = FusedBlocks(full).to(dev)
            shared = SharedSVF(svf).to(dev)
            row = {"n": n,
                   "loop_full_ms": timeit(lambda: [br.logits(h, ctx) for br in full], dev, reps=reps),
                   "fused_full_ms": timeit(lambda: fused(h, ctx), dev, reps=reps),
                   "loop_svf_ms": timeit(lambda: [br.logits(h, ctx) for br in svf], dev, reps=reps),
                   "shared_svf_ms": timeit(lambda: shared(h, ctx), dev, reps=reps)}
            out["rows"].append(row)
            del full, svf, fused, shared
    return out


def checks(trunk: Trunk, k: int) -> Dict:
    lin = trunk.model.layers[k].attn.Wqkv
    x = torch.randn(4, 7, trunk.hidden, device=trunk.device)
    s = SVFLinear(lin).to(trunk.device)
    e1 = float((s(x) - lin(x)).norm() / lin(x).norm())
    brs = [build("svf_norm", k, ["a", "b", "c"], trunk).to(trunk.device).eval() for _ in range(3)]
    for br in brs:
        for m in br.modules():
            if isinstance(m, SVFLinear):
                m.z.data.uniform_(0.8, 1.2)
    b = trunk.tokenize(["a short message", "a somewhat longer message about a refund"], 32)
    with torch.no_grad():
        taps, ctx = trunk.taps(b["input_ids"], b["attention_mask"], [k])
        loop = torch.stack([br.logits(taps[k], ctx) for br in brs])
        sh = SharedSVF(brs).to(trunk.device)(taps[k], ctx)
    e2 = float((loop - sh).abs().max())
    return {"svf_z1_rel_err": e1, "shared_vs_loop_abs_err": e2}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", nargs="*", default=["banking77:8:probe,full,svf,svf_norm,lora8",
                                                   "banking77:14:full,svf,svf_norm",
                                                   "clinc150:11:full,svf_norm",
                                                   "typed-decisions:14:full,svf_norm"],
                    help="dataset:split:arm,arm,...")
    ap.add_argument("--typed-workflows", nargs="*", default=["customer_service"])
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--latency-only", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.min_steps, a.epochs, a.latency = "cpu", 20, 1, True
        a.plan = ["banking77:4:probe,full,svf,svf_norm,lora8", "typed-decisions:4:svf_norm"]
        a.out = a.out or "results/tarski/explore_lit_svf_smoke.json"
    if not a.out:
        ap.error("--out is required")
    log = Log(a.out)
    threads()
    trunk = Trunk(device=a.device)
    res = {"args": vars(a), "runs": {}}
    ck = checks(trunk, 4 if a.smoke else 8)
    res["checks"] = ck
    log(f"checks: SVF(z=1) vs base linear rel err {ck['svf_z1_rel_err']:.1e}; SharedSVF vs loop abs err "
        f"{ck['shared_vs_loop_abs_err']:.1e}")
    # 1e-3: CUDA matmuls may run in TF32, which leaves ~2e-4 relative error in the SVD reconstruction
    assert ck["svf_z1_rel_err"] < 1e-3 and ck["shared_vs_loop_abs_err"] < 1e-3
    if a.latency or a.latency_only:
        lat = latency(trunk, 4 if a.smoke else 8, ns=(1, 5) if a.smoke else (1, 5, 20), reps=3 if a.smoke else 20)
        res["latency"] = lat
        log(f"latency on {lat['device']} (split {lat['split']}, trunk {lat['trunk_ms']:.2f} ms):")
        for r in lat["rows"]:
            log(f"   N={r['n']:2d}: loop full {r['loop_full_ms']:.2f} | FusedBlocks {r['fused_full_ms']:.2f} | "
                f"loop SVF {r['loop_svf_ms']:.2f} | SharedSVF {r['shared_svf_ms']:.2f} ms")
        save(res, a.out)
        if a.latency_only:
            return
    for spec in a.plan:
        name, k, arms = spec.split(":")
        k, arms = int(k), arms.split(",")
        ds = load_ds(name, a.smoke, workflows=a.typed_workflows if name == "typed-decisions" else None)
        allx, rng = index(ds)
        tasks = list(ds.tasks)[: (2 if a.smoke else None)]
        fc = FeatureCache(trunk, texts(ds), [k], ds.max_len)
        log(f"== {name} split {k}: {len(tasks)} tasks, cached in {fc.seconds:.0f}s")
        R = {}
        for arm in arms:
            per, t0 = {}, time.time()
            for task in tasks:
                sel = task_rows(allx, rng, task)
                br = build(arm, k, ds.tasks[task].labels, trunk)
                n_tr = trainable(br)
                r = train_eval(br, fc, allx, sel, task, "probe" if arm == "probe" else "blocks", min_steps=a.min_steps,
                               epochs=8 if arm == "probe" else a.epochs, lr_layers=LR[arm],
                               lr_head=3e-3 if arm == "probe" else 1e-3)
                per[task] = {"acc": r["test"]["acc"], "macro_f1": r["test"]["macro_f1"], "ece": r["test"]["ece"],
                             "trainable_params": n_tr, "stored_bytes_fp16": 2 * n_tr, "train_s": r["train_s"]}
            R[arm] = {"mean_acc": float(np.mean([v["acc"] for v in per.values()])),
                      "trainable_params": int(np.mean([v["trainable_params"] for v in per.values()])),
                      "per_task": per, "wall_s": round(time.time() - t0, 1)}
            log(f"   {arm:9s}: mean acc {R[arm]['mean_acc']:.4f} | trainable {R[arm]['trainable_params'] / 1e3:.1f}k "
                f"({2 * R[arm]['trainable_params'] / 1e6:.2f} MB fp16) | {R[arm]['wall_s']:.0f}s")
            res["runs"][f"{name}:{k}"] = R
            save(res, a.out)
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
