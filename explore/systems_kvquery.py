"""Decisions as read-only queries against the trunk's KV cache ("fork the CLS token").

Systems framing: the frozen trunk builds an index (the keys/values of every layer), and a decision is
a query against it. A `blocks@k+d` branch re-runs d whole layers over every token for every decision,
so N decisions cost N x d x (L tokens) of layer compute. A KV-query branch (`kvq@k+d`) instead forks
only the CLS token at depth k: a per-decision residual stream of r tokens (r = 1: just the CLS; r > 1:
CLS plus r-1 attention-pooled starts) runs through fine-tuned copies of base layers [k, k+d). At each
layer its queries read the keys/values that the *frozen* base layer computes for the message tokens
at that depth. The message tokens never read the stream (read-only), so their K/V are shared by every
decision, and N decisions become one small multi-query attention per layer (like decoding N tokens
against one shared prefix cache). Per-decision cost per layer: O(r*L*D + r*D^2) instead of
O(L*D^2 + L^2*D).

With r = 1, the base's own layers and local-window masking (`window`), the stream reproduces the
trunk's CLS path exactly at initialisation (checked by `exactness()`), so fine-tuning starts from
"the base model, continued for this one token".

Configs (per task, same cache and training recipe as tarski.train):
  probe            ProbeBranch at k (mean pool + linear)                   [baseline]
  blocks:D         BlockBranch at k with D layers                          [baseline]
  kvq:D:R          KV-query stream, D layers, R stream tokens, stream weights fine-tuned
  kvq:D:R:frozen   stream weights frozen (base layers); trains only the final norm, head and the
                   attention-pooling starts. R=1 is a CLS probe at depth k+D; R>1 is read-only
                   prompting (RPO-style) with trunk-sourced starts.
  kvq:D:R:global   stream reads all tokens at local layers too (no 128-token window)

Usage:
  python explore/systems_kvquery.py --smoke
  python explore/systems_kvquery.py --dataset typed-decisions --split 11 \
      --out results/tarski/explore_kvquery_typed.json
  python explore/systems_kvquery.py --dataset banking77 --split 8 --seeds 0 \
      --out results/tarski/explore_kvquery_banking77.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from tarski import data
from tarski.branches import BlockBranch, ProbeBranch
from tarski.fused import FusedBlocks
from tarski.train import FeatureCache, _targets, autocast, evaluate, fit_temperature, predict_logits, train_branch
from tarski.trunk import Context, Trunk


# ---------------------------------------------------------------------------------------------------
# The KV-query branch
# ---------------------------------------------------------------------------------------------------

def _row_mask(mask, att: torch.Tensor) -> torch.Tensor:
    """Keys visible to query position 0 under a layer's mask (bool (B, L))."""
    if mask is None:
        return att.bool()
    row = mask[:, 0, 0, :]
    row = row if row.dtype == torch.bool else (row == 0)
    return row.expand_as(att) & att.bool()


@torch.no_grad()
def shared_kv(layer: nn.Module, h: torch.Tensor, ctx: Context, heads: int):
    """Keys (RoPE applied) and values that frozen base `layer` computes for every message token.
    In a production engine these are by-products of the trunk pass; here they are recomputed."""
    B, L, D = h.shape
    hd = D // heads
    qkv = layer.attn.Wqkv(layer.attn_norm(h)).view(B, L, 3, heads, hd)
    _, k, v = qkv.unbind(2)
    k, v = k.transpose(1, 2), v.transpose(1, 2)                           # (B, H, L, hd)
    cos, sin = ctx.rope[layer.attention_type]
    k, _ = apply_rotary_pos_emb(k, k, cos, sin, unsqueeze_dim=1)
    return k, v


class KVQueryBranch(nn.Module):
    kind = "kvq"

    def __init__(self, trunk: Trunk, split: int, depth: int, labels: List[str], r: int = 1,
                 frozen: bool = False, window: bool = True, dropout: float = 0.1):
        super().__init__()
        if split < 1 or split + depth > trunk.n_layers:
            raise ValueError("kvq needs 1 <= split and split + depth <= n_layers")
        D = trunk.hidden
        self.split, self.depth, self.r, self.frozen, self.window = split, depth, r, frozen, window
        self.labels = list(labels)
        self.heads = trunk.cfg.num_attention_heads
        self.depths = list(range(split, split + depth))       # trunk states the stream reads
        self._kv = [trunk.model.layers[j] for j in self.depths]  # frozen, owned by the trunk (not registered)
        if frozen:
            self.layers = list(self._kv)       # the base layers themselves: untrained, not registered/saved
        else:
            self.layers = trunk.copy_layers(split, split + depth)
        self.norm = trunk.copy_final_norm()
        self.pool_q = nn.Parameter(torch.randn(r - 1, D) * D ** -0.5) if r > 1 else None
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(D, len(labels))
        self.register_buffer("temperature", torch.ones(()))

    @property
    def n_labels(self):
        return len(self.labels)

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.trainable())

    def stream_init(self, h0: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        x = h0[:, :1]                                                         # the CLS state at depth k
        if self.pool_q is not None:                                           # r-1 attention-pooled starts
            z = F.layer_norm(h0, h0.shape[-1:])
            s = torch.einsum("bld,rd->brl", z, self.pool_q.to(z.dtype))
            s = s.masked_fill(~att.bool()[:, None, :], float("-inf"))
            x = torch.cat([x, torch.einsum("brl,bld->brd", s.softmax(-1), h0)], 1)
        return x

    def key_masks(self, ctx: Context) -> Dict[str, torch.Tensor]:
        out = {}
        for lt in {l.attention_type for l in self._kv}:
            m = _row_mask(ctx.masks[lt], ctx.attention_mask) if self.window else ctx.attention_mask.bool()
            m = m.clone()
            m[:, 0] = False                         # the CLS slot is the stream itself (it attends to its own key)
            out[lt] = m
        return out

    def step(self, sl: nn.Module, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kmask: torch.Tensor):
        B, r, D = x.shape
        H, hd = self.heads, D // self.heads
        qkv = sl.attn.Wqkv(sl.attn_norm(x)).view(B, r, 3, H, hd)
        q, ks, vs = (t.transpose(1, 2) for t in qkv.unbind(2))              # (B, H, r, hd); position 0: RoPE = id
        scale = hd ** -0.5
        s_msg = torch.einsum("bhrd,bhld->bhrl", q, k.to(q.dtype)) * scale
        s_msg = s_msg.masked_fill(~kmask[:, None, None, :], float("-inf"))
        s_self = (q * ks).sum(-1, keepdim=True) * scale                     # each stream token sees itself only
        w = torch.cat([s_msg, s_self], -1).softmax(-1)
        o = torch.einsum("bhrl,bhld->bhrd", w[..., :-1], v.to(q.dtype)) + w[..., -1:] * vs
        x = x + sl.attn.Wo(o.transpose(1, 2).reshape(B, r, D))
        return x + sl.mlp(sl.mlp_norm(x))

    def stream(self, hs: Sequence[torch.Tensor], ctx: Context) -> torch.Tensor:
        x = self.stream_init(hs[0], ctx.attention_mask)
        km = self.key_masks(ctx)
        for h, fl, sl in zip(hs, self._kv, self.layers):
            k, v = shared_kv(fl, h, ctx, self.heads)
            x = self.step(sl, x, k, v, km[fl.attention_type])
        return x

    def logits(self, hs: Sequence[torch.Tensor], ctx: Context) -> torch.Tensor:
        return self.out(self.drop(self.norm(self.stream(hs, ctx)).mean(1)))


class FusedKVQ(nn.Module):
    """N KV-query branches with the same (split, depth, r) as one computation: per layer, the shared
    message K/V are computed once and every decision's stream queries them in one batched einsum."""

    def __init__(self, branches: Sequence[KVQueryBranch]):
        super().__init__()
        b0 = branches[0]
        self.b0, self.n, self.heads = b0, len(branches), b0.heads
        st = lambda ts: torch.stack([t.detach() for t in ts])
        self.P = []
        for i in range(b0.depth):
            ls = [b.layers[i] for b in branches]
            has = not isinstance(ls[0].attn_norm, nn.Identity)
            self.P.append({"an": st([l.attn_norm.weight for l in ls]) if has else None,
                           "Wqkv": st([l.attn.Wqkv.weight for l in ls]), "Wo": st([l.attn.Wo.weight for l in ls]),
                           "mn": st([l.mlp_norm.weight for l in ls]), "Wi": st([l.mlp.Wi.weight for l in ls]),
                           "Wo2": st([l.mlp.Wo.weight for l in ls]), "act": ls[0].mlp.act,
                           "eps": ls[0].mlp_norm.eps})
        self.norm_w = st([b.norm.weight for b in branches])
        self.eps = branches[0].norm.eps
        self.pool_q = st([b.pool_q for b in branches]) if b0.pool_q is not None else None
        self.out_w, self.out_b = st([b.out.weight for b in branches]), st([b.out.bias for b in branches])
        for l in branches[0].layers:
            assert l.attn.Wqkv.bias is None and l.mlp.Wi.bias is None, "biases not supported in the fused path"

    def forward(self, hs: Sequence[torch.Tensor], ctx: Context) -> torch.Tensor:
        n, H = self.n, self.heads
        h0, att = hs[0], ctx.attention_mask
        B, L, D = h0.shape
        hd = D // H
        x = h0[:, :1].unsqueeze(0).expand(n, B, 1, D)
        if self.pool_q is not None:
            z = F.layer_norm(h0, (D,))
            s = torch.einsum("bld,nrd->nbrl", z, self.pool_q.to(z.dtype))
            s = s.masked_fill(~att.bool()[None, :, None, :], float("-inf"))
            x = torch.cat([x, torch.einsum("nbrl,bld->nbrd", s.softmax(-1), h0)], 2)
        r = x.shape[2]
        km = self.b0.key_masks(ctx)
        ln = lambda t, w, eps: F.layer_norm(t, (D,), eps=eps) * w[:, None, None, :]
        for h, fl, p in zip(hs, self.b0._kv, self.P):
            k, v = shared_kv(fl, h, ctx, H)                                  # once for all n decisions
            a = ln(x, p["an"], p["eps"]) if p["an"] is not None else x
            qkv = torch.einsum("nbrd,ned->nbre", a, p["Wqkv"]).view(n, B, r, 3, H, hd)
            q, ks, vs = (t.transpose(2, 3) for t in qkv.unbind(3))          # (n, B, H, r, hd)
            s_msg = torch.einsum("nbhrd,bhld->nbhrl", q, k.to(q.dtype)) * hd ** -0.5
            s_msg = s_msg.masked_fill(~km[fl.attention_type][None, :, None, None, :], float("-inf"))
            s_self = (q * ks).sum(-1, keepdim=True) * hd ** -0.5
            w = torch.cat([s_msg, s_self], -1).softmax(-1)
            o = torch.einsum("nbhrl,bhld->nbhrd", w[..., :-1], v.to(q.dtype)) + w[..., -1:] * vs
            x = x + torch.einsum("nbrd,ned->nbre", o.transpose(2, 3).reshape(n, B, r, D), p["Wo"])
            inp, gate = torch.einsum("nbrd,ned->nbre", ln(x, p["mn"], p["eps"]), p["Wi"]).chunk(2, -1)
            x = x + torch.einsum("nbrd,ned->nbre", p["act"](inp) * gate, p["Wo2"])
        x = ln(x, self.norm_w, self.eps).mean(2)                              # (n, B, D)
        return torch.einsum("nbd,ncd->nbc", x, self.out_w) + self.out_b[:, None, :]


# ---------------------------------------------------------------------------------------------------
# Training on a multi-depth cache (mirrors tarski.train.train_branch)
# ---------------------------------------------------------------------------------------------------

def batch_multi(cache: FeatureCache, depths: Sequence[int], idx: Sequence[int]):
    dev, D = cache.trunk.device, cache.trunk.hidden
    dev_h = getattr(cache, "dev_h", None)
    if dev_h is not None:                              # states already on the device: pad-stack there
        from torch.nn.utils.rnn import pad_sequence
        lengths = torch.tensor([cache.lengths[i] for i in idx], device=dev)
        att = (torch.arange(int(lengths.max()), device=dev)[None] < lengths[:, None]).long()
        hs = [pad_sequence([dev_h[d][i] for i in idx], batch_first=True).float() for d in depths]
        return hs, cache.trunk.context(hs[0], att)
    L = max(cache.lengths[i] for i in idx)
    att = torch.zeros(len(idx), L, dtype=torch.long)
    for j, i in enumerate(idx):
        att[j, : cache.lengths[i]] = 1
    hs = []
    for d in depths:
        h = torch.zeros(len(idx), L, D, dtype=torch.float16)
        for j, i in enumerate(idx):
            h[j, : cache.lengths[i]] = cache.h[d][i]
        hs.append(h.to(dev, torch.float32))
    att = att.to(dev)
    return hs, cache.trunk.context(hs[0], att)


@torch.no_grad()
def predict_kvq(br: KVQueryBranch, cache: FeatureCache, idx: Sequence[int], bs: int = 64) -> torch.Tensor:
    br.eval()
    order = sorted(range(len(idx)), key=lambda j: cache.lengths[idx[j]])
    out = torch.zeros(len(idx), br.n_labels)
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        hs, ctx = batch_multi(cache, br.depths, [idx[j] for j in sel])
        out[sel] = br.logits(hs, ctx).float().cpu()
    return out


def train_kvq(br: KVQueryBranch, cache: FeatureCache, tr: List[int], y, soft, va: List[int], y_va,
              epochs: int, bs: int = 32, lr_layers: float = 1e-4, lr_head: float = 1e-3, patience: int = 3,
              seed: int = 0, log=print, min_val: int = 50, min_steps: int = 300) -> Dict:
    """Same schedule rules as tarski.train.train_branch: at least `min_steps` optimiser steps; with >=
    `min_val` validation rows keep the best-val epoch and never stop before half the schedule; with
    fewer, run the full schedule and keep the final weights."""
    per_epoch = (len(tr) + bs - 1) // bs
    epochs = max(epochs, -(-min_steps // per_epoch))
    select = len(va) >= min_val
    min_epochs = max(1, (epochs + 1) // 2)
    torch.manual_seed(seed)
    dev = cache.trunk.device
    br.to(dev).train()
    layer_p = [p for n, p in br.named_parameters() if n.startswith("layers.") and p.requires_grad]
    head_p = [p for n, p in br.named_parameters() if not n.startswith("layers.") and p.requires_grad]
    groups = [{"params": head_p, "lr": lr_head}] + ([{"params": layer_p, "lr": lr_layers}] if layer_p else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = epochs * ((len(tr) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(epochs):
        br.train()
        t0, tot = time.time(), 0.0
        order = sorted(range(len(tr)), key=lambda j: cache.lengths[tr[j]] + rng.random() * 8)
        chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
        rng.shuffle(chunks)
        for sel in chunks:
            hs, ctx = batch_multi(cache, br.depths, [tr[j] for j in sel])
            z = br.logits(hs, ctx).float()
            loss = -(soft[sel].to(dev) * F.log_softmax(z, -1)).sum(-1).mean() if soft is not None \
                else F.cross_entropy(z, y[sel].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(br.trainable(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(sel)
        acc = float((predict_kvq(br, cache, va).argmax(-1) == y_va).float().mean())
        hist.append({"epoch": ep + 1, "loss": tot / len(tr), "val_acc": acc, "s": round(time.time() - t0, 1)})
        log(f"    epoch {ep + 1}/{epochs} loss {tot / len(tr):.4f} val acc {acc:.4f} ({time.time() - t0:.1f}s)")
        if not select:
            best = acc
            continue
        if acc > best:
            best, best_state, bad = acc, {k: v.detach().clone() for k, v in br.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience and ep + 1 >= min_epochs:
                break
    if select:
        br.load_state_dict(best_state)
    br.eval()
    return {"val_acc": best, "history": hist, "selection": "best_val_acc" if select else "last_epoch",
            "steps": len(hist) * per_epoch}


# ---------------------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------------------

def parse_config(c: str) -> Dict:
    parts = c.split(":")
    if parts[0] == "probe":
        return {"kind": "probe", "depth": 0}
    if parts[0] == "blocks":
        return {"kind": "blocks", "depth": int(parts[1])}
    if parts[0] == "kvq":
        return {"kind": "kvq", "depth": int(parts[1]), "r": int(parts[2]), "frozen": "frozen" in parts[3:],
                "window": "global" not in parts[3:]}
    raise ValueError(c)


def flops_per_decision(cfg: Dict, L: int, D: int, I: int, r: int = 1) -> float:
    """Multiply-adds x2 for one extra decision on one message of L tokens (attention counted globally)."""
    layer_lin = lambda n: 2 * n * (3 * D * D + D * D + D * 2 * I + I * D)
    if cfg["kind"] == "probe":
        return 2 * L * D + 2 * D * 16
    if cfg["kind"] == "blocks":
        return cfg["depth"] * (layer_lin(L) + 4 * L * L * D)
    return cfg["depth"] * (layer_lin(cfg["r"]) + 4 * cfg["r"] * L * D)


def exactness(trunk: Trunk, split: int, depth: int, log=print) -> float:
    """At init, a window-masked r=1 stream must equal the trunk's own CLS path at depth split+depth."""
    br = KVQueryBranch(trunk, split, depth, ["a", "b"], r=1).to(trunk.device).eval()
    texts = ["the deploy is failing on staging and customers see 502s " * 12, "refund please", "ok"]
    b = trunk.tokenize(texts, 256)
    with torch.no_grad():
        taps, ctx = trunk.taps(b["input_ids"], b["attention_mask"], list(range(split, split + depth + 1)))
        x = br.stream([taps[d] for d in br.depths], ctx)[:, 0]
        ref = taps[split + depth][:, 0]
    err = float((x - ref).abs().max() / ref.abs().max())
    log(f"  exactness: stream@{split}+{depth} vs trunk CLS at depth {split + depth}: max rel err {err:.2e} "
        f"(seq len {b['input_ids'].shape[1]})")
    return err


def fused_check(trunk: Trunk, split: int, depth: int, r: int, log=print) -> float:
    brs = [KVQueryBranch(trunk, split, depth, [f"l{i}" for i in range(4)], r=r).to(trunk.device).eval()
           for _ in range(3)]
    for br in brs:
        with torch.no_grad():
            for p in br.trainable():
                p.add_(torch.randn_like(p) * 0.02)
    b = trunk.tokenize(["the deploy is failing on staging " * 20, "refund please"], 256)
    with torch.no_grad():
        taps, ctx = trunk.taps(b["input_ids"], b["attention_mask"], brs[0].depths)
        hs = [taps[d] for d in brs[0].depths]
        loop = torch.stack([br.logits(hs, ctx) for br in brs])
        fused = FusedKVQ(brs)(hs, ctx)
    err = float((loop - fused).abs().max())
    log(f"  fused kvq (r={r}) vs loop: max abs diff {err:.2e}")
    return err


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, dev, reps, warm=2):
    for _ in range(warm):
        fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


@torch.no_grad()
def latency(trunk: Trunk, split: int, texts: Dict[str, str], ns: Sequence[int], reps: int, log=print) -> Dict:
    dev, labels = trunk.device, [f"l{i}" for i in range(5)]
    res = {}
    for name, text in texts.items():
        b = trunk.tokenize([text], 512)
        ids, att = b["input_ids"], b["attention_mask"]
        rows = {"tokens": int(att.sum())}
        maxn = max(ns)
        blocks2 = [BlockBranch(split, labels, trunk.hidden, 2, trunk).to(dev).eval() for _ in range(maxn)]
        probes = [ProbeBranch(split, labels, trunk.hidden).to(dev).eval() for _ in range(maxn)]
        kvqs = {(d, r): [KVQueryBranch(trunk, split, d, labels, r=r).to(dev).eval() for _ in range(maxn)]
                for d, r in ((2, 1), (2, 4), (4, 4))}
        for n in ns:
            row = {}
            with autocast(dev):
                def probe_fn():
                    taps, ctx = trunk.taps(ids, att, [split])
                    for p in probes[:n]:
                        p.probs(taps[split], ctx)
                row["probe"] = timeit(probe_fn, dev, reps)
                fb = FusedBlocks(blocks2[:n]) if n > 1 else None

                def blocks_fn():
                    taps, ctx = trunk.taps(ids, att, [split])
                    fb.probs(taps[split], ctx) if fb is not None else blocks2[0].probs(taps[split], ctx)
                row["blocks:2 (fused)"] = timeit(blocks_fn, dev, reps)
                for (d, r), brs in kvqs.items():
                    fk = FusedKVQ(brs[:n])

                    def kvq_fn(fk=fk, depths=brs[0].depths):
                        taps, ctx = trunk.taps(ids, att, depths)
                        fk([taps[x] for x in depths], ctx)
                    row[f"kvq:{d}:{r} (fused)"] = timeit(kvq_fn, dev, reps)
                row["trunk only to k"] = timeit(lambda: trunk.taps(ids, att, [split]), dev, reps)
            rows[n] = {k: round(v, 2) for k, v in row.items()}
            log(f"  latency {name} ({rows['tokens']} tok) N={n}: " + json.dumps(rows[n]))
        res[name] = rows
    return res


def subset(ds: data.Dataset, tasks: List[str], n_tr: int, n_va: int, n_te: int, max_len: int) -> data.Dataset:
    keep = lambda xs, n: [e for e in xs if all(t in e.y for t in tasks)][:n]
    return data.Dataset(ds.name, {t: ds.tasks[t] for t in tasks}, keep(ds.train, n_tr), keep(ds.val, n_va),
                        keep(ds.test, n_te), max_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="typed-decisions")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--configs", nargs="*",
                    default=["probe", "blocks:2", "kvq:2:1", "kvq:4:4", "kvq:4:4:frozen", "kvq:2:4",
                             "kvq:4:4:global", "kvq:4:1:frozen"])   # most informative first
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--budget-min", type=float, default=27.0,
                    help="skip remaining configs once this many minutes have passed (latency still runs)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--min-steps", type=int, default=300, help="minimum optimiser steps per branch (as train_branch)")
    ap.add_argument("--latency-ns", type=int, nargs="*", default=[1, 5, 20])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-cache-on-device", dest="cache_on_device", action="store_false",
                    help="keep kvq training batches on the CPU (default: copy the fp16 cache to the GPU)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.smoke:
        a.device, a.split, a.seeds, a.epochs, a.probe_epochs, a.reps, a.latency_ns = "cpu", 4, [0], 1, 2, 2, [1, 3]
        a.min_steps = 6
        a.configs = ["probe", "blocks:2", "kvq:2:1", "kvq:2:4", "kvq:2:4:global", "kvq:2:4:frozen"]
    if not a.smoke and not a.out:
        ap.error("--out is required outside --smoke")
    logf = open(a.out.replace(".json", ".log"), "a") if a.out else None

    def log(msg):
        print(msg, flush=True)
        if logf:
            logf.write(msg + "\n")
            logf.flush()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    trunk = Trunk(device=a.device)
    ds = data.load(a.dataset)
    if a.smoke:
        ds = subset(ds, list(ds.tasks)[:2], 48, 32, 32, 128)
    tasks = a.tasks or list(ds.tasks)
    cfgs = {c: parse_config(c) for c in a.configs}
    dmax = max([c["depth"] for c in cfgs.values() if c["kind"] == "kvq"] + [1])
    depths = list(range(a.split, a.split + dmax))
    log(f"== {ds.summary()} | split {a.split} | trunk depths cached {depths} | {trunk.device}")
    res = {"dataset": ds.name, "split": a.split, "device": str(trunk.device), "configs": {}, "seeds": a.seeds}

    res["exactness_rel_err"] = exactness(trunk, a.split, 2, log)
    res["fused_abs_err"] = fused_check(trunk, a.split, 2, 4, log)
    assert res["exactness_rel_err"] < 1e-3, "stream does not reproduce the trunk CLS path at init"
    assert res["fused_abs_err"] < 1e-3, "fused kvq disagrees with the looped branches"

    texts = [e.text for e in ds.train + ds.val + ds.test]
    cache = FeatureCache(trunk, texts, depths, ds.max_len)
    log(f"  cached depths {depths} for {len(texts)} messages in {cache.seconds:.1f}s ({cache.bytes() / 1e6:.0f} MB)")
    if a.cache_on_device and trunk.device.type != "cpu":
        cache.dev_h = {d: [t.to(trunk.device) for t in cache.h[d]] for d in depths}
        log(f"  kvq batches are assembled from a copy of the cache on {trunk.device}")
    L_med = int(np.median(cache.lengths))
    n_tr, n_va = len(ds.train), len(ds.val)
    split_idx = {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va), "test": range(n_tr + n_va, len(texts))}
    allx = ds.train + ds.val + ds.test
    I = trunk.cfg.intermediate_size

    t_start = time.time()
    for cname, cfg in cfgs.items():
        if (time.time() - t_start) / 60 > a.budget_min:
            log(f"-- {cname}: skipped (time budget of {a.budget_min} min reached)")
            res.setdefault("skipped", []).append(cname)
            continue
        per_seed = []
        t_cfg = time.time()
        for seed in a.seeds:
            out = {}
            for task in tasks:
                sel = {s: [i for i in split_idx[s] if task in allx[i].y] for s in split_idx}
                y, soft = {}, {}
                for s in sel:
                    y[s], soft[s] = _targets([allx[i] for i in sel[s]], task)
                labels = ds.tasks[task].labels
                t0 = time.time()
                quiet = lambda s: None
                if cfg["kind"] == "kvq":
                    br = KVQueryBranch(trunk, a.split, cfg["depth"], labels, r=cfg["r"], frozen=cfg["frozen"],
                                       window=cfg["window"])
                    info = train_kvq(br, cache, sel["train"], y["train"], soft["train"], sel["val"], y["val"],
                                     a.epochs, seed=seed, log=quiet, min_steps=a.min_steps)
                    pred = lambda idx: predict_kvq(br, cache, idx)
                    params = br.n_params()
                else:
                    br = ProbeBranch(a.split, labels, trunk.hidden) if cfg["kind"] == "probe" else \
                        BlockBranch(a.split, labels, trunk.hidden, cfg["depth"], trunk)
                    ep, lr_h = (a.probe_epochs, 3e-3) if cfg["kind"] == "probe" else (a.epochs, 1e-3)
                    info = train_branch(br, cache, sel["train"], y["train"], soft["train"], sel["val"], y["val"],
                                        epochs=ep, lr_head=lr_h, seed=seed, min_steps=a.min_steps)
                    pred = lambda idx: predict_logits(br, cache, idx)
                    params = sum(p.numel() for p in br.parameters())
                t = fit_temperature(pred(sel["val"]), y["val"], soft["val"]) if len(sel["val"]) >= 30 else 1.0
                probs = torch.softmax(pred(sel["test"]) / t, -1).numpy()
                m = evaluate(probs, y["test"].numpy(), None if soft["test"] is None else soft["test"].numpy())
                m.update({"val_acc": info["val_acc"], "T": t, "train_s": round(time.time() - t0, 1), "params": params,
                          "epochs_run": len(info["history"]), "selection": info.get("selection")})
                out[task] = m
                del br
            per_seed.append(out)
            mean_acc = np.mean([v["acc"] for v in out.values()])
            log(f"  {cname} seed {seed}: mean acc {mean_acc:.4f} " +
                " ".join(f"{t.split('.')[-1][:10]}={v['acc']:.2f}" for t, v in list(out.items())[:6]))
        tasks_mean = {t: {m: float(np.mean([ps[t][m] for ps in per_seed])) for m in
                          ("acc", "macro_f1", "ece", "nll") + (("brier_soft",) if "brier_soft" in per_seed[0][t] else ())}
                      for t in tasks}
        summary = {m: float(np.mean([v[m] for v in tasks_mean.values()])) for m in ("acc", "macro_f1", "ece", "nll")}
        seed_accs = [float(np.mean([v["acc"] for v in ps.values()])) for ps in per_seed]
        summary["acc_seed_std"] = float(np.std(seed_accs))
        params = int(np.mean([v["params"] for v in per_seed[0].values()]))
        fl = flops_per_decision(cfg, L_med, trunk.hidden, I)
        trunk_depth = a.split + (cfg["depth"] - 1 if cfg["kind"] == "kvq" else 0)
        res["configs"][cname] = {**cfg, "mean": summary, "tasks": tasks_mean, "per_seed_mean_acc": seed_accs,
                                 "trainable_params": params, "gflops_per_extra_decision_at_median_len": fl / 1e9,
                                 "trunk_depth_needed": trunk_depth, "wall_s": round(time.time() - t_cfg, 1)}
        log(f"-- {cname}: mean acc {summary['acc']:.4f} (+-{summary['acc_seed_std']:.4f} over seeds) "
            f"F1 {summary['macro_f1']:.4f} ECE {summary['ece']:.3f} | {params / 1e6:.2f}M trainable | "
            f"{fl / 1e9:.3f} GFLOP/decision at L={L_med} | trunk to {trunk_depth} | {time.time() - t_cfg:.0f}s")
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)

    long_text = max(texts, key=len) if a.smoke else sorted(texts, key=len)[len(texts) // 2]
    res["latency_ms"] = latency(trunk, a.split, {"median_len_message": long_text,
                                                 "short": "hey can someone on billing look at the double charge"},
                                a.latency_ns, a.reps, log)
    log("== summary (mean acc over tasks and seeds)")
    for c, r in res["configs"].items():
        log(f"  {c:18s} acc {r['mean']['acc']:.4f}  F1 {r['mean']['macro_f1']:.4f}  ECE {r['mean']['ece']:.3f}  "
            f"params {r['trainable_params'] / 1e6:6.2f}M  GFLOP/decision {r['gflops_per_extra_decision_at_median_len']:.3f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
