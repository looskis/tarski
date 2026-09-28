"""Passenger tokens: read-only tokens that ride through the frozen trunk next to the message.

The message runs through the frozen trunk exactly as it does for every other branch. Each decision adds
a few "passenger" tokens that attend to the message's keys/values at every layer (with the trunk's own
frozen weights) but are invisible to the message: message rows never attend to passenger columns. So
the message's hidden states stay identical to the shared trunk pass (read once, shared by all
branches), and a decision costs P passenger tokens x the layers they ride, not d layers x all tokens as
a blocks branch does. For a 240-token JSON state and P=4 over 22 layers that is 88 token-layers per
decision, against 480 for blocks@k+2.

Two instantiations, both in this script:

  soft   (trained)  per-task learned passenger tokens (P x 768), optionally with a rank-r LoRA that is
                    applied only to that task's passenger rows (the message rows stay frozen, so the
                    message pass is still shared). Branches are independent per task.
  cloze  (zero-shot, no training) the passengers are the tokens of a cloze template
                    (" This message is about [MASK].") appended after the message; the frozen MLM head
                    reads the [MASK] passenger and verbalizer words score the labels. Compared against
                    the stock two-way forward, where the message also sees the template (classic PET).
                    One-way means every extra zero-shot question costs only its template tokens.

Parity checks (run in both modes): the message path reproduces `Trunk.taps`; the split passenger path
reproduces a full-sequence forward with a one-way mask; the full-sequence masked forward reproduces the
stock model with a two-way mask.

Baselines recomputed in the same run: the tarski probe at depth 22 (sweep settings) and a well-tuned
logistic regression on standardized mean-pooled states at depths 11 and 22 (L2 picked on validation).

Usage (from the repo root):
  .venv/bin/python explore/geometry_passenger.py --smoke
  .venv/bin/python explore/geometry_passenger.py --out results/tarski/explore_passenger.json   # everything, ~25-30 min
  # or as two queue entries (~15 min each):
  .venv/bin/python explore/geometry_passenger.py --datasets typed-decisions --cloze --out results/tarski/explore_passenger_typed.json
  .venv/bin/python explore/geometry_passenger.py --datasets clinc150 --out results/tarski/explore_passenger_clinc.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from tarski import data as tdata
from tarski import train as ttrain
from tarski.autosplit import pooled_by_depth
from tarski.branches import ProbeBranch
from tarski.trunk import Trunk


# ---------------------------------------------------------------------------------------------------
# The rider: message path (frozen, no grad, shared) + passenger paths (one-way, optionally trainable)
# ---------------------------------------------------------------------------------------------------

@dataclass
class Spec:
    """One block of passengers riding with a batch of messages."""
    h: torch.Tensor                      # (B, P, d) passenger states entering layer 0
    pos: torch.Tensor                    # (B, P) rotary positions
    group: torch.Tensor                  # (P,) passengers attend only to passengers of the same group
    hi: int                              # ride layers [0, hi)
    window: str = "global"               # "global": passengers read every message token in local layers
                                         # "stock": passengers obey the 128-token window like real tokens
    lora: Optional[Callable] = None      # (layer, "qkv"|"o", x) -> delta, applied to passenger rows only


class Rider:
    def __init__(self, trunk: Trunk):
        self.t = trunk
        self.cfg = trunk.cfg
        self.nh = self.cfg.num_attention_heads
        self.hd = trunk.hidden // self.nh
        self.sw = self.cfg.sliding_window
        self.types = sorted(set(self.cfg.layer_types))

    def _heads(self, qkv, B, T):
        q, k, v = qkv.view(B, T, 3, self.nh, self.hd).unbind(2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    @staticmethod
    def _finish(layer, h, a, o_delta=None):
        B, T = h.shape[:2]
        a = a.transpose(1, 2).reshape(B, T, -1)
        o = layer.attn.Wo(a)
        if o_delta is not None:
            o = o + o_delta(a)
        h = h + o
        return h + layer.mlp(layer.mlp_norm(h))

    def run(self, ids: torch.Tensor, att: torch.Tensor, specs: Sequence[Spec], message_depth: int = 0):
        """Returns (message states at max(hi, message_depth), [passenger states at each spec's hi])."""
        dev = self.t.device
        B, L = ids.shape
        valid = att.bool()
        m_pos = torch.arange(L, device=dev)[None].expand(B, L)
        eye = torch.eye(L, dtype=torch.bool, device=dev)
        near = (m_pos[:, :, None] - m_pos[:, None, :]).abs() <= self.sw
        base = valid[:, None, None, :]
        masks_m = {"full_attention": base | eye, "sliding_attention": (base & near[:, None]) | eye}
        top = max([s.hi for s in specs] + [message_depth])
        with torch.no_grad(), ttrain.autocast(dev):
            h = self.t.model.embeddings(input_ids=ids)
            rope_m = {lt: self.t.model.rotary_emb(h, m_pos[:1], lt) for lt in self.types}
        ps, masks_p, rope_p = [], [], []
        for s in specs:
            P = s.h.shape[1]
            p = s.h.float()
            ps.append(p)
            mk = valid[:, None, None, :].expand(B, 1, P, L)
            pp = (s.group[:, None] == s.group[None, :])[None].expand(B, P, P)
            mdict = {}
            for lt in self.types:
                a, b = mk, pp
                if lt == "sliding_attention" and s.window == "stock":
                    a = a & ((s.pos[:, :, None] - m_pos[:, None, :]).abs() <= self.sw)[:, None]
                    b = b & ((s.pos[:, :, None] - s.pos[:, None, :]).abs() <= self.sw)
                mdict[lt] = torch.cat([a, b[:, None]], -1)
            masks_p.append(mdict)
            with torch.no_grad():
                rope_p.append({lt: self.t.model.rotary_emb(p, s.pos, lt) for lt in self.types})
        for i in range(top):
            layer = self.t.model.layers[i]
            lt = layer.attention_type
            with torch.no_grad(), ttrain.autocast(dev):
                q, k, v = self._heads(layer.attn.Wqkv(layer.attn_norm(h)), B, L)
                q, k = apply_rotary_pos_emb(q, k, *rope_m[lt])
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=masks_m[lt])
                h_next = self._finish(layer, h, a)
            kf, vf = k.float(), v.float()
            for j, s in enumerate(specs):
                if i >= s.hi:
                    continue
                p = ps[j]
                P = p.shape[1]
                x = layer.attn_norm(p)
                qkv = layer.attn.Wqkv(x)
                if s.lora is not None:
                    qkv = qkv + s.lora(i, "qkv", x)
                qp, kp, vp = self._heads(qkv, B, P)
                qp, kp = apply_rotary_pos_emb(qp, kp, *rope_p[j][lt])
                # attention written out so that autograd keeps one shared reference to the message K/V per
                # layer (a torch.cat per spec would store a copy per spec for the backward pass)
                sc = torch.cat([qp @ kf.transpose(-1, -2), qp @ kp.transpose(-1, -2)], -1) * self.hd ** -0.5
                w = torch.softmax(sc.masked_fill(~masks_p[j][lt], float("-inf")), -1)
                a = w[..., :L] @ vf + w[..., L:] @ vp
                od = (lambda z, i=i, s=s: s.lora(i, "o", z)) if s.lora is not None else None
                ps[j] = self._finish(layer, p, a, od)
            h = h_next
        return h, ps


def masked_stock_forward(trunk: Trunk, ids: torch.Tensor, allowed: Callable[[str], torch.Tensor], depth: int):
    """Stock ModernBERT layers over a full sequence with caller-supplied boolean masks (parity reference)."""
    with torch.no_grad():
        h = trunk.model.embeddings(input_ids=ids)
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        rope = {lt: trunk.model.rotary_emb(h, pos, lt) for lt in set(trunk.cfg.layer_types)}
        for i in range(depth):
            layer = trunk.model.layers[i]
            h = layer(h, attention_mask=allowed(layer.attention_type), position_embeddings=rope[layer.attention_type])
    return h


def parity_checks(trunk: Trunk, rider: Rider, texts: Sequence[str], log) -> Dict[str, float]:
    """Relative max |diff| (max |a - b| / max |b|; the residual stream reaches ~4e4, so absolute diffs are
    not meaningful) of (1) rider message path vs Trunk.taps, (2) rider passengers vs a one-way full-sequence
    forward, (3) the full-sequence forward with a two-way mask vs the stock model."""
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))
    dev, n = trunk.device, trunk.n_layers
    tok = trunk.tok
    tmpl = tok(" This message is about [MASK].", add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
    out = {"message_vs_taps": 0.0, "passenger_vs_oneway_full": 0.0, "twoway_full_vs_stock": 0.0}
    for text in texts:
        m = trunk.token_ids([text], 64)[0]
        ids = torch.tensor([m], device=dev)
        att = torch.ones_like(ids)
        L, P = len(m), len(tmpl)
        spec = Spec(h=trunk.model.embeddings(input_ids=torch.tensor([tmpl], device=dev)).detach(),
                    pos=torch.arange(L, L + P, device=dev)[None], group=torch.zeros(P, dtype=torch.long, device=dev),
                    hi=n, window="stock")
        with torch.no_grad():
            h_m, (h_p,) = rider.run(ids, att, [spec], message_depth=n)
            ref, _ = trunk.taps(ids, att, [n])
        out["message_vs_taps"] = max(out["message_vs_taps"], rel(trunk.model.final_norm(h_m), ref[n]))
        cat = torch.tensor([m + tmpl], device=dev)
        T = L + P
        idx = torch.arange(T, device=dev)
        near = (idx[:, None] - idx[None, :]).abs() <= trunk.cfg.sliding_window
        is_msg = idx < L
        oneway = ~(is_msg[:, None] & ~is_msg[None, :])            # message rows cannot see passenger columns

        def allowed_one(lt):
            a = oneway & (near if lt == "sliding_attention" else torch.ones_like(near))
            return a[None, None]

        def allowed_two(lt):
            a = near if lt == "sliding_attention" else torch.ones_like(near)
            return a[None, None]

        full_one = masked_stock_forward(trunk, cat, allowed_one, n)
        out["passenger_vs_oneway_full"] = max(out["passenger_vs_oneway_full"], rel(h_p[0], full_one[0, L:]),
                                              rel(h_m[0], full_one[0, :L]))
        full_two = trunk.model.final_norm(masked_stock_forward(trunk, cat, allowed_two, n))
        ref2, _ = trunk.taps(cat, torch.ones_like(cat), [n])
        out["twoway_full_vs_stock"] = max(out["twoway_full_vs_stock"], rel(full_two, ref2[n]))
    log("  parity: " + json.dumps({k: float(f"{v:.2e}") for k, v in out.items()}))
    return out


# ---------------------------------------------------------------------------------------------------
# Soft passengers (trained, per task)
# ---------------------------------------------------------------------------------------------------

def parse_config(s: str, n_layers: int) -> Dict:
    """'P4r8@11' -> 4 tokens per task, LoRA rank 8 on passenger rows, ride layers [0, 11)."""
    hi = n_layers
    if "@" in s:
        s, h = s.split("@")
        hi = int(h)
    P, r = s[1:].split("r")
    return {"P": int(P), "rank": int(r), "hi": hi}


class PassengerSet(nn.Module):
    """One configuration's passengers for a list of tasks. Every parameter tensor has a leading task axis
    (or is a per-task module), so tasks never share parameters: training them in one batch is the same as
    training each alone with the same data order."""

    def __init__(self, name: str, trunk: Trunk, tasks: List[str], n_labels: List[int], P: int, rank: int,
                 hi: int, alpha: float = 16.0, dropout: float = 0.1, window: str = "global", seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.name, self.tasks, self.P, self.rank, self.hi, self.window = name, list(tasks), P, rank, hi, window
        T, d = len(tasks), trunk.hidden
        with torch.no_grad():
            cls = torch.tensor([[trunk.tok.cls_token_id]], device=trunk.device)
            e = trunk.model.embeddings(input_ids=cls)[0, 0].float().cpu()
        self.tokens = nn.Parameter(e[None, None].repeat(T, P, 1) + 0.1 * e.std() * torch.randn(T, P, d, generator=g))
        if rank:
            self.A_qkv = nn.Parameter(torch.randn(T, hi, d, rank, generator=g) / math.sqrt(d))
            self.B_qkv = nn.Parameter(torch.zeros(T, hi, rank, 3 * d))
            self.A_o = nn.Parameter(torch.randn(T, hi, d, rank, generator=g) / math.sqrt(d))
            self.B_o = nn.Parameter(torch.zeros(T, hi, rank, d))
            self.scale = alpha / rank
        self.ln_w = nn.Parameter(torch.ones(T, d))
        self.ln_b = nn.Parameter(torch.zeros(T, d))
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList(nn.Linear(d, n) for n in n_labels)

    def params_per_task(self) -> int:
        d = self.tokens.shape[-1]
        n = self.P * d + 2 * d + int(np.mean([h.weight.numel() + h.bias.numel() for h in self.heads]))
        if self.rank:
            n += self.hi * self.rank * (d + 3 * d + d + d)
        return n

    def spec(self, tids: torch.Tensor, B: int) -> Spec:
        Tg, P, d = len(tids), self.P, self.tokens.shape[-1]
        dev = self.tokens.device
        h = self.tokens[tids].reshape(1, Tg * P, d).expand(B, -1, -1)
        lora = None
        if self.rank:
            def lora(i, kind, x, tids=tids):
                A, Bm = (self.A_qkv, self.B_qkv) if kind == "qkv" else (self.A_o, self.B_o)
                z = torch.einsum("btpd,tdr->btpr", x.reshape(B, Tg, P, -1), A[tids, i])
                z = torch.einsum("btpr,tre->btpe", z, Bm[tids, i]) * self.scale
                return z.reshape(B, Tg * P, -1)
        return Spec(h=h, pos=torch.zeros(B, Tg * P, dtype=torch.long, device=dev),
                    group=torch.arange(Tg, device=dev).repeat_interleave(P), hi=self.hi, window=self.window,
                    lora=lora)

    def logits(self, out: torch.Tensor, tids: torch.Tensor) -> List[torch.Tensor]:
        B, Tg = out.shape[0], len(tids)
        x = out.reshape(B, Tg, self.P, -1)
        x = F.layer_norm(x, x.shape[-1:])
        x = x * self.ln_w[tids][None, :, None] + self.ln_b[tids][None, :, None]
        x = self.drop(x.mean(2))
        return [self.heads[int(t)](x[:, j]) for j, t in enumerate(tids)]


def pad_batch(ids_list: List[List[int]], pad_id: int, dev) -> tuple:
    L = max(len(x) for x in ids_list)
    ids = torch.full((len(ids_list), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids_list), L), dtype=torch.long)
    for j, x in enumerate(ids_list):
        ids[j, : len(x)] = torch.tensor(x)
        att[j, : len(x)] = 1
    return ids.to(dev), att.to(dev)


def task_groups(ds: tdata.Dataset, tasks: List[str], allx: List[tdata.Example]) -> List[Dict]:
    """Tasks labelled on exactly the same messages share a group (typed-decisions: one per workflow)."""
    sig = {}
    for t in tasks:
        key = tuple(i for i, e in enumerate(allx) if t in e.y)
        sig.setdefault(key, []).append(t)
    return [{"tasks": ts, "msgs": list(key)} for key, ts in sig.items()]


def binary_auroc(probs: np.ndarray, y: np.ndarray) -> Optional[float]:
    if probs.shape[1] != 2 or len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, probs[:, 1]))


def score_task(val_logits, y_val, soft_val, test_logits, y_test, soft_test) -> Dict:
    zv, zt = torch.tensor(val_logits), torch.tensor(test_logits)
    t = ttrain.fit_temperature(zv, torch.tensor(y_val), None if soft_val is None else torch.tensor(soft_val)) \
        if len(y_val) >= 30 else 1.0
    probs = torch.softmax(zt / t, -1).numpy()
    m = ttrain.evaluate(probs, y_test, soft_test)
    m["temperature"] = t
    au = binary_auroc(probs, y_test)
    if au is not None:
        m["auroc"] = au
    return m


def train_passengers(trunk: Trunk, rider: Rider, ds: tdata.Dataset, tasks: List[str], configs: List[str],
                     epochs: int, bs: int, lr_tok: float, lr_lora: float, lr_head: float, seed: int, log,
                     min_val: int = 50, min_steps: int = 300) -> Dict:
    dev = trunk.device
    allx = ds.train + ds.val + ds.test
    n_tr, n_va = len(ds.train), len(ds.val)
    split_of = lambda i: "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    ids_all = trunk.token_ids([e.text for e in allx], ds.max_len)
    groups = task_groups(ds, tasks, allx)
    tindex = {t: i for i, t in enumerate(tasks)}
    sets = nn.ModuleList()
    for c in configs:
        cfg = parse_config(c, trunk.n_layers)
        sets.append(PassengerSet(c, trunk, tasks, [len(ds.tasks[t].labels) for t in tasks], cfg["P"], cfg["rank"],
                                 cfg["hi"], seed=seed))
    sets.to(dev)
    tok_p = [s.tokens for s in sets]
    lora_p = [p for s in sets for n, p in s.named_parameters() if n.startswith(("A_", "B_"))]
    head_p = [p for s in sets for n, p in s.named_parameters() if n.startswith(("ln_", "heads"))]
    groups_opt = [{"params": tok_p, "lr": lr_tok, "weight_decay": 0.0},
                  {"params": head_p, "lr": lr_head, "weight_decay": 0.01}]
    if lora_p:
        groups_opt.append({"params": lora_p, "lr": lr_lora, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups_opt)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    def chunks(idx: List[int], shuffle: bool, size: int):
        order = sorted(idx, key=lambda i: len(ids_all[i]) + (rng.random() * 8 if shuffle else 0))
        return [order[k:k + size] for k in range(0, len(order), size)]

    train_batches = lambda: [(g, c) for g in groups
                             for c in chunks([i for i in g["msgs"] if split_of(i) == "train"], True, bs)]
    # every task's parameters only step on its own group's batches: make sure each task gets at least
    # `min_steps` optimiser steps (same rule as tarski.train.train_branch)
    per_task_epoch = min(len(chunks([i for i in g["msgs"] if split_of(i) == "train"], False, bs)) for g in groups)
    if per_task_epoch * epochs < min_steps:
        epochs = -(-min_steps // per_task_epoch)
    log(f"  {epochs} epochs x {per_task_epoch} steps/epoch per task (min_steps {min_steps})")
    steps = epochs * len(train_batches())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups_opt], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")

    def targets(idx, t):
        y = torch.tensor([allx[i].y[t] for i in idx])
        soft = torch.tensor(np.stack([allx[i].soft[t] for i in idx])) if all(t in allx[i].soft for i in idx) else None
        return y, soft

    @torch.no_grad()
    def predict(split: str) -> Dict:
        sets.eval()
        out = {(s.name, t): {} for s in sets for t in tasks}
        for g in groups:
            tids = torch.tensor([tindex[t] for t in g["tasks"]], device=dev)
            for c in chunks([i for i in g["msgs"] if split_of(i) == split], False, bs * 2):
                ids, att = pad_batch([ids_all[i] for i in c], trunk.tok.pad_token_id, dev)
                _, ps = rider.run(ids, att, [s.spec(tids, len(c)) for s in sets])
                for s, p in zip(sets, ps):
                    for t, z in zip(g["tasks"], s.logits(p, tids)):
                        for i, zi in zip(c, z.float().cpu().numpy()):
                            out[(s.name, t)][i] = zi
        return out

    hist, best = [], {}
    for ep in range(epochs):
        sets.train()
        t0, tot, nb = time.time(), 0.0, 0
        batches = train_batches()
        rng.shuffle(batches)
        for g, c in batches:
            tids = torch.tensor([tindex[t] for t in g["tasks"]], device=dev)
            ids, att = pad_batch([ids_all[i] for i in c], trunk.tok.pad_token_id, dev)
            _, ps = rider.run(ids, att, [s.spec(tids, len(c)) for s in sets])
            loss = 0.0
            for s, p in zip(sets, ps):
                for t, z in zip(g["tasks"], s.logits(p, tids)):
                    y, soft = targets(c, t)
                    z = z.float()
                    loss = loss + (-(soft.to(dev) * F.log_softmax(z, -1)).sum(-1).mean() if soft is not None
                                   else F.cross_entropy(z, y.to(dev)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() / (len(sets) * len(g["tasks"]))
            nb += 1
        pv, pt = predict("val"), predict("test")
        row = {"epoch": ep + 1, "loss": tot / max(nb, 1), "s": round(time.time() - t0, 1), "val_acc": {}}
        for s in sets:
            accs = []
            for t in tasks:
                vi = sorted(pv[(s.name, t)])
                y = np.array([allx[i].y[t] for i in vi])
                acc = float((np.stack([pv[(s.name, t)][i] for i in vi]).argmax(-1) == y).mean())
                accs.append(acc)
                # same rule as tarski.train.train_branch: select the best-validation epoch only with at least
                # `min_val` validation rows; otherwise keep the last (annealed) epoch
                select = len(vi) >= min_val
                if (s.name, t) not in best or not select or acc > best[(s.name, t)]["val_acc"]:
                    best[(s.name, t)] = {"val_acc": acc, "epoch": ep + 1, "val": pv[(s.name, t)], "test": pt[(s.name, t)],
                                         "selection": "best_val_acc" if select else "last_epoch"}
            row["val_acc"][s.name] = float(np.mean(accs))
        hist.append(row)
        log(f"    epoch {ep + 1}/{epochs} loss {row['loss']:.4f} val acc " +
            " ".join(f"{k}={v:.3f}" for k, v in row["val_acc"].items()) + f" ({row['s']}s)")

    results = {}
    for s in sets:
        per = {}
        for t in tasks:
            b = best[(s.name, t)]
            vi, ti = sorted(b["val"]), sorted(b["test"])
            yv, sv = targets(vi, t)
            yt, st = targets(ti, t)
            m = score_task(np.stack([b["val"][i] for i in vi]), yv.numpy(), None if sv is None else sv.numpy(),
                           np.stack([b["test"][i] for i in ti]), yt.numpy(), None if st is None else st.numpy())
            m.update({"val_acc": b["val_acc"], "epoch_used": b["epoch"], "selection": b["selection"]})
            per[t] = m
        cfg = parse_config(s.name, trunk.n_layers)
        med = float(np.median([len(ids_all[i]) for i in range(len(allx))]))
        results[s.name] = {
            "config": cfg, "tasks": per, "mean": mean_metrics(per),
            "params_per_task": s.params_per_task(),
            "token_layers_per_decision": cfg["P"] * cfg["hi"],
            "median_message_tokens": med,
            "blocks2_token_layers_per_decision_at_median": 2 * med}
    return {"configs": results, "history": hist}


def mean_metrics(per: Dict[str, Dict]) -> Dict[str, float]:
    keys = [k for k in ("acc", "macro_f1", "ece", "nll", "brier_soft", "auroc") if any(k in v for v in per.values())]
    return {k: float(np.mean([v[k] for v in per.values() if k in v])) for k in keys}


# ---------------------------------------------------------------------------------------------------
# Baselines on the same data: tarski probe@22 and a tuned logistic regression on pooled states
# ---------------------------------------------------------------------------------------------------

def logreg(xtr, ttr, xva, yva, xte, dev, lambdas=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1)):
    """Full-batch L-BFGS multinomial logistic regression on standardized features; soft or hard targets
    (`ttr` is a (n, C) target distribution); L2 strength chosen by validation accuracy."""
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-4)
    xtr, xva, xte = ((x - mu) / sd for x in (xtr, xva, xte))
    xtr, xva, xte, ttr = xtr.to(dev), xva.to(dev), xte.to(dev), ttr.to(dev)
    best = None
    for lam in lambdas:
        W = torch.zeros(xtr.shape[1], ttr.shape[1], device=dev, requires_grad=True)
        b = torch.zeros(ttr.shape[1], device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=300, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = -(ttr * F.log_softmax(xtr @ W + b, -1)).sum(-1).mean() + lam * (W ** 2).sum() / 2
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            zv, zt = (xva @ W + b).cpu(), (xte @ W + b).cpu()
        acc = float((zv.argmax(-1) == yva).float().mean())
        if best is None or acc > best[0]:
            best = (acc, lam, zv, zt)
    return best


def baselines(trunk: Trunk, ds: tdata.Dataset, tasks: List[str], log, probe_epochs: int, depths=(11, 22)) -> Dict:
    dev = trunk.device
    allx = ds.train + ds.val + ds.test
    n_tr, n_va = len(ds.train), len(ds.val)
    split_idx = {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va), "test": range(n_tr + n_va, len(allx))}
    texts = [e.text for e in allx]
    out = {}
    top = trunk.n_layers
    t0 = time.time()
    fc = ttrain.FeatureCache(trunk, texts, [top], ds.max_len)
    log(f"  baseline cache depth {top}: {time.time() - t0:.1f}s")
    per = {}
    for t in tasks:
        sel = {s: [i for i in split_idx[s] if t in allx[i].y] for s in split_idx}
        y = {s: torch.tensor([allx[i].y[t] for i in sel[s]]) for s in sel}
        soft = {s: (torch.tensor(np.stack([allx[i].soft[t] for i in sel[s]])) if all(t in allx[i].soft for i in sel[s])
                    else None) for s in sel}
        br = ProbeBranch(top, ds.tasks[t].labels, trunk.hidden)
        ttrain.train_branch(br, fc, sel["train"], y["train"], soft["train"], sel["val"], y["val"],
                            epochs=probe_epochs, lr_layers=1e-4, lr_head=3e-3, seed=0)
        zv, zt = ttrain.predict_logits(br, fc, sel["val"]), ttrain.predict_logits(br, fc, sel["test"])
        per[t] = score_task(zv.numpy(), y["val"].numpy(), None if soft["val"] is None else soft["val"].numpy(),
                            zt.numpy(), y["test"].numpy(), None if soft["test"] is None else soft["test"].numpy())
    out[f"probe@{top}"] = {"tasks": per, "mean": mean_metrics(per)}
    log(f"  baseline probe@{top} mean: " + json.dumps({k: round(v, 4) for k, v in out[f'probe@{top}']['mean'].items()}))
    del fc
    feats = pooled_by_depth(trunk, texts, list(depths), ds.max_len)
    for d in depths:
        per = {}
        for t in tasks:
            sel = {s: [i for i in split_idx[s] if t in allx[i].y] for s in split_idx}
            C = len(ds.tasks[t].labels)
            y = {s: torch.tensor([allx[i].y[t] for i in sel[s]]) for s in sel}
            has_soft = all(t in allx[i].soft for i in sel["train"])
            ttr = torch.tensor(np.stack([allx[i].soft[t] for i in sel["train"]])) if has_soft \
                else F.one_hot(y["train"], C).float()
            acc, lam, zv, zt = logreg(feats[d][sel["train"]], ttr, feats[d][sel["val"]], y["val"], feats[d][sel["test"]],
                                      torch.device("cpu"))   # L-BFGS on CPU: small problem, avoids MPS quirks
            sv = np.stack([allx[i].soft[t] for i in sel["val"]]) if has_soft else None
            st = np.stack([allx[i].soft[t] for i in sel["test"]]) if has_soft else None
            per[t] = score_task(zv.numpy(), y["val"].numpy(), sv, zt.numpy(), y["test"].numpy(), st)
            per[t]["l2"] = lam
        out[f"logreg@{d}"] = {"tasks": per, "mean": mean_metrics(per)}
        log(f"  baseline logreg@{d} mean: " + json.dumps({k: round(v, 4) for k, v in out[f'logreg@{d}']['mean'].items()}))
    return out


# ---------------------------------------------------------------------------------------------------
# Cloze passengers: zero-shot, read-once prompting with the frozen MLM head
# ---------------------------------------------------------------------------------------------------

CLINC_VERBALIZERS = {
    "banking": ["bank", "banking", "account", "money", "payment"],
    "credit_cards": ["credit", "card", "cards", "rewards"],
    "kitchen_and_dining": ["food", "cooking", "recipe", "restaurant", "dinner"],
    "home": ["music", "calendar", "shopping", "reminder", "home"],
    "auto_and_commute": ["car", "driving", "traffic", "gas", "directions"],
    "travel": ["travel", "flight", "trip", "hotel", "vacation"],
    "utility": ["weather", "time", "alarm", "phone", "timer"],
    "work": ["work", "job", "insurance", "payroll", "taxes"],
    "small_talk": ["chat", "conversation", "joke", "greeting", "you"],
    "meta": ["settings", "voice", "language", "volume", "assistant"],
}
STOP = {"a", "an", "the", "of", "to", "by", "in", "on", "for", "or", "and", "is", "my", "not", "via", "from", "if"}
TEMPLATES = [" This message is about [MASK].", " Topic: [MASK].", " In one word, the request concerns [MASK]."]


def load_mlm_head(trunk: Trunk):
    from transformers import AutoModelForMaskedLM
    mlm = AutoModelForMaskedLM.from_pretrained(trunk.base)
    same = torch.equal(mlm.model.layers[0].attn.Wqkv.weight.cpu(), trunk.model.layers[0].attn.Wqkv.weight.cpu())
    head, dec = mlm.head.to(trunk.device).eval(), mlm.decoder.to(trunk.device).eval()
    del mlm
    for p in list(head.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)
    return head, dec, same


def verbalizer_ids(trunk: Trunk, labels: List[str], words: Dict[str, List[str]]) -> List[List[int]]:
    out = []
    for lab in labels:
        ws = words.get(lab) or [w.lower() for w in lab.replace("-", "_").split("_")
                                if w and w.lower() not in STOP] or [lab.lower()]
        out.append(sorted({trunk.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws}))
    return out


@torch.no_grad()
def cloze_eval(trunk: Trunk, rider: Rider, name: str, limit: int, log, bs: int = 64) -> Dict:
    ds = tdata.load(name)
    dev, tok = trunk.device, trunk.tok
    if name == "clinc150":
        task, labels = "domain", [l for l in ds.tasks["domain"].labels if l != "oos"]
        full = ds.tasks["domain"].labels
        ex = [e for e in ds.test if full[e.y["domain"]] != "oos"]
        y = np.array([labels.index(full[e.y["domain"]]) for e in ex])
        words = CLINC_VERBALIZERS
    else:
        task, labels = "intent", ds.tasks["intent"].labels
        ex = list(ds.test)
        y = np.array([e.y["intent"] for e in ex])
        words = {}
    if limit:
        ex, y = ex[:limit], y[:limit]
    vids = verbalizer_ids(trunk, labels, words)
    head, dec, same = load_mlm_head(trunk)
    msgs = trunk.token_ids([e.text for e in ex], ds.max_len)
    order = sorted(range(len(msgs)), key=lambda i: len(msgs[i]))

    def label_scores(hmask):
        lp = F.log_softmax(dec(head(hmask.float())).float(), -1)
        return torch.stack([lp[:, v].mean(-1) for v in vids], -1).cpu().numpy()

    res = {"task": task, "n": len(ex), "chance": 1.0 / len(labels), "mlm_head_matches_trunk": same,
           "verbalizers": {l: tok.convert_ids_to_tokens(v) for l, v in zip(labels, vids)}, "templates": {}}
    for tmpl in TEMPLATES:
        t_ids = tok(tmpl, add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
        mj = t_ids.index(tok.mask_token_id)
        P = len(t_ids)
        S = {v: np.zeros((len(ex), len(labels)), dtype=np.float32) for v in ("two_way", "one_way_stock", "one_way_global")}
        t0 = time.time()
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            B = len(idx)
            # two-way (classic PET): the message sees the template
            ids, att = pad_batch([msgs[i] + t_ids for i in idx], tok.pad_token_id, dev)
            with ttrain.autocast(dev):
                taps, _ = trunk.taps(ids, att, [trunk.n_layers])
            pos = torch.tensor([len(msgs[i]) + mj for i in idx], device=dev)
            S["two_way"][idx] = label_scores(taps[trunk.n_layers][torch.arange(B, device=dev), pos])
            # one-way: the template rides as passengers behind the cached message
            ids, att = pad_batch([msgs[i] for i in idx], tok.pad_token_id, dev)
            emb = trunk.model.embeddings(input_ids=torch.tensor([t_ids], device=dev)).float().expand(B, -1, -1)
            ppos = torch.tensor([[len(msgs[i]) + j for j in range(P)] for i in idx], device=dev)
            grp = torch.zeros(P, dtype=torch.long, device=dev)
            specs = [Spec(emb, ppos, grp, trunk.n_layers, "stock"), Spec(emb, ppos, grp, trunk.n_layers, "global")]
            _, ps = rider.run(ids, att, specs)
            for v, p in zip(("one_way_stock", "one_way_global"), ps):
                S[v][idx] = label_scores(trunk.model.final_norm(p[:, mj]))
        r = {}
        for v, sc in S.items():
            cal = sc - sc.mean(0, keepdims=True)          # label-prior correction (unsupervised, transductive)
            r[v] = {"acc": float((sc.argmax(-1) == y).mean()), "acc_prior_corrected": float((cal.argmax(-1) == y).mean())}
            if v != "two_way":
                r[v]["agree_with_two_way"] = float((sc.argmax(-1) == S["two_way"].argmax(-1)).mean())
        mean_len = float(np.mean([len(m) for m in msgs]))
        r["token_layers_per_question"] = {"two_way": (mean_len + P) * trunk.n_layers, "one_way": P * trunk.n_layers}
        r["s"] = round(time.time() - t0, 1)
        res["templates"][tmpl] = r
        log(f"  cloze {name}/{task} {tmpl!r}: " + json.dumps({k: v for k, v in r.items() if k != 'token_layers_per_question'}))
    return res


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------

def subsample(ds: tdata.Dataset, n: Dict[str, int], tasks: Optional[List[str]], max_len: Optional[int]) -> tdata.Dataset:
    keep = lambda xs, k: [e for e in xs if not tasks or any(t in e.y for t in tasks)][:k]
    tk = {t: ds.tasks[t] for t in (tasks or ds.tasks)}
    return tdata.Dataset(ds.name, tk, keep(ds.train, n["train"]), keep(ds.val, n["val"]), keep(ds.test, n["test"]),
                         max_len or ds.max_len)


def prior_sweep(name: str) -> Dict:
    path = os.path.join("results", "tarski", f"sweep_{'typed' if name == 'typed-decisions' else name}.json")
    if not os.path.exists(path):
        return {}
    r = json.load(open(path))
    return {k: v["mean"]["acc"] for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny subsets, 2 epochs, parity checks")
    ap.add_argument("--datasets", nargs="*", default=["typed-decisions", "clinc150"])
    ap.add_argument("--configs", nargs="*", default=["P4r0", "P4r8", "P4r8@11", "P1r0"],
                    help="P<tokens>r<lora rank>[@<ride depth>]")
    ap.add_argument("--cloze", nargs="*", default=["clinc150", "banking77"])
    ap.add_argument("--epochs", type=int, default=0, help="default: 8 for typed-decisions, 4 otherwise")
    ap.add_argument("--bs", type=int, default=0, help="default: 16 for typed-decisions, 64 otherwise")
    ap.add_argument("--lr-tok", type=float, default=3e-3)
    ap.add_argument("--lr-lora", type=float, default=1e-3)
    ap.add_argument("--lr-head", type=float, default=3e-3)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--min-steps", type=int, default=300, help="minimum optimiser steps per task (tarski rule)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out_path = args.out or ("results/tarski/explore_passenger_smoke.json" if args.smoke
                            else "results/tarski/explore_passenger.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    logf = open(out_path.replace(".json", ".log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.manual_seed(args.seed)
    trunk = Trunk(device="cpu" if args.smoke else None)
    rider = Rider(trunk)
    results = {"args": vars(args), "device": str(trunk.device), "datasets": {}, "cloze": {}}
    log(f"== passengers | base {trunk.base} on {trunk.device} | smoke={args.smoke}")

    results["parity"] = parity_checks(trunk, rider, ["Can you freeze my account?",
                                                     '{"invoice": {"amount": 1200, "po_match": false}}'], log)
    if args.smoke:
        assert max(results["parity"].values()) < 1e-3, results["parity"]

    for name in args.datasets:
        t0 = time.time()
        ds = tdata.load(name)
        tasks = list(ds.tasks)
        if args.smoke:
            if name == "typed-decisions":
                tasks = [t for t in tasks if t.startswith("invoice_processing.")][:3]
            ds = subsample(ds, {"train": 24, "val": 12, "test": 12}, tasks, 128)
        epochs = args.epochs or (2 if args.smoke else (8 if name == "typed-decisions" else 4))
        bs = args.bs or (8 if args.smoke else (16 if name == "typed-decisions" else 64))
        configs = [c.replace("@11", "@4") for c in args.configs[:2]] + ["P1r0@4"] if args.smoke else args.configs
        log(f"-- {ds.summary()} | configs {configs} | epochs {epochs} bs {bs}")
        r = {"prior_sweep_mean_acc": prior_sweep(name)}
        if not args.no_baselines:
            r["baselines"] = baselines(trunk, ds, tasks, log, 2 if args.smoke else args.probe_epochs)
        r["passengers"] = train_passengers(trunk, rider, ds, tasks, configs, epochs, bs, args.lr_tok, args.lr_lora,
                                           args.lr_head, args.seed, log, min_steps=4 if args.smoke else args.min_steps)
        for c, v in r["passengers"]["configs"].items():
            log(f"  {c}: mean " + json.dumps({k: round(x, 4) for k, x in v["mean"].items()}) +
                f" | {v['params_per_task']} params/task | {v['token_layers_per_decision']} token-layers/decision")
        r["wall_s"] = round(time.time() - t0, 1)
        results["datasets"][name] = r
        json.dump(results, open(out_path, "w"), indent=1)

    for name in args.cloze:
        results["cloze"][name] = cloze_eval(trunk, rider, name, 32 if args.smoke else 0, log)
        json.dump(results, open(out_path, "w"), indent=1)
    log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
