"""One-way laya, retrained: how much of the one-way gap does fine-tuning recover?

`geometry_oneway_laya.py` showed that laya's typed-decisions checkpoint, with the state blind to the
question at every layer (question rows still read the state) and no retraining, scores 0.684 on the
typed-decisions test set vs 0.767 unsplit (block split: 0.453), at 1,942 vs 8,736 token-layers per
question. Here the checkpoint is fine-tuned on typed-decisions train under that one-way mask.

Variants (each starts from the same laya typed-decisions checkpoint):
  joint_ft      control: the same recipe with the stock (joint) attention. Separates "one-way training
                recovers the gap" from "more training on this train set helps laya anyway".
  oneway_ft     encoder + head fine-tuned with the one-way mask at every encoder layer and in the head.
                The state encoding is still question-independent (cacheable), but it is a new encoder.
  oneway_qlora  one-way mask; the state rows keep laya's frozen weights exactly, and a rank-r LoRA is
                applied only to question rows (Wqkv, Wo, MLP Wi/Wo in every encoder layer), plus the head.
                This is the encoder analogue of activated LoRA: the cached state encoding is bit-for-bit
                the untrained one-way laya's, so one cached state serves the base model and the adapter.

Evaluation for every variant (and the untrained checkpoint): test accuracy (vs gold argmax), accuracy by
question type, Brier vs gold soft labels, ECE after per-type temperature on validation, under both the
one-way deployment mask (k=28 + one-way head, fully cacheable) and the joint mask.

Training: soft cross-entropy over option markers, AdamW, linear warm-up then linear decay, bf16 autocast on
CUDA; at least --min-steps optimiser steps; the best validation epoch is kept (600 validation pairs).

Usage (from the repo root; needs a CUDA GPU with >= 40 GB for full fine-tuning at bs 16 -> A100):
  .venv/bin/python explore/geometry_oneway_train.py --smoke
  .venv/bin/python explore/geometry_oneway_train.py --out results/tarski/explore_oneway_train.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
if "--smoke" in sys.argv:
    os.environ["READONCE_DEVICE"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from readonce.core import DEVICE, batches, build_items, collate, head_forward, load_agent, metrics, split_train_val
from readonce.run import apply_temperature, fit_temperature
from geometry_oneway_laya import encode, head_oneway


class RowGate:
    """Holds the current batch's question-row mask (B, L, 1) for the row-restricted LoRA layers."""
    mask = None


class RowLoRA(nn.Module):
    """y = base(x) + scale * (x A B) on question rows only; state rows get exactly the frozen base."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, gate: RowGate):
        super().__init__()
        self.base, self.gate, self.scale = base, gate, alpha / rank
        self.A = nn.Parameter(torch.randn(base.in_features, rank) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(rank, base.out_features))

    def forward(self, x):
        y = self.base(x)
        if self.gate.mask is None:
            return y
        return y + ((x @ self.A.to(x.dtype)) @ self.B.to(x.dtype)) * self.scale * self.gate.mask.to(y.dtype)


def add_row_lora(model, rank: int, alpha: float) -> RowGate:
    gate = RowGate()
    for layer in model.encoder.layers:
        layer.attn.Wqkv = RowLoRA(layer.attn.Wqkv, rank, alpha, gate)
        layer.attn.Wo = RowLoRA(layer.attn.Wo, rank, alpha, gate)
        layer.mlp.Wi = RowLoRA(layer.mlp.Wi, rank, alpha, gate)
        layer.mlp.Wo = RowLoRA(layer.mlp.Wo, rank, alpha, gate)
    return gate


def amp():
    if DEVICE.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if DEVICE.type == "mps":
        return torch.autocast("mps", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def forward(model, b, mask_mode: str, gate: RowGate = None):
    """mask_mode 'oneway': one-way at every encoder layer and in the head; 'joint': stock attention."""
    n_layers = model.encoder.config.num_hidden_layers
    if gate is not None:
        gate.mask = ((b["segment"] == 0) & b["attention_mask"].bool()).unsqueeze(-1)
    if mask_mode == "oneway":
        h = encode(model.encoder, b["input_ids"], b["attention_mask"], b["segment"], n_layers, "oneway")
        z = head_oneway(model, h, b["attention_mask"], b["segment"], b["marker_pos"], b["marker_mask"], b["qtype"])
    else:
        h = encode(model.encoder, b["input_ids"], b["attention_mask"], b["segment"], 0, "block")
        z = head_forward(model, h, b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    return z


@torch.no_grad()
def predict(model, items, mask_mode, pad, bs, gate=None):
    model.eval()
    out = [None] * len(items)
    index = {id(it): i for i, it in enumerate(items)}
    for chunk in batches(items, bs):
        b = collate(chunk, pad)
        with amp():
            z = forward(model, b, mask_mode, gate)
        for j, it in enumerate(chunk):
            out[index[id(it)]] = z[j].float().cpu().numpy()
    return out


def score(model, val, test, mask_mode, pad, bs, gate=None):
    zv, zt = predict(model, val, mask_mode, pad, bs, gate), predict(model, test, mask_mode, pad, bs, gate)
    temps = fit_temperature(zv, val)
    m = metrics(apply_temperature(zt, test, temps), test)
    m["val_acc"] = metrics(apply_temperature(zv, val, temps), val)["acc"]
    return m


def train_variant(name, base_model, train, val, test, pad, args, log):
    model = copy.deepcopy(base_model).to(DEVICE)
    gate = None
    if name == "oneway_qlora":
        for p in model.parameters():
            p.requires_grad_(False)
        gate = add_row_lora(model, args.rank, args.alpha)
        model.to(DEVICE)
        for n, p in model.named_parameters():
            if n.endswith((".A", ".B")) or n.startswith(("head.", "type_emb.", "scorer.")):
                p.requires_grad_(True)
        groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and n.endswith((".A", ".B"))],
                   "lr": args.lr_lora},
                  {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith((".A", ".B"))],
                   "lr": args.lr_head}]
    else:
        for n, p in model.named_parameters():
            p.requires_grad_(not n.startswith(("encoder.embeddings", "act_head")))
        groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")],
                   "lr": args.lr_enc},
                  {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")],
                   "lr": args.lr_head}]
    mask_mode = "joint" if name == "joint_ft" else "oneway"
    n_train = sum(p.numel() for g in groups for p in g["params"])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    per_epoch = -(-len(train) // args.bs)
    epochs = max(args.epochs, -(-args.min_steps // per_epoch))
    total = epochs * per_epoch
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    log(f"-- {name}: {n_train / 1e6:.1f}M trainable params, {epochs} epochs x {per_epoch} steps, mask {mask_mode}")
    hist, best = [], None
    step = 0
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for chunk in batches(train, args.bs, shuffle=True, seed=args.seed + ep):
            b = collate(chunk, pad)
            with amp():
                z = forward(model, b, mask_mode, gate)
            loss = -(b["target"] * F.log_softmax(z.float(), -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        r = {"epoch": ep + 1, "loss": tot / per_epoch, "s": round(time.time() - t0, 1)}
        # one-way variants are scored under the deployment (one-way) mask and under the joint mask
        r["oneway"] = score(model, val, test, "oneway", pad, args.eval_bs, gate)
        r["joint"] = score(model, val, test, "joint", pad, args.eval_bs, gate)
        hist.append(r)
        log(f"   epoch {ep + 1}/{epochs} loss {r['loss']:.4f} ({r['s']}s) | oneway: val {r['oneway']['val_acc']:.4f} "
            f"test {r['oneway']['acc']:.4f} | joint: val {r['joint']['val_acc']:.4f} test {r['joint']['acc']:.4f}")
        key = "joint" if mask_mode == "joint" else "oneway"
        if best is None or r[key]["val_acc"] > best[key]["val_acc"]:
            best = r
        if args.max_steps and step >= args.max_steps:
            break
    del model, opt
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"trainable_params": n_train, "epochs": epochs, "steps": step, "history": hist,
            "selected_epoch": best["epoch"], "oneway": best["oneway"], "joint": best["joint"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variants", nargs="*", default=["joint_ft", "oneway_ft", "oneway_qlora"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval-bs", type=int, default=32)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.limit, args.max_steps, args.min_steps, args.bs, args.eval_bs, args.epochs = 2, 2, 1, 4, 8, 1
    out_path = args.out or ("results/tarski/explore_oneway_train_smoke.json" if args.smoke
                            else "results/tarski/explore_oneway_train.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    logf = open(out_path.replace(".json", ".log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.manual_seed(args.seed)
    agent = load_agent("typed-decisions")
    base = agent.model.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    pad = agent.tok.pad_token_id
    train, val = split_train_val(build_items(agent, "train"))
    test = build_items(agent, "test")
    if args.limit:
        keep = lambda xs: [it for it in xs if it.case in sorted({x.case for x in xs})[:args.limit]]
        train, val, test = keep(train), keep(val), keep(test)
    log(f"== one-way laya retraining on {DEVICE} | train {len(train)} val {len(val)} test {len(test)} pairs | "
        f"variants {args.variants}")
    res = {"args": vars(args), "device": str(DEVICE), "reference": {"unsplit": 0.767, "oneway_no_retrain": 0.684,
                                                                      "block_k28": 0.453}}
    t0 = time.time()
    base.to(DEVICE)
    res["untrained"] = {"oneway": score(base, val, test, "oneway", pad, args.eval_bs),
                        "joint": score(base, val, test, "joint", pad, args.eval_bs)}
    log(f"   untrained: oneway test {res['untrained']['oneway']['acc']:.4f} | joint test {res['untrained']['joint']['acc']:.4f} "
        f"({time.time() - t0:.0f}s)")
    if args.smoke:          # the row-restricted LoRA must leave the state rows exactly as the frozen model computes them
        m = copy.deepcopy(base)
        gate = add_row_lora(m, 4, 8.0, )
        for mod in m.modules():
            if isinstance(mod, RowLoRA):
                nn.init.normal_(mod.B, std=0.5)
        b = collate(test[:2], pad)
        gate.mask = ((b["segment"] == 0) & b["attention_mask"].bool()).unsqueeze(-1)
        n_layers = m.encoder.config.num_hidden_layers
        with torch.no_grad():
            h1 = encode(m.encoder, b["input_ids"], b["attention_mask"], b["segment"], n_layers, "oneway")
            h0 = encode(base.encoder, b["input_ids"], b["attention_mask"], b["segment"], n_layers, "oneway")
        st = (b["segment"] == 1) & b["attention_mask"].bool()
        q = (b["segment"] == 0) & b["attention_mask"].bool()
        res["qlora_check"] = {"state_rows_max_diff": float((h1 - h0)[st].abs().max()),
                              "question_rows_max_diff": float((h1 - h0)[q].abs().max())}
        log("   qlora check: " + json.dumps(res["qlora_check"]))
        assert res["qlora_check"]["state_rows_max_diff"] < 1e-4 < res["qlora_check"]["question_rows_max_diff"]
        del m
    json.dump(res, open(out_path, "w"), indent=1)
    for name in args.variants:
        t1 = time.time()
        res[name] = train_variant(name, base, train, val, test, pad, args, log)
        res[name]["wall_s"] = round(time.time() - t1, 1)
        log(f"   {name}: selected epoch {res[name]['selected_epoch']} | oneway test {res[name]['oneway']['acc']:.4f} "
            f"| joint test {res[name]['joint']['acc']:.4f} ({res[name]['wall_s']}s)")
        json.dump(res, open(out_path, "w"), indent=1)
    log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
