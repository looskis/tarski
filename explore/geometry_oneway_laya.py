"""One-way questions on a frozen laya: the question reads the message, the message never reads the question.

laya encodes `[CLS] question [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] | state [SEP]` once per question.
readonce (the sibling line of work) makes the state cacheable with a *block-diagonal* split: below depth k
the question and the state do not see each other at all, above k they are joint (0.617 at k=14, 0.453 at
k=28 on typed-decisions, vs 0.766 unsplit, no retraining).

Here the split is *one-way*: below depth k, state rows attend only to state rows (so the state's hidden
states do not depend on the question and are computed once per message, exactly as in the block split),
but question rows attend to the question AND the state at every layer. The per-question cost is the same
as the block split (question tokens x k layers, plus everything x the joint layers above k), because a
question row attending to cached state keys/values adds only attention reads. The question is a
read-only passenger on the state's encoding. At k=28 with a one-way head, the state is encoded once and
every further question costs only its own ~40-60 tokens; that is laya's flexibility (any question, any
options, no labels) at read-once cost.

No training: laya's weights are used as they are, which isolates how much of laya's accuracy needs the
state to see the question. Positions: ModernBERT's RoPE and index-distance windows are relative, so state
rows that see only state rows get the same states whatever the question length in front of them; the
script checks this numerically ("cache_invariance").

Usage (from the repo root; ~6 min on the M6 GPU for the default grid):
  .venv/bin/python explore/geometry_oneway_laya.py --smoke
  .venv/bin/python explore/geometry_oneway_laya.py --out results/tarski/explore_oneway_laya.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
if "--smoke" in sys.argv:
    os.environ["READONCE_DEVICE"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from readonce.core import (DEVICE, autocast, batches, build_items, collate, encode_split, head_forward,
                           load_agent, metrics, split_train_val)
from readonce.run import apply_temperature, fit_temperature


def masks_for(cfg, attention_mask: torch.Tensor, segment: torch.Tensor, k: int, mode: str):
    """Per-layer boolean (n,1,L,L) masks, True = attend. segment: 0 = question, 1 = state/pad.
    Layers < k: "block" = question and state see only themselves (readonce); "oneway" = state rows see only
    the state, question rows see everything. Layers >= k are joint."""
    L = attention_mask.shape[1]
    valid_k = attention_mask.bool()[:, None, None, :]
    idx = torch.arange(L, device=attention_mask.device)
    window = (idx[:, None] - idx[None, :]).abs() <= cfg.sliding_window
    same = segment[:, :, None] == segment[:, None, :]
    lower = same if mode == "block" else (same | (segment[:, :, None] == 0))
    eye = torch.eye(L, dtype=torch.bool, device=attention_mask.device)
    out = []
    for li, lt in enumerate(cfg.layer_types):
        m = valid_k.expand(-1, 1, L, L)
        if lt == "sliding_attention":
            m = m & window
        if li < k:
            m = m & lower[:, None]
        out.append(m | eye)
    return out


def encode(encoder, input_ids, attention_mask, segment, k: int, mode: str) -> torch.Tensor:
    cfg = encoder.config
    h = encoder.embeddings(input_ids=input_ids)
    pos = torch.arange(h.shape[1], device=h.device)[None]
    rope = {lt: encoder.rotary_emb(h, pos, lt) for lt in set(cfg.layer_types)}
    masks = masks_for(cfg, attention_mask, segment, k, mode)
    for i, layer in enumerate(encoder.layers):
        h = layer(h, attention_mask=masks[i], position_embeddings=rope[layer.attention_type])
    return encoder.final_norm(h)


def head_oneway(model, h, attention_mask, segment, marker_pos, marker_mask, qtype) -> torch.Tensor:
    """laya's decision head with state rows blocked from question columns (True in src_mask = blocked)."""
    h = h + model.type_emb(qtype)[:, None, :]
    pad = ~attention_mask.bool()
    blocked = (segment[:, :, None] == 1) & (segment[:, None, :] == 0)
    for layer in model.head.layers:
        mask = blocked.repeat_interleave(layer.self_attn.num_heads, 0)
        h = layer(h, src_mask=mask, src_key_padding_mask=pad)
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    logits = model.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
    return logits.masked_fill(~marker_mask, -1e4)


@torch.no_grad()
def logits_for(model, items, k: int, mode: str, head_mode: str, pad_id: int, bs: int):
    out = [None] * len(items)
    index = {id(it): i for i, it in enumerate(items)}
    for chunk in batches(items, bs):
        b = collate(chunk, pad_id)
        with autocast():
            h = encode(model.encoder, b["input_ids"], b["attention_mask"], b["segment"], k, mode)
            if head_mode == "oneway":
                z = head_oneway(model, h, b["attention_mask"], b["segment"], b["marker_pos"], b["marker_mask"], b["qtype"])
            else:
                z = head_forward(model, h, b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        for j, it in enumerate(chunk):
            out[index[id(it)]] = z[j].float().cpu().numpy()
    return out


def cost(items, k: int, head_mode: str, n_layers: int, n_head: int) -> dict:
    """Token-layers per question (question-specific work) and per message (shared, once)."""
    q = np.mean([it.a_len for it in items])
    full = np.mean([len(it.ids) for it in items])
    s = full - q
    per_q = q * k + full * (n_layers - k) + (q if head_mode == "oneway" else full) * n_head
    per_msg = s * k + (s * n_head if head_mode == "oneway" else 0)
    return {"question_tokens": float(q), "state_tokens": float(s), "per_question_token_layers": float(per_q),
            "per_message_token_layers_shared": float(per_msg), "unsplit_per_question": float(full * (n_layers + n_head))}


@torch.no_grad()
def checks(model, items, pad_id, log) -> dict:
    """(1) this script's block mask reproduces readonce.encode_split; (2) with the one-way mask the state
    rows' final states do not depend on which question is in front of them."""
    out = {}
    b = collate(items[:4], pad_id)
    with autocast():
        a = encode(model.encoder, b["input_ids"], b["attention_mask"], b["segment"], 14, "block")
        r = encode_split(model.encoder, b["input_ids"], b["attention_mask"], b["segment"], 14)
    out["block_vs_readonce"] = float((a.float() - r.float()).abs().max() / r.float().abs().max())
    same_case = []
    for case in sorted({it.case for it in items}):
        its = [it for it in items if it.case == case]
        for x in its:                       # two questions about one message whose state segments are identical
            y = next((y for y in its if y is not x and y.ids[y.a_len:] == x.ids[x.a_len:] and y.a_len != x.a_len), None)
            if y is not None:
                same_case = [x, y]
                break
        if same_case:
            break
    if not same_case:
        log("  checks: no two questions share an identical state segment; skipping cache_invariance")
        out["cache_invariance"] = None
        return out
    enc = []
    for it in same_case:
        bb = collate([it], pad_id)
        with autocast():
            h = encode(model.encoder, bb["input_ids"], bb["attention_mask"], bb["segment"], model.encoder.config.num_hidden_layers, "oneway")
        enc.append(h[0, it.a_len: len(it.ids)].float())
    n = min(len(e) for e in enc)
    out["cache_invariance"] = float((enc[0][:n] - enc[1][:n]).abs().max() / enc[1][:n].abs().max())
    out["cache_invariance_question_lengths"] = [it.a_len for it in same_case]
    log("  checks: " + json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="CPU, 2 messages per split, 3 split depths")
    ap.add_argument("--backbone", default="typed-decisions", help="laya subfolder: typed-decisions or english")
    ap.add_argument("--ks", type=int, nargs="*", default=[0, 7, 14, 20, 24, 26, 28])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N messages per split")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ks = [0, 14, 28] if args.smoke else args.ks
    limit = 2 if args.smoke else args.limit
    out_path = args.out or ("results/tarski/explore_oneway_laya_smoke.json" if args.smoke
                            else "results/tarski/explore_oneway_laya.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    logf = open(out_path.replace(".json", ".log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    agent = load_agent(args.backbone)
    model = agent.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    pad = agent.tok.pad_token_id
    n_layers = model.encoder.config.num_hidden_layers
    n_head = len(model.head.layers) if model.head is not None else 0
    _, val = split_train_val(build_items(agent, "train"))
    test = build_items(agent, "test")
    if limit:
        keep = lambda xs: [it for it in xs if it.case in sorted({x.case for x in xs})[:limit]]
        val, test = keep(val), keep(test)
    log(f"== one-way laya | backbone={args.backbone} on {DEVICE} | val={len(val)} test={len(test)} pairs | "
        f"{n_layers} encoder layers + {n_head} head layers | ks={ks}")
    results = {"args": vars(args), "device": str(DEVICE), "checks": checks(model, test, pad, log), "runs": {}}
    if args.smoke:
        assert results["checks"]["block_vs_readonce"] < 1e-3, results["checks"]
        assert (results["checks"]["cache_invariance"] or 0.0) < 1e-2, results["checks"]

    grid = [("block", k, "joint") for k in ks] + [("oneway", k, "joint") for k in ks if k > 0]
    if n_layers in ks:
        grid.append(("oneway", n_layers, "oneway"))
    for mode, k, head_mode in grid:
        key = f"{mode}@{k}/head={head_mode}"
        t0 = time.time()
        zv = logits_for(model, val, k, mode, head_mode, pad, args.bs)
        zt = logits_for(model, test, k, mode, head_mode, pad, args.bs)
        temps = fit_temperature(zv, val)
        m = metrics(apply_temperature(zt, test, temps), test)
        m.update(cost(test, k, head_mode, n_layers, n_head))
        m["s"] = round(time.time() - t0, 1)
        results["runs"][key] = m
        log(f"  {key}: acc {m['acc']:.4f} (choice {m.get('acc_choice', float('nan')):.3f}, score "
            f"{m.get('acc_score', float('nan')):.3f}, noul {m.get('acc_noul', float('nan')):.3f}) brier {m['brier']:.4f} "
            f"| per-question token-layers {m['per_question_token_layers']:.0f} vs unsplit {m['unsplit_per_question']:.0f} "
            f"({m['s']}s)")
        json.dump(results, open(out_path, "w"), indent=1)
    log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
