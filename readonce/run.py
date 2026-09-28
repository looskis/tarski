"""Sweep split depth k on typed-decisions with a frozen laya backbone; train only the appended head.

For each k:
  1. zero-shot: laya's own head on the split encoder (no training)  -> how much does splitting cost?
  2. cache the frozen encoder's final states for every (state, question) pair (fp16, CPU)
  3. train a fresh copy of laya's head (type_emb + 2 transformer layers + scorer) on the cache
     with soft cross-entropy against gold distributions; pick the epoch by validation accuracy
  4. fit one temperature per question type on validation, report test accuracy / Brier / ECE

Usage: python -m readonce.run --backbone english --ks 0 14 20 24 26 28 --out results/english.json
"""

import argparse
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from readonce.core import (DEVICE, Upper, autocast, batches, build_items, collate, empty_cache, encode_lower,
                           encode_split, head_forward, load_agent, mem_report, metrics, split_train_val)


@torch.no_grad()
def encode_all(model, items, k, pad_id, bs=32, path=None, lower=False):
    """Frozen encoder states per item, fp16 on CPU, in item order (cached on disk at `path`).
    Final states by default; with `lower`, the states entering layer k (for training the layers above)."""
    if path and os.path.exists(path):
        return torch.load(path)
    out = [None] * len(items)
    index = {id(it): i for i, it in enumerate(items)}
    for chunk in batches(items, bs):
        b = collate(chunk, pad_id)
        with autocast():
            enc = encode_lower if lower else encode_split
            h = enc(model.encoder, b["input_ids"], b["attention_mask"], b["segment"], k)
        h = h.half().cpu()
        for j, it in enumerate(chunk):
            out[index[id(it)]] = h[j, : len(it.ids)].clone()
    if path:
        torch.save(out, path)
    return out


def cached_batch(items, states, idx, pad_id):
    chunk = [items[i] for i in idx]
    b = collate(chunk, pad_id)
    L = b["input_ids"].shape[1]
    h = torch.zeros(len(idx), L, states[idx[0]].shape[-1], dtype=torch.float16)
    for j, i in enumerate(idx):
        h[j, : states[i].shape[0]] = states[i]
    b["h"] = h.to(DEVICE).float()
    return b


def forward_logits(head_model, upper, b):
    h = b["h"] if upper is None else upper(b["h"], b["attention_mask"])
    return head_forward(head_model, h, b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])


def train_autocast(upper):
    """bf16 autocast when encoder layers are in the loop on CUDA; head-only runs stay fp32 everywhere."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda" and upper is not None)


def predict(head_model, items, states, pad_id, bs=64, upper=None):
    head_model.eval()
    if upper is not None:
        upper.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i].ids))
    logits = [None] * len(items)
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            b = cached_batch(items, states, idx, pad_id)
            with train_autocast(upper):
                z = forward_logits(head_model, upper, b)
            for j, i in enumerate(idx):
                logits[i] = z[j].float().cpu().numpy()
    return logits


def fit_temperature(logits, items):
    """One temperature per question type, fitted on validation NLL against gold soft labels."""
    temps = {}
    for qt in {it.qtype for it in items}:
        sel = [(z[: len(it.markers)], it.target) for z, it in zip(logits, items) if it.qtype == qt]
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = sum(-(torch.tensor(t) * F.log_softmax(torch.tensor(z) / log_t.exp(), -1)).sum() for z, t in sel)
            loss = loss / len(sel)
            loss.backward()
            return loss

        opt.step(closure)
        temps[qt] = float(log_t.detach().exp().clamp(0.2, 10.0))
    return temps


def apply_temperature(logits, items, temps):
    return [z / temps.get(it.qtype, 1.0) for z, it in zip(logits, items)]


def train_head(agent, train, tr_states, val, va_states, epochs, lr, bs, seed, log, k=None, lr_upper=None):
    """Train a copy of laya's head; with `lr_upper`, also train a copy of encoder layers k.. (Upper)."""
    torch.manual_seed(seed)
    enc, agent.model.encoder = agent.model.encoder, None   # copy only laya's trained head, not the encoder
    model = copy.deepcopy(agent.model)
    agent.model.encoder = enc
    for layer in model.head.layers:              # MPS scaled_dot_product_attention has no dropout;
        layer.self_attn.dropout = 0.0             # residual/FFN dropout (dropout1/2) stays at 0.1
    model.to(DEVICE)
    params = [p for n, p in model.named_parameters() if not n.startswith("act_head")]
    upper, groups = None, [{"params": params, "lr": lr}]
    if lr_upper is not None:
        upper = Upper(agent.model.encoder, k).to(DEVICE)
        groups.append({"params": list(upper.parameters()), "lr": lr_upper})
        params = params + groups[-1]["params"]
    for p in params:                             # copies inherit the frozen flags of agent.model
        p.requires_grad_(True)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = epochs * ((len(train) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    pad = agent.tok.pad_token_id
    best, best_state = -1.0, None
    order = list(range(len(train)))
    for ep in range(epochs):
        model.train()
        if upper is not None:
            upper.train()
        rng = np.random.default_rng(seed + ep)
        rng.shuffle(order)
        # length-bucketed shuffled batches
        buckets = sorted(order, key=lambda i: len(train[i].ids) + rng.random())
        chunks = [buckets[i:i + bs] for i in range(0, len(buckets), bs)]
        rng.shuffle(chunks)
        t0, tot = time.time(), 0.0
        for idx in chunks:
            b = cached_batch(train, tr_states, idx, pad)
            with train_autocast(upper):
                z = forward_logits(model, upper, b).float()
            loss = -(b["target"] * F.log_softmax(z, -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        va = metrics(predict(model, val, va_states, pad, upper=upper), val)
        empty_cache()
        log(f"    epoch {ep + 1}/{epochs} loss {tot / len(train):.4f} val acc {va['acc']:.4f} ({time.time() - t0:.0f}s)")
        if va["acc"] > best:
            best = va["acc"]
            best_state = [{n: v.detach().clone() for n, v in m.state_dict().items()}
                          for m in (model, upper) if m is not None]
    model.load_state_dict(best_state[0])
    if upper is not None:
        upper.load_state_dict(best_state[1])
    return model, upper, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="english")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 14, 20, 24, 26, 28])
    ap.add_argument("--sink", action="store_true",
                    help="give the text segment its own [CLS] sink token (changes k=0 too; exclude k=0)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--train-upper", action="store_true",
                    help="also train encoder layers k.. (they run per question anyway); lower layers stay frozen")
    ap.add_argument("--lr-upper", type=float, default=5e-5)
    ap.add_argument("--cache-dir", default=os.environ.get("READONCE_CACHE", ""),
                    help="directory for encoder-state caches, reused across runs with the same backbone/k")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: keep only the first N cases per split")
    ap.add_argument("--train-workflows", nargs="+", default=None, help="train/val only on these workflows")
    ap.add_argument("--eval-workflows", nargs="+", default=None, help="test only on these workflows")
    ap.add_argument("--save-trunk", default="", help="with --train-upper at k=0: save trained encoder layers + head")
    ap.add_argument("--load-trunk", default="", help="start from a saved trunk (encoder layers + head)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logf = open(args.out.replace(".json", ".log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    agent = load_agent(args.backbone)
    if args.load_trunk:
        sd = torch.load(args.load_trunk, map_location="cpu")
        assert sd["k"] == 0, "a trunk must be trained with every encoder layer (k=0)"
        enc = agent.model.encoder
        for i, layer in enumerate(enc.layers):
            pre = f"layers.{i}."
            layer.load_state_dict({n[len(pre):]: v for n, v in sd["upper"].items() if n.startswith(pre)})
        enc.final_norm.load_state_dict({n[len("final_norm."):]: v for n, v in sd["upper"].items()
                                        if n.startswith("final_norm.")})
        missing, unexpected = agent.model.load_state_dict(sd["head"], strict=False)
        assert not unexpected and all(n.startswith("encoder.") for n in missing), (missing, unexpected)
    agent.model.eval()
    for p in agent.model.parameters():
        p.requires_grad_(False)
    pad = agent.tok.pad_token_id
    train, val = split_train_val(build_items(agent, "train", sink=args.sink))
    test = build_items(agent, "test", sink=args.sink)
    if args.train_workflows:
        train, val = ([it for it in x if it.workflow in args.train_workflows] for x in (train, val))
    if args.eval_workflows:
        test = [it for it in test if it.workflow in args.eval_workflows]
    if args.limit:                               # smoke tests: the first N remaining cases of each split
        def first_cases(items):
            keep = set(sorted({it.case for it in items})[: args.limit])
            return [it for it in items if it.case in keep]
        train, val, test = first_cases(train), first_cases(val), first_cases(test)
    variant = "|".join(map(str, (args.load_trunk, args.train_workflows, args.eval_workflows)))
    tag = "" if variant == "|None|None" else "_" + hashlib.md5(variant.encode()).hexdigest()[:8]
    log(f"== backbone={args.backbone} trunk={args.load_trunk or None} train_wf={args.train_workflows} "
        f"eval_wf={args.eval_workflows} sink={args.sink} train={len(train)} val={len(val)} test={len(test)} "
        f"ks={args.ks} epochs={args.epochs} lr={args.lr} seed={args.seed} train_upper={args.train_upper} "
        f"lr_upper={args.lr_upper if args.train_upper else None}")

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for k in args.ks:
        key = f"k={k}"
        t0 = time.time()
        tr, va, te = train, val, test
        agent.model.encoder.to(DEVICE)
        lower = args.train_upper
        cp = (lambda split: os.path.join(args.cache_dir, f"{args.backbone}_k{k}_{'lower' if lower else 'final'}"
                                         f"_sink{int(args.sink)}_lim{args.limit}{tag}_{split}.pt") if args.cache_dir else None)
        tr_s = None if args.no_train else encode_all(agent.model, tr, k, pad, path=cp("train"), lower=lower)
        va_s = encode_all(agent.model, va, k, pad, path=cp("val"), lower=lower)
        te_s = encode_all(agent.model, te, k, pad, path=cp("test"), lower=lower)
        if not lower:
            agent.model.encoder.to("cpu")       # the head never needs it; free unified memory for training
        empty_cache()
        log(f"-- {key}: encoded {len(tr) + len(va) + len(te)} pairs in {time.time() - t0:.0f}s ({mem_report()})")
        r = {}
        zs_upper = Upper(agent.model.encoder, k).to(DEVICE) if lower else None
        zs_val = predict(agent.model, va, va_s, pad, upper=zs_upper)
        zs_test = predict(agent.model, te, te_s, pad, upper=zs_upper)
        del zs_upper
        temps = fit_temperature(zs_val, va)
        r["zero_shot"] = metrics(apply_temperature(zs_test, te, temps), te)
        log(f"   zero-shot test: {json.dumps(r['zero_shot'])}")
        if not args.no_train:
            t1 = time.time()
            head, upper, best_val = train_head(agent, tr, tr_s, va, va_s, args.epochs, args.lr, args.bs, args.seed,
                                               log, k=k, lr_upper=args.lr_upper if lower else None)
            lv, lt = predict(head, va, va_s, pad, upper=upper), predict(head, te, te_s, pad, upper=upper)
            temps = fit_temperature(lv, va)
            r["trained_head"] = metrics(apply_temperature(lt, te, temps), te)
            r["trained_head"]["val_acc"] = best_val
            r["trained_head"]["train_s"] = round(time.time() - t1)
            n_task = sum(p.numel() for n, p in head.named_parameters() if not n.startswith("act_head"))
            n_task += sum(p.numel() for p in upper.parameters()) if upper is not None else 0
            r["trained_head"]["task_params_M"] = round(n_task / 1e6, 1)
            r["trained_head"]["task_module_MB_fp16"] = round(n_task * 2 / 2**20, 1)
            lt_t = apply_temperature(lt, te, temps)
            r["trained_head"]["acc_by_workflow"] = {
                w: metrics([z for z, it in zip(lt_t, te) if it.workflow == w], [it for it in te if it.workflow == w])["acc"]
                for w in sorted({it.workflow for it in te})}
            if args.save_trunk and upper is not None:
                torch.save({"k": k, "upper": upper.state_dict(), "head": head.state_dict()}, args.save_trunk)
                log(f"   saved trunk to {args.save_trunk}")
            del upper
            log(f"   trained-head test: {json.dumps(r['trained_head'])}")
        results[key] = r
        del tr_s, va_s, te_s
        json.dump(results, open(args.out, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
