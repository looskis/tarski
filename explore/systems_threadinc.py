"""Incremental view maintenance for a bidirectional trunk: Slack-thread appends without re-encoding.

A thread grows one message at a time and every new message needs its decisions. Re-encoding the whole
thread each time costs O(T) passes over a growing sequence (O(T^2) tokens in total). In a causal LM the
prefix KV cache makes appends exact; in a bidirectional encoder every old token should also attend to
the new one, so cached prefix states go stale. This script measures how stale is too stale for the
*decisions*, using the trunk's own layer structure:

  stale      prefix states are reused as-is at every layer; only the new tokens are computed (they read
             the stale prefix K/V). Cost per append ~ new tokens only.
  bandW      also recompute the last W prefix tokens at every layer. ModernBERT's local layers (2 of
             every 3) have a 128-token window, so the tokens nearest the append are the ones whose
             states move most.
  selectP    CacheBlend-style: run layer 0 exactly for every token, keep the P% of prefix tokens whose
             layer-0 output moved most, and recompute only those (plus the new tokens) above layer 0.
  full       re-encode everything (the reference). The masked implementation with every token active
             reproduces a plain encode; `--smoke` checks that.

Threads are built from CLINC150 (messages of the same domain, with out-of-scope messages sprinkled in),
so every message in the test set gets its decisions at the step it arrives, with the thread so far as
context. Branches (probe and blocks:2 at split k, pooling over the new message's tokens) are trained on
full encodings, as a normal deployment would, then evaluated on each incremental mode. A second set is
trained on stale-mode states to see whether training on the serving distribution closes any gap.

Reported per mode: mean accuracy over tasks and all thread steps, agreement with full-encoding
predictions, cosine of the new message's pooled state vs full, and trunk FLOPs relative to a full
re-encode (linear layers + attention, stale K/V assumed cached).

Usage:
  python explore/systems_threadinc.py --smoke
  python explore/systems_threadinc.py --dataset clinc150 --split 11 --thread-len 10 \
      --out results/tarski/explore_threadinc_clinc150.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

from tarski import data
from tarski.branches import BlockBranch, ProbeBranch
from tarski.train import autocast, evaluate, fit_temperature, predict_logits, train_branch
from tarski.trunk import Context, Trunk


# ---------------------------------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------------------------------

def make_threads(examples: List[data.Example], T: int, seed: int) -> List[List[int]]:
    """Group message indices into threads of ~T messages. With a `domain` task, a thread holds messages
    of one domain and out-of-scope messages are inserted at random positions of random threads."""
    rng = random.Random(seed)
    if examples and "domain" in examples[0].y and "oos" in examples[0].y:
        by, oos = defaultdict(list), []
        for i, e in enumerate(examples):
            (oos if e.y["oos"] == 1 else by[e.y["domain"]]).append(i)
        threads = []
        for d in sorted(by):
            rng.shuffle(by[d])
            threads += [by[d][s:s + T] for s in range(0, len(by[d]), T)]
        for i in oos:
            th = threads[rng.randrange(len(threads))]
            th.insert(rng.randrange(len(th) + 1), i)
    else:
        idx = list(range(len(examples)))
        rng.shuffle(idx)
        threads = [idx[s:s + T] for s in range(0, len(idx), T)]
    return threads


def thread_ids(tok, texts: List[str], per_msg: int = 48) -> List[List[int]]:
    return [tok(t, add_special_tokens=False, truncation=True, max_length=per_msg)["input_ids"] for t in texts]


# ---------------------------------------------------------------------------------------------------
# A cache of depth-k states for thread steps, with the FeatureCache interface train_branch expects
# ---------------------------------------------------------------------------------------------------

class StepCache:
    """h[i]: (L_i, D) fp16 states of step i's whole sequence at the split; span[i]: the new message."""

    def __init__(self, trunk: Trunk):
        self.trunk, self.h, self.lengths, self.spans = trunk, [], [], []

    def add(self, h: torch.Tensor, span: Tuple[int, int]) -> int:
        self.h.append(h.to("cpu", torch.float16).clone())
        self.lengths.append(h.shape[0])
        self.spans.append(span)
        return len(self.h) - 1

    def batch(self, depth: int, idx: Sequence[int], dtype=torch.float32):
        L = max(self.lengths[i] for i in idx)
        h = torch.zeros(len(idx), L, self.trunk.hidden, dtype=torch.float16)
        att = torch.zeros(len(idx), L, dtype=torch.long)
        span = torch.zeros(len(idx), L, dtype=torch.long)
        for j, i in enumerate(idx):
            h[j, : self.lengths[i]] = self.h[i]
            att[j, : self.lengths[i]] = 1
            s, e = self.spans[i]
            span[j, s:e] = 1
        dev = self.trunk.device
        h, att, span = h.to(dev, dtype), att.to(dev), span.to(dev)
        ctx = self.trunk.context(h, att)
        return h, Context(ctx.masks, ctx.rope, span)        # attend over the thread, pool over the new message

    def pooled(self, idx: Sequence[int]) -> torch.Tensor:
        return torch.stack([self.h[i][slice(*self.spans[i])].float().mean(0) for i in idx])


# ---------------------------------------------------------------------------------------------------
# Incremental encoding
# ---------------------------------------------------------------------------------------------------

def lin_flops(D: int, I: int) -> float:
    return 2 * (3 * D * D + D * D + 2 * D * I + I * D)


@torch.no_grad()
def run_chains(trunk: Trunk, k: int, threads: List[List[List[int]]], mode: str, bs: int,
               sink, log=print) -> Dict[str, float]:
    """Feed every thread one message at a time. `threads`: per thread, the token ids of each message.
    For each step calls sink(thread_idx, step, states_at_k (L, D), span). Returns FLOP totals.

    mode: full | stale | band<W> | select<P>. Masked implementation: each layer is computed for every
    token, and inactive tokens then take their cached (stale) state, which is exactly what computing
    queries only for active tokens against the mixed K/V would give."""
    cls, sep = trunk.tok.cls_token_id, trunk.tok.sep_token_id
    D, I = trunk.hidden, trunk.cfg.intermediate_size
    cl, ca = lin_flops(D, I), 4 * D
    fl_full = fl_inc = 0.0
    kind = "band" if mode.startswith("band") else "select" if mode.startswith("select") else mode
    arg = float(mode[len(kind):]) if kind in ("band", "select") else 0.0
    dev = trunk.device
    order = sorted(range(len(threads)), key=lambda t: -len(threads[t]))
    for s in range(0, len(order), bs):
        grp = order[s:s + bs]
        seqs = {t: [cls] for t in grp}
        prev: Dict[int, List[torch.Tensor]] = {}                            # thread -> states per layer 0..k
        for step in range(max(len(threads[t]) for t in grp)):
            live = [t for t in grp if step < len(threads[t])]
            P = {t: len(seqs[t]) for t in live}
            for t in live:
                seqs[t] = seqs[t] + threads[t][step] + [sep]
            Ls = {t: len(seqs[t]) for t in live}
            Lm = max(Ls.values())
            ids = torch.full((len(live), Lm), trunk.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros(len(live), Lm, dtype=torch.long)
            active = torch.zeros(len(live), Lm, dtype=torch.bool)
            for b, t in enumerate(live):
                ids[b, : Ls[t]] = torch.tensor(seqs[t])
                att[b, : Ls[t]] = 1
                lo = P[t] if step > 0 else 0                                # step 0: nothing is cached yet
                if kind == "full":
                    lo = 0
                elif kind == "band":
                    lo = max(0, lo - int(arg))
                active[b, lo: Ls[t]] = True
            ids, att, active = ids.to(dev), att.to(dev), active.to(dev)
            with autocast(dev):
                x = trunk.model.embeddings(input_ids=ids)
                ctx = trunk.context(x, att)
                states = [x]
                for j in range(k):
                    layer = trunk.model.layers[j]
                    y = layer(x, attention_mask=ctx.masks[layer.attention_type],
                              position_embeddings=ctx.rope[layer.attention_type])
                    if step > 0 and kind != "full":
                        stale = torch.zeros_like(y)
                        for b, t in enumerate(live):
                            stale[b, : P[t]] = prev[t][j + 1].to(y.dtype)
                        if kind == "select" and j == 0:                     # CacheBlend-style token choice
                            dev_tok = (y - stale).float().norm(dim=-1)
                            for b, t in enumerate(live):
                                n_sel = int(round(arg / 100 * P[t]))
                                if n_sel > 0:
                                    top = dev_tok[b, : P[t]].topk(min(n_sel, P[t])).indices
                                    active[b, top] = True
                            x = y                                           # layer 0 was computed for all
                        else:
                            x = torch.where(active[..., None], y, stale)
                    else:
                        x = y
                    states.append(x)
            for b, t in enumerate(live):
                prev[t] = [st[b, : Ls[t]] for st in states]
                sink(t, step, states[k][b, : Ls[t]], (P[t], Ls[t] - 1))
                L, A = Ls[t], int(active[b].sum()) if (step > 0 and kind != "full") else Ls[t]
                fl_full += k * (L * cl + L * L * ca)
                if kind == "select" and step > 0:
                    fl_inc += (L * cl + L * L * ca) + (k - 1) * (A * cl + A * L * ca)
                else:
                    fl_inc += k * (A * cl + A * L * ca)
    return {"flops_full": fl_full, "flops_mode": fl_inc, "ratio": fl_inc / max(fl_full, 1.0)}


# ---------------------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------------------

def build(trunk, k, threads_ex, texts, mode, bs, log) -> Tuple[StepCache, Dict[Tuple[int, int], int], Dict]:
    """Encode all threads of one split in `mode`; returns the cache and (thread, step) -> cache index."""
    ids = [thread_ids(trunk.tok, [texts[i] for i in th]) for th in threads_ex]
    cache, where = StepCache(trunk), {}

    def sink(t, step, h, span):
        if span[1] <= span[0]:                                              # empty message: pool its SEP
            span = (span[0], span[0] + 1)
        where[(t, step)] = cache.add(h, span)

    t0 = time.time()
    fl = run_chains(trunk, k, ids, mode, bs, sink, log)
    fl["seconds"] = round(time.time() - t0, 1)
    return cache, where, fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="clinc150")
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--thread-len", type=int, default=10)
    ap.add_argument("--modes", nargs="*", default=["stale", "band16", "band32", "band64", "select10", "select25"])
    ap.add_argument("--kinds", nargs="*", default=["probe", "blocks:2"])
    ap.add_argument("--max-train", type=int, default=10000, help="training messages (thread steps) to use")
    ap.add_argument("--max-test", type=int, default=3000, help="test messages (thread steps) to use")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--probe-epochs", type=int, default=10)
    ap.add_argument("--min-steps", type=int, default=300, help="minimum optimiser steps per branch (as train_branch)")
    ap.add_argument("--no-train-on-stale", action="store_true")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.split, a.thread_len, a.max_train, a.epochs, a.probe_epochs, a.bs = "cpu", 4, 4, 80, 1, 2, 16
        a.min_steps = 6
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
    k = a.split
    rng = random.Random(a.seed)
    splits = {"train": list(ds.train), "val": list(ds.val), "test": list(ds.test)}
    if a.smoke:
        splits = {"train": splits["train"][::200][:80], "val": splits["val"][::100][:30],
                  "test": splits["test"][::150][:36]}
    threads = {s: make_threads(ex, a.thread_len, a.seed + i) for i, (s, ex) in enumerate(splits.items())}
    for s, cap in (("train", a.max_train), ("test", a.max_test)):             # keep whole threads
        if cap and sum(map(len, threads[s])) > cap:
            rng.shuffle(threads[s])
            keep, n = [], 0
            for th in threads[s]:
                if n >= cap:
                    break
                keep.append(th)
                n += len(th)
            threads[s] = keep
    texts = {s: [e.text for e in ex] for s, ex in splits.items()}
    tasks = list(ds.tasks)
    lens = [len(th) for th in threads["test"]]
    log(f"== {ds.name}: threads train/val/test {len(threads['train'])}/{len(threads['val'])}/{len(threads['test'])} "
        f"(test thread length mean {np.mean(lens):.1f}, max {max(lens)}) | split {k} | {trunk.device}")
    res = {"dataset": ds.name, "split": k, "thread_len": a.thread_len, "device": str(trunk.device), "modes": {}}

    # sanity: the masked incremental path with every token active equals a plain encode
    th0 = threads["test"][0][:3]
    ids0 = thread_ids(trunk.tok, [texts["test"][i] for i in th0])
    got = {}
    run_chains(trunk, k, [ids0], "full", 1, lambda t, s, h, sp: got.__setitem__(s, h.float().cpu()), log)
    full_ids = [trunk.tok.cls_token_id] + sum((m + [trunk.tok.sep_token_id] for m in ids0), [])
    with torch.no_grad(), autocast(trunk.device):
        taps, _ = trunk.taps(torch.tensor([full_ids], device=trunk.device),
                             torch.ones(1, len(full_ids), dtype=torch.long, device=trunk.device), [k])
    err = float((got[len(ids0) - 1] - taps[k][0].float().cpu()).abs().max() / taps[k].abs().max().float().cpu())
    res["masked_path_rel_err"] = err
    log(f"  sanity: masked incremental path (all active) vs plain encode: max rel err {err:.2e}")
    assert err < 2e-2, "masked incremental path does not reproduce a plain encode"

    # full encodings (reference) for every split
    caches = {}
    for s in ("train", "val", "test"):
        caches[("full", s)] = build(trunk, k, threads[s], texts[s], "full", a.bs, log)
        log(f"  full {s}: {len(caches[('full', s)][0].h)} steps in {caches[('full', s)][2]['seconds']}s")

    def rows(s, mode="full"):
        """Cache indices and per-task labels for every (thread, step) of split s."""
        cache, where, _ = caches[(mode, s)]
        idx, ys = [], {t: [] for t in tasks}
        for ti, th in enumerate(threads[s]):
            for st, mi in enumerate(th):
                idx.append(where[(ti, st)])
                for t in tasks:
                    ys[t].append(splits[s][mi].y[t])
        steps = [st for th in threads[s] for st in range(len(th))]
        return idx, {t: torch.tensor(v) for t, v in ys.items()}, np.array(steps)

    # incremental encodings of the test threads (and of train/val in stale mode, for train-on-stale)
    for m in a.modes:
        caches[(m, "test")] = build(trunk, k, threads["test"], texts["test"], m, a.bs, log)
        log(f"  mode {m}: test encoded in {caches[(m, 'test')][2]['seconds']}s, "
            f"trunk FLOPs vs full re-encode {caches[(m, 'test')][2]['ratio']:.3f}")
    if not a.no_train_on_stale and "stale" in a.modes:
        for s in ("train", "val"):
            caches[("stale", s)] = build(trunk, k, threads[s], texts[s], "stale", a.bs, log)

    # representation drift of the new message (pooled over its tokens) vs full
    te_full, y_te, steps_te = rows("test")
    pooled_full = caches[("full", "test")][0].pooled(te_full)
    for m in a.modes:
        idx_m, _, _ = rows("test", m)
        cos = F.cosine_similarity(caches[(m, "test")][0].pooled(idx_m), pooled_full, -1).numpy()
        res["modes"][m] = {"flops": caches[(m, "test")][2], "cos_mean": float(cos.mean()),
                           "cos_p05": float(np.percentile(cos, 5)), "branches": {}}
        log(f"  mode {m}: cosine to full (new message, pooled at depth {k}) mean {cos.mean():.4f} "
            f"p05 {np.percentile(cos, 5):.4f}")

    def fit_eval(kind, train_mode):
        """Train one branch per task on `train_mode` states; evaluate on full and every mode."""
        tr_idx, y_tr, _ = rows("train", train_mode)
        va_idx, y_va, _ = rows("val", train_mode)
        out = {}
        for task in tasks:
            labels = ds.tasks[task].labels
            depth = int(kind.split(":")[1]) if kind.startswith("blocks") else 0
            br = ProbeBranch(k, labels, trunk.hidden) if kind == "probe" else \
                BlockBranch(k, labels, trunk.hidden, depth, trunk)
            ep, lr_h = (a.probe_epochs, 3e-3) if kind == "probe" else (a.epochs, 1e-3)
            t0 = time.time()
            train_branch(br, caches[(train_mode, "train")][0], tr_idx, y_tr[task], None,
                         va_idx, y_va[task], epochs=ep, lr_head=lr_h, seed=a.seed, min_steps=a.min_steps)
            vz = predict_logits(br, caches[(train_mode, "val")][0], va_idx)
            temp = fit_temperature(vz, y_va[task]) if len(va_idx) >= 30 else 1.0
            full_pred = None
            per = {}
            for m in ["full"] + list(a.modes):
                idx_m, _, _ = rows("test", m)
                z = predict_logits(br, caches[(m, "test")][0], idx_m)
                p = torch.softmax(z / temp, -1).numpy()
                met = evaluate(p, y_te[task].numpy())
                pred = p.argmax(-1)
                if m == "full":
                    full_pred = pred
                met["agree_with_full"] = float((pred == full_pred).mean())
                late = steps_te >= 4
                met["acc_steps_ge5"] = float((pred[late] == y_te[task].numpy()[late]).mean()) if late.any() else None
                per[m] = met
            out[task] = per
            log(f"   [{kind} trained on {train_mode}] {task}: " + " ".join(
                f"{m}={per[m]['acc']:.3f}/{per[m]['agree_with_full']:.3f}" for m in per) +
                f"  (acc/agree, {time.time() - t0:.0f}s)")
        return out

    for kind in a.kinds:
        for train_mode in ["full"] + (["stale"] if (not a.no_train_on_stale and "stale" in a.modes) else []):
            r = fit_eval(kind, train_mode)
            key = f"{kind}|train={train_mode}"
            for m in ["full"] + list(a.modes):
                entry = {"acc": float(np.mean([r[t][m]["acc"] for t in tasks])),
                         "agree_with_full": float(np.mean([r[t][m]["agree_with_full"] for t in tasks])),
                         "per_task_acc": {t: r[t][m]["acc"] for t in tasks}}
                late = [r[t][m]["acc_steps_ge5"] for t in tasks if r[t][m]["acc_steps_ge5"] is not None]
                entry["acc_steps_ge5"] = float(np.mean(late)) if late else None
                res["modes"].setdefault(m, {"branches": {}})["branches"][key] = entry
            log(f"-- {key}: " + " | ".join(
                f"{m} {res['modes'][m]['branches'][key]['acc']:.4f}" for m in ["full"] + list(a.modes)))
            if a.out:
                json.dump(res, open(a.out, "w"), indent=1)

    log("== summary: mean acc over tasks and all thread steps (agreement with full); trunk FLOPs vs re-encode")
    for m in ["full"] + list(a.modes):
        info = res["modes"][m]
        br = " ".join(f"{key}: {v['acc']:.4f} ({v['agree_with_full']:.3f})" for key, v in info["branches"].items())
        ratio = info.get("flops", {}).get("ratio", 1.0)
        log(f"  {m:9s} FLOPs x{ratio:.3f}  cos {info.get('cos_mean', 1.0):.4f}  {br}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
