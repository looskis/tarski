"""Idea 4: schema-constant templates for JSON states ("columnar encoding").

typed-decisions states are JSON with one fixed key set per workflow; ~44% of tokens are keys and
punctuation. Treat the schema as a table header: for every structure token signature
(workflow, key, token id, occurrence since the key), average its trunk state at every layer over the
training messages ("template"). At encode time, structure tokens take their template state at every
layer and only value tokens (and CLS/SEP) are computed; value tokens read the templated structure
tokens as keys/values (RoPE still uses the true positions). In a real engine the structure tokens'
pre-RoPE K/V would be a table lookup, so trunk FLOPs scale with the value-token count.

The masked implementation computes every token and then overwrites the templated ones, which is exactly
what computing only value-token queries against the mixed K/V gives.

Tested prediction: branches trained on normal encodings lose < 2 points on templated encodings, and
branches trained on templated encodings lose less; trunk FLOPs drop to the value-token fraction.

Usage:
  python explore/systems_schema.py --smoke
  python explore/systems_schema.py --out results/tarski/explore_schema.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explore.systems_common import (Log, all_texts, cache_from_states, json_structure, save, subset, threads,
                                    train_tasks)

import numpy as np
import torch
import torch.nn.functional as F

from tarski import data
from tarski.branches import mean_pool
from tarski.train import FeatureCache, autocast, evaluate
from tarski.trunk import Trunk


def encode(trunk: Trunk, ids_list: List[List[int]], sig_list: List[List[int]], k: int, bs: int,
           templates=None, sums=None, counts=None):
    """Run layers [0, k) over all messages. With `sums`, accumulate per-signature states at every depth
    1..k. With `templates` ({depth: (S, D)}, S signatures), overwrite every templated token's state after
    each layer. Returns the depth-k states per message (fp16, CPU)."""
    dev = trunk.device
    out = [None] * len(ids_list)
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(ids_list[i]) for i in idx)
        x = torch.full((len(idx), L), trunk.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        sig = torch.full((len(idx), L), -1, dtype=torch.long)
        for j, i in enumerate(idx):
            n = len(ids_list[i])
            x[j, :n] = torch.tensor(ids_list[i])
            att[j, :n] = 1
            sig[j, :n] = torch.tensor(sig_list[i])
        x, att, sig = x.to(dev), att.to(dev), sig.to(dev)
        use = sig >= 0
        with torch.no_grad(), autocast(dev):
            h = trunk.model.embeddings(input_ids=x)
            ctx = trunk.context(h, att)
            for j in range(k):
                layer = trunk.model.layers[j]
                h = layer(h, attention_mask=ctx.masks[layer.attention_type],
                          position_embeddings=ctx.rope[layer.attention_type])
                if sums is not None:
                    sums[j + 1].index_add_(0, sig[use], h[use].float())
                    if j == 0:
                        counts.index_add_(0, sig[use], torch.ones_like(sig[use], dtype=torch.float))
                if templates is not None:
                    t = templates[j + 1].to(h.dtype)
                    h = h.clone()
                    h[use] = t[sig[use]]
        for j, i in enumerate(idx):
            out[i] = h[j, : len(ids_list[i])].to("cpu", torch.float16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--min-count", type=int, default=5, help="signatures seen fewer times stay computed")
    ap.add_argument("--block-workflows", nargs="*", default=["customer_service", "security_incidents"],
                    help="workflows whose tasks also get blocks:2 branches (probes run on all tasks)")
    ap.add_argument("--no-blocks", action="store_true")
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.split, a.min_steps, a.bs, a.min_count = "cpu", 4, 5, 8, 2
    if not a.smoke and not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    trunk = Trunk(device=a.device)
    ds = data.load("typed-decisions")
    if a.smoke:
        wf = [t for t in ds.tasks if t.startswith("customer_service")][:2]
        ds = subset(ds, wf, 40, 30, 24, max_len=256)
        a.block_workflows = ["customer_service"]
    k, texts = a.split, all_texts(ds)
    allx = ds.train + ds.val + ds.test
    n_tr = len(ds.train)
    wf_of = [next(iter(e.y)).split(".")[0] for e in allx]

    # structure signatures
    t0 = time.time()
    sig_ids: Dict = {}
    ids_list, raw_sigs, n_struct, n_tok = [], [], 0, 0
    for text, wf in zip(texts, wf_of):
        ids, is_s, sigs = json_structure(trunk.tok, text, ds.max_len, wf)
        ids_list.append(ids)
        raw_sigs.append(sigs)
        n_struct += sum(is_s)
        n_tok += len(ids)
    for i in range(n_tr):                                                  # signatures come from train only
        for sg in raw_sigs[i]:
            if sg is not None and sg not in sig_ids:
                sig_ids[sg] = len(sig_ids)
    sig_list = [[sig_ids.get(sg, -1) if sg is not None else -1 for sg in sigs] for sigs in raw_sigs]
    log(f"== {ds.summary()} | split {k} | structure tokens {n_struct / n_tok:.3f} of {n_tok} | "
        f"{len(sig_ids)} train signatures ({time.time() - t0:.1f}s)")

    # templates: mean state per signature per depth, over training messages
    S, D = len(sig_ids), trunk.hidden
    sums = {j: torch.zeros(S, D, device=trunk.device) for j in range(1, k + 1)}
    counts = torch.zeros(S, device=trunk.device)
    t0 = time.time()
    encode(trunk, ids_list[:n_tr], sig_list[:n_tr], k, a.bs, sums=sums, counts=counts)
    ok = counts >= a.min_count
    templates = {j: sums[j] / counts.clamp_min(1)[:, None] for j in sums}
    remap = torch.where(ok, torch.arange(S, device=counts.device), torch.full_like(counts, -1, dtype=torch.long))
    sig_used = [[int(remap[s]) if s >= 0 else -1 for s in sl] for sl in sig_list]
    frac_tpl = sum(sum(1 for s in sl if s >= 0) for sl in sig_used) / n_tok
    log(f"  templates from {n_tr} train messages in {time.time() - t0:.1f}s; {int(ok.sum())}/{S} signatures used; "
        f"templated token fraction {frac_tpl:.3f} (all splits)")

    t0 = time.time()
    full = encode(trunk, ids_list, sig_used, k, a.bs)
    tpl = encode(trunk, ids_list, sig_used, k, a.bs, templates=templates)
    log(f"  encoded all messages normally and templated in {time.time() - t0:.1f}s")
    fc_full = cache_from_states(trunk, {k: full})
    fc_tpl = cache_from_states(trunk, {k: tpl})

    # drift of value tokens and of the pooled message state
    te = range(n_tr + len(ds.val), len(texts))
    cos_val, cos_pool = [], []
    for i in te:
        v = torch.tensor([s < 0 for s in sig_used[i]])
        a_, b_ = full[i].float(), tpl[i].float()
        cos_val.append(float(F.cosine_similarity(a_[v], b_[v], -1).mean()))
        cos_pool.append(float(F.cosine_similarity(a_.mean(0), b_.mean(0), 0)))
    res = {"split": k, "structure_frac": n_struct / n_tok, "templated_frac": frac_tpl,
           "trunk_flops_ratio": 1 - frac_tpl, "cos_value_tokens": float(np.mean(cos_val)),
           "cos_pooled": float(np.mean(cos_pool)), "branches": {}}
    log(f"  test value-token cosine (templated vs normal) {np.mean(cos_val):.4f}; pooled {np.mean(cos_pool):.4f}; "
        f"trunk FLOPs ratio ~{1 - frac_tpl:.3f}")

    def run(kind, tasks):
        for train_on, fc_train in (("normal", fc_full), ("templated", fc_tpl)):
            r = train_tasks(trunk, ds, fc_train, tasks, kind, k, seed=a.seed, min_steps=a.min_steps, log=lambda s: None)
            evals = {}
            for eval_on, fc_eval in (("normal", fc_full), ("templated", fc_tpl)):
                accs = []
                for t, v in r.items():
                    from tarski.train import predict_logits
                    z = predict_logits(v["branch"], fc_eval, v["sel"]["test"])
                    p = torch.softmax(z / v["T"], -1).numpy()
                    accs.append(evaluate(p, v["y"]["test"].numpy())["acc"])
                evals[eval_on] = {"mean_acc": float(np.mean(accs)), "per_task": dict(zip(r, accs))}
            key = f"{kind}|train={train_on}"
            res["branches"][key] = evals
            log(f"-- {key} ({len(tasks)} tasks): eval normal {evals['normal']['mean_acc']:.4f} | "
                f"eval templated {evals['templated']['mean_acc']:.4f}")
            save(res, a.out)

    run("probe", list(ds.tasks))
    if not a.no_blocks:
        run("blocks:2", [t for t in ds.tasks if t.split(".")[0] in a.block_workflows])
    save(res, a.out)
    log("done")


if __name__ == "__main__":
    main()
