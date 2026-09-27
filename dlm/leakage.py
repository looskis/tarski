"""Attention leakage across questions, per question order (torch backend, eager attention).

For every slot on a five-question canvas, under each cyclic rotation of the questions (and the
reversed order): the share of the slot row's attention mass, per layer, on (a) its own question's
template rows, (b) other questions' template rows, (c) other slots, (d) the encoder prefix (state +
system prompt), (e) itself. Averaged over heads. Saved with the slot's read under that order, so the
report can test whether leakage to other questions predicts the bad orders.

  .venv/bin/python dlm/leakage.py --limit 200 --out results/dlm/leakage_typed_test.jsonl
  .venv/bin/python dlm/leakage.py --report results/dlm/leakage_typed_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm import data  # noqa: E402
from dlm.metrics import brier, ece  # noqa: E402

GROUPS = ("own", "other_q", "other_slot", "prefix", "self")


def orders(qids):
    out = {f"rot{k}": qids[k:] + qids[:k] for k in range(len(qids))}
    out["rev"] = list(reversed(qids))
    return out


def question_rows(prep):
    """Map each canvas row to the question whose line it belongs to (rows between one slot and the
    next belong to the next question; rows before the first slot belong to the first)."""
    width = prep["width"]
    positions = [s["pos"] for s in prep["slots"]]
    owner = [None] * width
    start = 0
    for qi, pos in enumerate(positions):
        for j in range(start, pos + 1):
            owner[j] = qi
        start = pos + 1
    return owner, positions


def run(a):
    import torch
    from dlm.reads_torch import AttentionRecorder, Reader
    recs = data.typed_decisions(a.split)[: a.limit]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} states, {len(todo)} to run", flush=True)
    r = Reader(attn_implementation="eager")
    n_layers = len(r.dec.layers)
    t0 = time.time()
    with open(a.out, "a") as f:
        for n, rec in enumerate(todo, 1):
            qids = list(rec["questions"])
            out = {qid: {} for qid in qids}
            for name, order in orders(qids).items():
                prep = r.prepare(rec["state"], {q: rec["questions"][q] for q in order})
                cache = r.prefill(prep["prompt"])
                owner, positions = question_rows(prep)
                width = prep["width"]
                keys = [q["key"] for q in prep["qs"]]
                for mode in ("random", "mean"):
                    with AttentionRecorder(r) as rec_attn:
                        probs = r.read(prep, mode, seed=0)
                    # per slot, per layer: mass shares by group (heads averaged)
                    shares = {k: [[0.0] * n_layers for _ in positions] for k in GROUPS}
                    for layer, w in rec_attn.weights.items():
                        w = w[0].mean(dim=0)                       # [Q=width, K]
                        key_len = w.shape[-1]
                        enc_len = key_len - width
                        for si, pos in enumerate(positions):
                            row = w[pos]
                            prefix = float(row[:enc_len].sum())
                            canvas = row[enc_len:]
                            own = other_q = other_slot = 0.0
                            for j in range(width):
                                v = float(canvas[j])
                                if j == pos:
                                    continue
                                if j in positions:
                                    other_slot += v
                                elif owner[j] == si:
                                    own += v
                                else:
                                    other_q += v
                            shares["own"][si][layer] = own
                            shares["other_q"][si][layer] = other_q
                            shares["other_slot"][si][layer] = other_slot
                            shares["prefix"][si][layer] = prefix
                            shares["self"][si][layer] = float(canvas[pos])
                    for si, (k, p) in enumerate(zip(keys, probs)):
                        out[k][f"{name}_{mode}"] = {"probs": p, "pos": si,
                                                    "shares": {g: shares[g][si] for g in GROUPS}}
            row = {"id": rec["id"], "family": rec["family"], "questions": []}
            base = r.prepare(rec["state"], rec["questions"])
            for q in base["qs"]:
                g = rec["gold"].get(q["key"], {})
                row["questions"].append({"qid": q["key"], "type": q["type"], "labels": [c[0] for c in q["choices"]],
                                         "gold": g.get("label"), "conds": out[q["key"]]})
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({(time.time() - t0) / n:.1f} s/state)", flush=True)


def report(path):
    import numpy as np
    rows = [json.loads(l) for l in open(path) if l.strip()]
    conds = sorted({c for row in rows for q in row["questions"] for c in q["conds"]})
    print(f"{len(rows)} states")
    for mode in ("random", "mean"):
        print(f"\n== slot {mode}: accuracy and mean attention shares (all layers) per order")
        print(f"{'order':8s} {'acc':>6s} {'brier':>6s} {'ece':>6s} | " + " ".join(f"{g:>10s}" for g in GROUPS))
        per_slot = []      # (correct, other_q share, order)
        for c in [c for c in conds if c.endswith(f"_{mode}")]:
            items, shares = [], defaultdict(list)
            for row in rows:
                for q in row["questions"]:
                    if q["gold"] not in q["labels"] or c not in q["conds"]:
                        continue
                    y = q["labels"].index(q["gold"]); d = q["conds"][c]; p = d["probs"]
                    top = max(range(len(p)), key=p.__getitem__)
                    items.append((top == y, p[top], brier(p, y)))
                    for g in GROUPS:
                        shares[g].append(float(np.mean(d["shares"][g])))
                    per_slot.append((top == y, float(np.mean(d["shares"]["other_q"])), float(np.mean(d["shares"]["own"])), c))
            n = len(items)
            print(f"{c[:-len(mode) - 1]:8s} {sum(i[0] for i in items) / n:6.3f} {sum(i[2] for i in items) / n:6.3f} "
                  f"{ece([(i[1], i[0]) for i in items]):6.3f} | " + " ".join(f"{np.mean(shares[g]):10.3f}" for g in GROUPS))
        # does leakage predict correctness within a question across orders?
        ok = np.array([x[0] for x in per_slot], dtype=float); oq = np.array([x[1] for x in per_slot]); own = np.array([x[2] for x in per_slot])
        print(f"  corr(correct, other_q share) = {np.corrcoef(ok, oq)[0, 1]:.3f};  corr(correct, own share) = {np.corrcoef(ok, own)[0, 1]:.3f}")
        # label-free selector: per question, the order with the least other_q leakage / most own share
        for crit, key in (("least other_q", lambda d: np.mean(d["shares"]["other_q"])), ("most own", lambda d: -np.mean(d["shares"]["own"]))):
            items = []
            for row in rows:
                for q in row["questions"]:
                    if q["gold"] not in q["labels"]:
                        continue
                    cs = [c for c in q["conds"] if c.endswith(f"_{mode}")]
                    best = min(cs, key=lambda c: key(q["conds"][c]))
                    p = q["conds"][best]["probs"]; y = q["labels"].index(q["gold"]); top = max(range(len(p)), key=p.__getitem__)
                    items.append((top == y, p[top], brier(p, y)))
            n = len(items)
            print(f"  selector '{crit}': acc {sum(i[0] for i in items) / n:.3f} brier {sum(i[2] for i in items) / n:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--out", default="results/dlm/leakage_typed_test.jsonl")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        run(a)


if __name__ == "__main__":
    main()
