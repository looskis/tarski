"""Do the learned query and the selected order stack? Typed-test states read under the given order
(rot0) and the selected order (rot3) with the vocabulary-mean slot and with the gold-target query
(results/dlm/queries_gold_type.safetensors), MLX backend.

  .venv/bin/python dlm/order_query.py --limit 400 --out results/dlm/order_query_typed_test.jsonl
  .venv/bin/python dlm/order_query.py --report results/dlm/order_query_typed_test.jsonl
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

ORDERS = {"rot0": 0, "rot3": 3}


def run(a):
    from dlm.reads import Reader
    from dlm.train_query import Queries
    recs = data.typed_decisions("test")[: a.limit]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} states, {len(todo)} to run", flush=True)
    r = Reader()
    queries, _ = Queries.load(a.queries, r)
    t0 = time.time()
    with open(a.out, "a") as f:
        for n, rec in enumerate(todo, 1):
            qids = list(rec["questions"])
            out = {q: {} for q in qids}
            for oname, k in ORDERS.items():
                order = qids[k:] + qids[:k]
                prep = r.prepare(rec["state"], {q: rec["questions"][q] for q in order})
                keys = [q["key"] for q in prep["qs"]]
                rows = queries.rows(queries.params, [q["type"] for q in prep["qs"]])
                for name, kw in ((f"{oname}_mean", {"mode": "mean"}), (f"{oname}_query", {"mode": "query", "queries": rows}),
                                 (f"{oname}_random", {"mode": "random", "seed": 0})):
                    for key, p in zip(keys, r.read(prep, **kw)):
                        out[key][name] = p
            base = r.prepare(rec["state"], rec["questions"])
            row = {"id": rec["id"], "family": rec["family"], "questions": []}
            for q in base["qs"]:
                g = rec["gold"].get(q["key"], {})
                row["questions"].append({"qid": q["key"], "type": q["type"], "labels": [c[0] for c in q["choices"]],
                                         "gold": g.get("label"), "reads": out[q["key"]]})
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({(time.time() - t0) / n:.1f} s/state)", flush=True)


def report(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    acc = defaultdict(list)
    for row in rows:
        for q in row["questions"]:
            if q["gold"] not in q["labels"]:
                continue
            y = q["labels"].index(q["gold"])
            for c, p in q["reads"].items():
                top = max(range(len(p)), key=p.__getitem__)
                acc[c].append((top == y, p[top], brier(p, y)))
    print(f"{len(rows)} states")
    print(f"{'condition':14s} {'acc':>6s} {'brier':>6s} {'ece':>6s}")
    for c in sorted(acc):
        v = acc[c]; n = len(v)
        print(f"{c:14s} {sum(x[0] for x in v) / n:6.3f} {sum(x[2] for x in v) / n:6.3f} {ece([(x[1], x[0]) for x in v]):6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--queries", default="results/dlm/queries_gold_type.safetensors")
    ap.add_argument("--out", default="results/dlm/order_query_typed_test.jsonl")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        run(a)


if __name__ == "__main__":
    main()
