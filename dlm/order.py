"""Question order on multi-question canvases: where does the order effect live, and can it be a
policy? Typed-decisions test states have five questions; each rotation puts every question at every
canvas position, and the hybrids split the order between the system prompt and the canvas.

Conditions (single read, seed 0, plus the mean-embedding read `*_mean`):
  rot0..rot4          cyclic rotations of the given order (rot0 = OpenJev's order = "joint")
  rev                 reversed order (the interference run's `reversed`, repeated for a same-file baseline)
  prompt_rev          system prompt lists the questions reversed, canvas keeps the given order
  canvas_rev          canvas reversed, system prompt keeps the given order

Per question: probabilities under each condition and the question's canvas position (0-4) in it.
`--report` prints accuracy / Brier / ECE per condition by type, accuracy by (type, canvas position),
and the held-out "best position per type" policy (position chosen on one half, scored on the other).

  .venv/bin/python dlm/order.py --limit 200 --out results/dlm/order_typed_test.jsonl
  .venv/bin/python dlm/order.py --report results/dlm/order_typed_test.jsonl
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
from dlm.reads import Reader  # noqa: E402


def orders(qids):
    n = len(qids)
    out = {f"rot{k}": qids[k:] + qids[:k] for k in range(n)}
    out["rev"] = list(reversed(qids))
    return out


def run(a):
    recs = data.typed_decisions("test")[: a.limit]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} states, {len(todo)} to run", flush=True)
    r = Reader()
    t0 = time.time()
    with open(a.out, "a") as f:
        for n, rec in enumerate(todo, 1):
            qids = list(rec["questions"])
            out = {qid: {} for qid in qids}
            pos = {qid: {} for qid in qids}
            preps = {}
            for name, order in orders(qids).items():
                preps[name] = r.prepare(rec["state"], {qid: rec["questions"][qid] for qid in order})
            # hybrids: prompt from one order, canvas (template/slots/qs) from another
            preps["prompt_rev"] = dict(preps["rot0"], prompt=preps["rev"]["prompt"])
            preps["canvas_rev"] = dict(preps["rev"], prompt=preps["rot0"]["prompt"])
            for name, prep in preps.items():
                keys = [q["key"] for q in prep["qs"]]
                for mode, suffix in (("random", ""), ("mean", "_mean")):
                    for i, (k, p) in enumerate(zip(keys, r.read(prep, mode, seed=0))):
                        out[k][name + suffix] = p
                        pos[k][name + suffix] = i
            row = {"id": rec["id"], "family": rec["family"], "questions": []}
            for q in preps["rot0"]["qs"]:
                g = rec["gold"].get(q["key"], {})
                row["questions"].append({"qid": q["key"], "type": q["type"], "labels": [c[0] for c in q["choices"]],
                                         "gold": g.get("label"), "reads": out[q["key"]], "pos": pos[q["key"]]})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({(time.time() - t0) / n:.1f} s/state)", flush=True)


def stats(items):
    n = len(items)
    return (sum(x["correct"] for x in items) / n, sum(x["brier"] for x in items) / n,
            ece([(x["conf"], x["correct"]) for x in items]))


def report(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    per = defaultdict(lambda: defaultdict(list))        # type -> condition -> items
    bypos = defaultdict(lambda: defaultdict(list))      # (type, suffix) -> position -> items (rotations only)
    for si, row in enumerate(rows):
        for q in row["questions"]:
            if q["gold"] not in q["labels"]:
                continue
            y = q["labels"].index(q["gold"])
            for c, p in q["reads"].items():
                top = max(range(len(p)), key=p.__getitem__)
                item = {"correct": top == y, "conf": p[top], "brier": brier(p, y), "state": si, "half": si % 2}
                per[q["type"]][c].append(item)
                base = c.replace("_mean", "")
                if base.startswith("rot"):
                    bypos[(q["type"], "_mean" if c.endswith("_mean") else "")][q["pos"][c]].append(item)
    conds = ["rot0", "rot1", "rot2", "rot3", "rot4", "rev", "prompt_rev", "canvas_rev"]
    for t in sorted(per):
        print(f"\n== {t} (n={len(per[t]['rot0'])} slots)")
        print(f"{'condition':12s} {'acc':>6s} {'brier':>6s} {'ece':>6s} | {'mean-slot acc':>13s} {'brier':>6s} {'ece':>6s}")
        for c in conds:
            a1 = stats(per[t][c])
            a2 = stats(per[t][c + "_mean"])
            print(f"{c:12s} {a1[0]:6.3f} {a1[1]:6.3f} {a1[2]:6.3f} | {a2[0]:13.3f} {a2[1]:6.3f} {a2[2]:6.3f}")
        for suffix in ("", "_mean"):
            d = bypos[(t, suffix)]
            print(f"  acc by canvas position{' (mean slot)' if suffix else ''}: " +
                  "  ".join(f"pos{p}={stats(d[p])[0]:.3f}" for p in sorted(d)))
        # held-out policy: choose the best position per type on one half of the states, score the other half
        for suffix in ("", "_mean"):
            d = bypos[(t, suffix)]
            tot, n = 0, 0
            for h in (0, 1):
                fit = {p: sum(x["correct"] for x in v if x["half"] != h) / max(sum(x["half"] != h for x in v), 1)
                       for p, v in d.items()}
                best = max(fit, key=fit.get)
                held = [x for x in d[best] if x["half"] == h]
                tot += sum(x["correct"] for x in held)
                n += len(held)
            print(f"  best-position policy (2-fold){' (mean slot)' if suffix else ''}: acc {tot / max(n, 1):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="results/dlm/order_typed_test.jsonl")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        run(a)


if __name__ == "__main__":
    main()
