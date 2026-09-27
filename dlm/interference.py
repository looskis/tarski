"""Cross-slot interference and slot isolation on multi-question canvases (typed-decisions: five
questions per state).

Conditions, all single reads at seed 0 unless stated:
  joint        the five questions in one canvas (OpenJev's default)
  alone        each question in its own canvas (five prefills of five different prompts)
  isolate      joint, with slot rows masked from other slots
  invisible    joint, with no canvas row seeing another slot
  reversed     joint, questions in reverse order (position effect)
  joint_seed1  joint at seed 1 (the seed effect, for scale)
  meanemb      joint, vocabulary-mean slot (deterministic)
  isolate_mean invisible mask + vocabulary-mean slot: a read that depends on nothing random and on no
               other slot

Per question: probabilities under each condition. `--report` prints TV and flip rates against
`joint`, and accuracy / Brier / ECE per condition, from the saved file.

  .venv/bin/python dlm/interference.py --limit 200 --out results/dlm/interference_typed_test.jsonl
  .venv/bin/python dlm/interference.py --report results/dlm/interference_typed_test.jsonl
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
from dlm.metrics import brier, ece, tv  # noqa: E402
from dlm.reads import Reader  # noqa: E402

CONDITIONS = ("joint", "alone", "isolate", "invisible", "reversed", "joint_seed1", "meanemb", "isolate_mean")


def run(a):
    recs = data.typed_decisions("test")[: a.limit]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} states, {len(todo)} to run", flush=True)
    r = Reader()
    mx = r.mx
    t0 = time.time()
    with open(a.out, "a") as f:
        for n, rec in enumerate(todo, 1):
            qids = list(rec["questions"])
            out = {qid: {} for qid in qids}
            # joint canvases
            prep = r.prepare(rec["state"], rec["questions"])
            cache = r.prefill(prep["prompt"])
            positions = [s["pos"] for s in prep["slots"]]
            keys = [q["key"] for q in prep["qs"]]
            for name, kw in (("joint", {"mode": "random", "seed": 0}), ("joint_seed1", {"mode": "random", "seed": 1}),
                             ("meanemb", {"mode": "mean"})):
                for k, p in zip(keys, r.read(prep, **kw)):
                    out[k][name] = p
            for name, mmode, rmode in (("isolate", "isolate", "random"), ("invisible", "invisible", "random"),
                                       ("isolate_mean", "invisible", "mean")):
                masks = r.slot_masks(cache, prep["width"], positions, mmode)
                for k, p in zip(keys, r.read(prep, rmode, seed=0, masks=masks)):
                    out[k][name] = p
            # reversed order
            rev = {qid: rec["questions"][qid] for qid in reversed(qids)}
            prep_r = r.prepare(rec["state"], rev)
            for k, p in zip([q["key"] for q in prep_r["qs"]], r.read(prep_r, "random", seed=0)):
                out[k]["reversed"] = p
            # alone
            for qid in qids:
                prep_1 = r.prepare(rec["state"], {qid: rec["questions"][qid]})
                out[qid]["alone"] = r.read(prep_1, "random", seed=0)[0]
            row = {"id": rec["id"], "family": rec["family"], "questions": []}
            for q in prep["qs"]:
                g = rec["gold"].get(q["key"], {})
                row["questions"].append({"qid": q["key"], "type": q["type"], "labels": [c[0] for c in q["choices"]],
                                         "gold": g.get("label"), "reads": out[q["key"]]})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({(time.time() - t0) / n:.1f} s/state)", flush=True)


def report(path, by=None):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    acc = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for q in row["questions"]:
            key = {"type": q["type"], "family": row["family"]}.get(by, "all") if by else "all"
            ref = q["reads"]["joint"]
            top_ref = max(range(len(ref)), key=ref.__getitem__)
            y = q["labels"].index(q["gold"]) if q["gold"] in q["labels"] else None
            for c, p in q["reads"].items():
                top = max(range(len(p)), key=p.__getitem__)
                acc[key][c].append({"tv": tv(p, ref), "flip": top != top_ref,
                                    "correct": (top == y) if y is not None else None,
                                    "conf": p[top], "brier": brier(p, y) if y is not None else None})
    for key in sorted(acc):
        d = acc[key]
        n = len(d["joint"])
        print(f"\n== {key} (n={n} slots)")
        print(f"{'condition':13s} {'tv->joint':>9s} {'flip':>5s} {'acc':>6s} {'brier':>6s} {'ece':>6s}")
        for c in CONDITIONS:
            if c not in d:
                continue
            v = [x for x in d[c] if x["correct"] is not None]
            print(f"{c:13s} {sum(x['tv'] for x in d[c]) / n:9.3f} {sum(x['flip'] for x in d[c]) / n:5.2f} "
                  f"{sum(x['correct'] for x in v) / max(len(v), 1):6.3f} {sum(x['brier'] for x in v) / max(len(v), 1):6.3f} "
                  f"{ece([(x['conf'], x['correct']) for x in v]) if v else float('nan'):6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="results/dlm/interference_typed_test.jsonl")
    ap.add_argument("--report", default=None, help="score this file instead of running")
    ap.add_argument("--by", choices=["type", "family"], default=None)
    a = ap.parse_args()
    if a.report:
        report(a.report, a.by)
    else:
        run(a)


if __name__ == "__main__":
    main()
