"""State-first prompting. OpenJev puts the question list in the system turn, before the state; the
encoder is causal, so the state's encoding depends on the question list and its order. Putting the
state first and the questions after it (same text, same canvas) makes the state's encoding
independent of the questions: order-invariant by construction on the state side, and one cached
prefix serves any question set. This measures what that costs or gains.

Formats:  stock        system = generic + question list; user = state                 (OpenJev)
          state_first  system = generic; user = state, then the question list
          user_first   system = generic; user = the question list, then the state   (control: the
                       questions still precede the state, just not in the system turn)
Orders rot0 / rot3 / rev; random slot (seed 0) and mean slot.

  .venv/bin/python dlm/state_first.py --limit 200 --out results/dlm/state_first_typed_test.jsonl
  .venv/bin/python dlm/state_first.py --report results/dlm/state_first_typed_test.jsonl
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
from dlm.backend import add_backend_arg, get_reader  # noqa: E402
from dlm.metrics import brier, ece, tv  # noqa: E402

FORMATS_ = ("stock", "state_first", "user_first")
ORDERS = {"rot0": 0, "rot3": 3, "rev": None}


def split_system(engine, qs, fmt):
    from openjev.engine import FORMATS
    full = engine.system_text(qs, fmt)
    head, rest = full.split("\nQuestion ", 1)
    fmt_line = FORMATS[fmt][2]
    qblock = "Question " + rest[: rest.rfind("\n" + fmt_line)].rstrip()
    generic = head.rstrip() + "\n\n" + fmt_line
    return generic, qblock


def prepare(r, state_text, questions, format_):
    schema = r.schema(questions)
    qs, fmt = schema["questions"], schema["format"]
    if format_ == "stock":
        sys_text, user = r.engine.system_text(qs, fmt), state_text
    else:
        generic, qblock = split_system(r.engine, qs, fmt)
        sys_text = generic
        user = (state_text + "\n\nThe questions:\n\n" + qblock) if format_ == "state_first" else \
               ("The questions:\n\n" + qblock + "\n\nThe state:\n\n" + state_text)
    prompt = r.engine.chat_prompt_ids(sys_text, user)
    template, slots = r.engine.resolve_template(qs, fmt)
    return {"qs": qs, "fmt": fmt, "forced": schema["forced"], "prompt": prompt, "template": template,
            "slots": slots, "width": r.engine.canvas_width(template)}


def run(a):
    if a.dataset == "jevbench":
        recs = data.jevbench_public(tuple(a.tiers))[: a.limit]
        orders = {"rot0": 0}                       # one question per item: order is moot
    else:
        recs = data.typed_decisions(a.split)[: a.limit]
        orders = ORDERS
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [x for x in recs if x["id"] not in done]
    print(f"{len(recs)} states, {len(todo)} to run", flush=True)
    r = get_reader(a.backend)
    t0 = time.time()
    with open(a.out, "a") as f:
        for n, rec in enumerate(todo, 1):
            qids = list(rec["questions"])
            out = {q: {} for q in qids}
            for oname, k in orders.items():
                order = list(reversed(qids)) if k is None else qids[k:] + qids[:k]
                qdict = {q: rec["questions"][q] for q in order}
                for format_ in FORMATS_:
                    prep = prepare(r, rec["state"], qdict, format_)
                    keys = [q["key"] for q in prep["qs"]]
                    for mode in ("random", "mean"):
                        for key, p in zip(keys, r.read(prep, mode, seed=0)):
                            out[key][f"{format_}|{oname}|{mode}"] = p
                    if n == 1 and oname == "rot0":
                        print(f"  {format_}: prompt {len(prep['prompt'])} tokens", flush=True)
            base = r.prepare(rec["state"], rec["questions"])
            row = {"id": rec["id"], "family": rec.get("family"), "source": rec.get("source"), "questions": []}
            for q in base["qs"]:
                g = rec["gold"].get(q["key"], {})
                row["questions"].append({"qid": q["key"], "type": q["type"], "labels": [c[0] for c in q["choices"]],
                                         "gold": g.get("label"), "reads": out[q["key"]]})
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({(time.time() - t0) / n:.1f} s/state)", flush=True)


def report(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"{len(rows)} states")
    for mode in ("random", "mean"):
        print(f"\n== slot {mode}")
        present = [o for o in ORDERS if any(f"{FORMATS_[0]}|{o}|{mode}" in q["reads"] for q in rows[0]["questions"])]
        print(f"{'format':12s} " + " ".join(f"{o:>16s}" for o in present) + f" {'spread':>7s} {'TV between orders':>18s}")
        for format_ in FORMATS_:
            stats, tvs = {}, []
            for oname in present:
                items = []
                for row in rows:
                    for q in row["questions"]:
                        if q["gold"] not in q["labels"]:
                            continue
                        p = q["reads"][f"{format_}|{oname}|{mode}"]; y = q["labels"].index(q["gold"])
                        top = max(range(len(p)), key=p.__getitem__)
                        items.append((top == y, p[top], brier(p, y)))
                n = len(items)
                stats[oname] = (sum(i[0] for i in items) / n, sum(i[2] for i in items) / n, ece([(i[1], i[0]) for i in items]))
            for row in rows:
                for q in row["questions"]:
                    ps = [q["reads"][f"{format_}|{o}|{mode}"] for o in present]
                    tvs.append(sum(tv(ps[i], ps[j]) for i in range(len(ps)) for j in range(i + 1, len(ps))) / max(len(ps) * (len(ps) - 1) / 2, 1))
            accs = [stats[o][0] for o in present]
            print(f"{format_:12s} " + " ".join(f"{stats[o][0]:.3f}/{stats[o][1]:.3f}/{stats[o][2]:.3f}" for o in present)
                  + f" {max(accs) - min(accs):7.3f} {sum(tvs) / len(tvs):18.3f}")
        print("  (cells: acc / Brier / ECE)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--dataset", choices=["typed", "jevbench"], default="typed")
    ap.add_argument("--tiers", nargs="+", default=["hard"])
    ap.add_argument("--out", default="results/dlm/state_first_typed_test.jsonl")
    ap.add_argument("--report", default=None)
    add_backend_arg(ap)
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        run(a)


if __name__ == "__main__":
    main()
