"""Phase A: OpenJev anatomy. Many seeded reads per canvas, plus the deterministic mean-embedding read.

  .venv/bin/python dlm/anatomy.py --dataset jevbench --seeds 16 --out results/dlm/anatomy_jevbench.jsonl
  .venv/bin/python dlm/anatomy.py --dataset typed-test --seeds 16 --out results/dlm/anatomy_typed_test.jsonl
  .venv/bin/python dlm/anatomy.py --dataset typed-train --seeds 16 --out results/dlm/anatomy_typed_train.jsonl

One JSONL row per (state, question): the gold label and distribution, every read's label probabilities
in label order, the top-k entropy OpenJev uses to trigger re-reads, and timings. Rows already in the
output file are skipped, so a run can be resumed. `dlm/metrics.py` scores the file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm import data  # noqa: E402
from dlm.reads import Reader  # noqa: E402

DATASETS = {"jevbench": lambda: data.jevbench_public(),
            "typed-test": lambda: data.typed_decisions("test"),
            "typed-train": lambda: data.typed_decisions("train")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-mean", action="store_true", help="skip the mean-embedding read")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    records = DATASETS[a.dataset]()
    if a.limit:
        records = records[: a.limit]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in records if r["id"] not in done]
    print(f"{a.dataset}: {len(records)} states, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    t0 = time.time()
    r = Reader()
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    with open(a.out, "a") as f:
        for n, rec in enumerate(todo, 1):
            t_start = time.time()
            prep = r.prepare(rec["state"], rec["questions"])
            n_prompt = len(prep["prompt"])
            t_prefill = time.time()
            r.prefill(prep["prompt"])
            t_prefill = time.time() - t_prefill
            reads, ents, t_reads = {}, {}, {}
            for seed in range(a.seeds):
                ts = time.time()
                probs, ent = r.read(prep, "random", seed=seed, with_entropy=True)
                t_reads[f"seed:{seed}"] = time.time() - ts
                reads[f"seed:{seed}"], ents[f"seed:{seed}"] = probs, ent
            if not a.no_mean:
                ts = time.time()
                probs, ent = r.read(prep, "mean", with_entropy=True)
                t_reads["mean"] = time.time() - ts
                reads["mean"], ents["mean"] = probs, ent
            row = {"id": rec["id"], "source": rec["source"], "family": rec["family"], "group": rec.get("group"),
                   "prompt_tokens": n_prompt, "canvas_width": prep["width"], "prefill_s": round(t_prefill, 3),
                   "questions": []}
            for qi, q in enumerate(prep["qs"]):
                names = [c[0] for c in q["choices"]]
                g = rec["gold"].get(q["key"], {})
                row["questions"].append({
                    "qid": q["key"], "type": q["type"], "labels": names, "slot_pos": prep["slots"][qi]["pos"],
                    "gold": g.get("label"), "gold_probs": g.get("probs"),
                    "reads": {k: v[qi] for k, v in reads.items()},
                    "entropy": {k: v[qi] for k, v in ents.items()}})
            row["read_s"] = {k: round(v, 3) for k, v in t_reads.items()}
            row["total_s"] = round(time.time() - t_start, 3)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} states, {el / 60:.1f} min, {el / n:.1f} s/state", flush=True)


if __name__ == "__main__":
    main()
