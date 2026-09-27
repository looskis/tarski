"""Parity of the torch (bf16, CUDA) reader with the MLX (4-bit) reads stored in an anatomy file:
seed-0 random read and mean-embedding read on the same items, label-distribution TV, argmax
agreement, accuracy of each, and timings.

  .venv/bin/python dlm/parity_torch.py --anatomy results/dlm/anatomy_jevbench.jsonl --limit 40 --tiers hard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm import data  # noqa: E402
from dlm.metrics import tv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomy", required=True)
    ap.add_argument("--dataset", choices=["jevbench", "typed-test"], default="jevbench")
    ap.add_argument("--tiers", nargs="+", default=["hard"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--attn", default="eager")
    a = ap.parse_args()
    from dlm.reads_torch import Reader
    ref = {}
    with open(a.anatomy) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                ref[row["id"]] = row
    recs = data.jevbench_public(tuple(a.tiers)) if a.dataset == "jevbench" else data.typed_decisions("test")
    recs = [r for r in recs if r["id"] in ref][: a.limit]
    t0 = time.time()
    r = Reader(attn_implementation=a.attn)
    print(f"model loaded in {time.time() - t0:.0f} s; {len(recs)} items", flush=True)
    stats = {"random": [], "mean": []}
    t_prefill = t_read = 0.0
    for n, rec in enumerate(recs, 1):
        prep = r.prepare(rec["state"], rec["questions"])
        t = time.time()
        r.prefill(prep["prompt"])
        t_prefill += time.time() - t
        t = time.time()
        p_rand = r.read(prep, "random", seed=0)
        p_mean = r.read(prep, "mean")
        t_read += (time.time() - t) / 2
        for q, pr, pm in zip(ref[rec["id"]]["questions"], p_rand, p_mean):
            y = q["labels"].index(q["gold"]) if q["gold"] in q["labels"] else None
            for name, p, mlx in (("random", pr, q["reads"]["seed:0"]), ("mean", pm, q["reads"]["mean"])):
                top, top_m = max(range(len(p)), key=p.__getitem__), max(range(len(mlx)), key=mlx.__getitem__)
                stats[name].append({"tv": tv(p, mlx), "agree": top == top_m,
                                    "acc_torch": (top == y) if y is not None else None,
                                    "acc_mlx": (top_m == y) if y is not None else None})
        if n % 10 == 0 or n == len(recs):
            print(f"  {n}/{len(recs)}  prefill {t_prefill / n:.2f} s, read {t_read / n:.3f} s", flush=True)
    for name, v in stats.items():
        k = len(v)
        vv = [x for x in v if x["acc_torch"] is not None]
        print(f"{name:7s} n={k}  TV torch-vs-mlx mean {sum(x['tv'] for x in v) / k:.3f}  share>0.05 {sum(x['tv'] > 0.05 for x in v) / k:.2f}  "
              f"argmax agree {sum(x['agree'] for x in v) / k:.2f}  acc torch {sum(x['acc_torch'] for x in vv) / max(len(vv), 1):.3f}  "
              f"acc mlx {sum(x['acc_mlx'] for x in vv) / max(len(vv), 1):.3f}")


if __name__ == "__main__":
    main()
