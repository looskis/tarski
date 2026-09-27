"""Add a learned query's reads to an anatomy file (e.g. JevBench, for zero-shot transfer of a query
trained on typed-decisions), so dlm/metrics.py can score it beside the other conditions.

  .venv/bin/python dlm/apply_query.py --queries results/dlm/queries_meanK_type.safetensors \
      --dataset jevbench --anatomy results/dlm/anatomy_jevbench.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm import data  # noqa: E402
from dlm.reads import Reader  # noqa: E402
from dlm.train_query import Queries, evaluate, load_anatomy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--dataset", choices=["jevbench", "typed-test", "typed-train"], required=True)
    ap.add_argument("--anatomy", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    loaders = {"typed-train": lambda: data.typed_decisions("train"), "typed-test": lambda: data.typed_decisions("test"),
               "jevbench": lambda: data.jevbench_public()}
    recs = loaders[a.dataset]()
    anatomy = load_anatomy(a.anatomy)
    r = Reader()
    queries, meta = Queries.load(a.queries, r)
    out = a.out or a.anatomy.replace(".jsonl", f".{meta['name']}.jsonl")
    m = evaluate(r, recs, anatomy, queries, meta["name"], out_path=out)
    print(json.dumps(m, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
