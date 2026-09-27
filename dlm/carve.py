"""Simulated expert carve: restrict every layer's router to the experts that carry a given fraction of
the census mass, and measure what the restricted model loses on decision reads. Nothing is deleted;
disallowed experts get -inf gate scores, so the forward pass is exactly what a physically carved
model would compute (prefill and canvas alike). Choose the experts on one dataset's census and
evaluate on the other to keep the selection honest.

  .venv/bin/python dlm/carve.py --census results/dlm/census_jevbench_en.json --keep 0.99 \
      --dataset typed-test --limit 200 --out results/dlm/carve_typed_test_0.99.jsonl
  .venv/bin/python dlm/carve.py --census results/dlm/census_typed_test.json --keep 0.99 \
      --dataset jevbench --tiers hard --out results/dlm/carve_jevbench_hard_0.99.jsonl
  .venv/bin/python dlm/carve.py --report results/dlm/carve_typed_test_0.99.jsonl

Allowed set per layer = union over census files and token roles (prompt, template, slot) of the
experts needed for `keep` of that role's gate mass. Per state: full and carved reads, random slot
(seed 0) and mean-embedding slot, on the given question order (`--order rotK` rotates typed-test
questions; rot0 is OpenJev's order).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm import data  # noqa: E402
from dlm.metrics import brier, ece, tv  # noqa: E402
from dlm.reads import Reader  # noqa: E402

ROLES = ("prompt", "template", "slot")


def allowed_sets(census_paths, keep):
    sets = None
    for path in census_paths:
        rep = json.load(open(path))
        for role in ROLES:
            mass = np.array(rep["mass"][role])
            if sets is None:
                sets = [set() for _ in range(mass.shape[0])]
            for layer, m in enumerate(mass):
                order = np.argsort(-m)
                cum = np.cumsum(m[order]) / max(m.sum(), 1e-12)
                sets[layer] |= set(order[: int(np.searchsorted(cum, keep) + 1)].tolist())
    return sets


def routers(r):
    from mlx_vlm.models.diffusion_gemma import language as L
    out = []
    for layer in r.dec.layers:
        rs = [m for _, m in layer.named_modules() if isinstance(m, L.Router)]
        assert len(rs) == 1, f"{len(rs)} routers in a layer"
        out.append(rs[0])
    return out


def set_carve(r, sets):
    """sets: per-layer allowed expert sets, or None to restore the full model."""
    n_experts = r.dec.config.num_experts
    for router, s in zip(routers(r), sets or [None] * len(r.dec.layers)):
        if s is None:
            router._dlm_allowed = None
        else:
            router._dlm_allowed = r.mx.array([i in s for i in range(n_experts)])
    r._prefills.clear()


def run(a):
    if a.dataset == "jevbench":
        recs = data.jevbench_public(tuple(a.tiers))
    else:
        recs = data.typed_decisions(a.dataset.split("-")[1])
    recs = recs[: a.limit]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [x for x in recs if x["id"] not in done]
    sets = allowed_sets(a.census, a.keep)
    kept = [len(s) for s in sets]
    print(f"{len(recs)} states, {len(todo)} to run; keep {a.keep}: {sum(kept) / len(kept):.1f} experts/layer "
          f"(min {min(kept)}, max {max(kept)}) of 128", flush=True)
    r = Reader()
    t0 = time.time()
    k = int(a.order[3:]) if a.order.startswith("rot") else 0
    with open(a.out, "a") as f:
        for n, rec in enumerate(todo, 1):
            qids = list(rec["questions"])
            qids = qids[k:] + qids[:k]
            prep = r.prepare(rec["state"], {q: rec["questions"][q] for q in qids})
            keys = [q["key"] for q in prep["qs"]]
            out = {key: {} for key in keys}
            for tag, carve in (("full", None), ("carved", sets)):
                set_carve(r, carve)
                for mode, suffix in (("random", "_random"), ("mean", "_mean")):
                    for key, p in zip(keys, r.read(prep, mode, seed=0)):
                        out[key][tag + suffix] = p
            set_carve(r, None)
            row = {"id": rec["id"], "family": rec.get("family"), "source": rec.get("source"), "keep": a.keep,
                   "experts_per_layer": kept, "questions": []}
            for q in prep["qs"]:
                g = rec["gold"].get(q["key"], {})
                row["questions"].append({"qid": q["key"], "type": q["type"], "labels": [c[0] for c in q["choices"]],
                                         "gold": g.get("label"), "reads": out[q["key"]]})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({(time.time() - t0) / n:.1f} s/state)", flush=True)


def report(path, by=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    kept = rows[0]["experts_per_layer"]
    print(f"{path}: {len(rows)} states, keep {rows[0]['keep']}, {sum(kept) / len(kept):.1f} experts/layer "
          f"({sum(kept) / (128 * len(kept)):.0%} of routed experts)")
    acc = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for q in row["questions"]:
            if q["gold"] not in q["labels"]:
                continue
            y = q["labels"].index(q["gold"])
            key = {"type": q["type"], "family": row["family"], "source": row["source"]}.get(by, "all") if by else "all"
            for c, p in q["reads"].items():
                top = max(range(len(p)), key=p.__getitem__)
                ref = q["reads"]["full" + c[c.index("_"):]]
                acc[key][c].append({"correct": top == y, "conf": p[top], "brier": brier(p, y), "tv": tv(p, ref),
                                    "flip": top != max(range(len(ref)), key=ref.__getitem__)})
    for key in sorted(acc):
        d = acc[key]
        print(f"\n== {key} (n={len(d['full_random'])} slots)")
        print(f"{'condition':14s} {'acc':>6s} {'brier':>6s} {'ece':>6s} {'tv->full':>8s} {'flip':>5s}")
        for c in ("full_random", "carved_random", "full_mean", "carved_mean"):
            v = d[c]
            n = len(v)
            print(f"{c:14s} {sum(x['correct'] for x in v) / n:6.3f} {sum(x['brier'] for x in v) / n:6.3f} "
                  f"{ece([(x['conf'], x['correct']) for x in v]):6.3f} {sum(x['tv'] for x in v) / n:8.3f} "
                  f"{sum(x['flip'] for x in v) / n:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", nargs="+", default=None)
    ap.add_argument("--keep", type=float, default=0.99)
    ap.add_argument("--dataset", choices=["jevbench", "typed-test", "typed-train"], default="typed-test")
    ap.add_argument("--tiers", nargs="+", default=["hard"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--order", default="rot0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--by", choices=["type", "family", "source"], default=None)
    a = ap.parse_args()
    if a.report:
        report(a.report, a.by)
    else:
        if not a.census or not a.out:
            ap.error("--census and --out are required to run")
        run(a)


if __name__ == "__main__":
    main()
