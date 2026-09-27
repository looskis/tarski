"""Expert census: which of DiffusionGemma's 128 experts per layer carry the routing mass on decision
canvases, split by token role (prompt tokens in the encoder prefill, template rows and answer-slot
rows in the decoder pass).

  .venv/bin/python dlm/census.py --dataset jevbench --limit 100 --out results/dlm/census_jevbench.json
  .venv/bin/python dlm/census.py --dataset typed-test --limit 100 --out results/dlm/census_typed_test.json

Per layer and token group: routing mass per expert (gate weight summed over tokens), the number of
experts needed for 90% / 99% of the mass, and the Jaccard overlap of the 99%-mass expert sets between
groups. Run with `--translate <lang>` later for the language control once translated states exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm import data  # noqa: E402
from dlm.reads import Reader, RoutingRecorder  # noqa: E402

GROUPS = ("prompt", "template", "slot")


def accumulate(mass, calls, n_layers, rows_by_group):
    """`calls`: list of (indices [N, k], weights [N, k]) in layer order; add gate mass per expert."""
    assert len(calls) == n_layers, f"{len(calls)} router calls, expected {n_layers}"
    for layer, (idx, w) in enumerate(calls):
        idx, w = np.array(idx.tolist()), np.array(w.tolist(), dtype=np.float64)
        for group, rows in rows_by_group.items():
            if not rows:
                continue
            i, ww = idx[rows].reshape(-1), w[rows].reshape(-1)
            np.add.at(mass[group][layer], i, ww)


def experts_for(mass_row, frac):
    order = np.argsort(-mass_row)
    cum = np.cumsum(mass_row[order]) / max(mass_row.sum(), 1e-12)
    return int(np.searchsorted(cum, frac) + 1), set(order[: int(np.searchsorted(cum, frac) + 1)].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["jevbench", "typed-test", "typed-train"], required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    loaders = {"typed-train": lambda: data.typed_decisions("train"), "typed-test": lambda: data.typed_decisions("test"),
               "jevbench": lambda: data.jevbench_public()}
    recs = loaders[a.dataset]()[: a.limit]
    r = Reader(prefill_cache_tokens=1)          # every prefill must run inside the recorder
    mx = r.mx
    n_layers, n_experts = len(r.dec.layers), r.dec.config.num_experts
    mass = {g: np.zeros((n_layers, n_experts)) for g in GROUPS}
    tokens = {g: 0 for g in GROUPS}

    for n, rec in enumerate(recs, 1):
        prep = r.prepare(rec["state"], rec["questions"])
        r._prefills.clear()
        with RoutingRecorder() as rr:
            cache = r.prefill(prep["prompt"])
        accumulate(mass, rr.calls, n_layers, {"prompt": list(range(len(prep["prompt"])))})
        tokens["prompt"] += len(prep["prompt"])
        positions = [s["pos"] for s in prep["slots"]]
        ids = mx.array([r.canvas(prep, 0)])
        with RoutingRecorder() as rr:
            logits = r.slot_logits(cache, r.embed(ids), positions, r.masks(cache, prep["width"]))
            mx.eval(logits)
        template_rows = [j for j in range(prep["width"]) if j not in positions]
        accumulate(mass, rr.calls, n_layers, {"template": template_rows, "slot": positions})
        tokens["template"] += len(template_rows)
        tokens["slot"] += len(positions)
        if n % 20 == 0:
            print(f"  {n}/{len(recs)}", flush=True)

    report = {"dataset": a.dataset, "states": len(recs), "tokens": tokens, "layers": []}
    for layer in range(n_layers):
        row = {"layer": layer}
        sets = {}
        for g in GROUPS:
            m = mass[g][layer]
            n90, _ = experts_for(m, 0.90)
            n99, s99 = experts_for(m, 0.99)
            sets[g] = s99
            p = m / max(m.sum(), 1e-12)
            row[g] = {"experts_90": n90, "experts_99": n99,
                      "entropy_bits": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
                      "top8": [int(x) for x in np.argsort(-m)[:8]]}
        row["jaccard_99"] = {f"{x}/{y}": (len(sets[x] & sets[y]) / max(len(sets[x] | sets[y]), 1))
                             for x, y in (("prompt", "slot"), ("prompt", "template"), ("template", "slot"))}
        report["layers"].append(row)
    report["mass"] = {g: mass[g].tolist() for g in GROUPS}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f)
    print(f"{'layer':>5s} {'prompt 90/99':>13s} {'template 90/99':>15s} {'slot 90/99':>11s} {'J(prompt,slot)':>15s}")
    for row in report["layers"]:
        print(f"{row['layer']:5d} {row['prompt']['experts_90']:6d}/{row['prompt']['experts_99']:<6d} "
              f"{row['template']['experts_90']:7d}/{row['template']['experts_99']:<7d} "
              f"{row['slot']['experts_90']:5d}/{row['slot']['experts_99']:<5d} {row['jaccard_99']['prompt/slot']:15.2f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
