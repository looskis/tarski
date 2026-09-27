"""Score anatomy files: reading conditions against gold and against the model's own noise-averaged read.

  .venv/bin/python dlm/metrics.py results/dlm/anatomy_jevbench.jsonl [more.jsonl ...] [--by type|family|source]

Conditions (derived offline from the stored reads):
  single    seed 0, OpenJev with samples=1
  auto4     OpenJev's default policy: seed 0, and the mean of seeds 0-3 if seed 0's top-k entropy > 0.1
  mean4     the mean of seeds 0-3, always
  meanK     the mean of every stored seed (the reference "noise-averaged" read)
  meanemb   the deterministic vocabulary-mean-embedding read
  query     a learned-query read, when present

Metrics follow JevBench: accuracy (argmax over the label set), Brier (2-class convention for two labels,
multi-class sum otherwise), ECE (top-label confidence, 10 equal-width bins). Against meanK: mean total
variation of a condition's distribution from meanK, and the argmax flip rate. Seed variance: the mean
TV of a single seeded read from meanK, and the share of slots where it exceeds 0.05.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def brier(p, y_idx):
    if len(p) == 2:
        return (p[y_idx] - 1.0) ** 2 + (p[1 - y_idx]) ** 2
    return sum((pi - (1.0 if i == y_idx else 0.0)) ** 2 for i, pi in enumerate(p))


def ece(pairs, bins=10):
    b = [[0, 0.0, 0] for _ in range(bins)]
    for conf, correct in pairs:
        i = min(int(min(max(conf, 0.0), 1.0) * bins), bins - 1)
        b[i][0] += 1
        b[i][1] += conf
        b[i][2] += 1 if correct else 0
    n = sum(x[0] for x in b)
    return sum((x[0] / n) * abs(x[2] / x[0] - x[1] / x[0]) for x in b if x[0]) if n else float("nan")


def tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def mean_of(reads):
    k = len(reads[0])
    return [sum(r[i] for r in reads) / len(reads) for i in range(k)]


def conditions(q):
    reads = q["reads"]
    seeds = sorted((k for k in reads if k.startswith("seed:")), key=lambda s: int(s.split(":")[1]))
    out = {"single": reads[seeds[0]]}
    first4 = [reads[s] for s in seeds[:4]]
    out["mean4"] = mean_of(first4)
    out["auto4"] = out["mean4"] if q["entropy"][seeds[0]] > 0.1 else reads[seeds[0]]
    out["meanK"] = mean_of([reads[s] for s in seeds])
    if "mean" in reads:
        out["meanemb"] = reads["mean"]
    for k in reads:
        if k.startswith("query"):
            out[k] = reads[k]
    return out, [reads[s] for s in seeds]


def score(rows, by=None):
    acc = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for q in row["questions"]:
            if q["gold"] is None or q["gold"] not in q["labels"]:
                continue
            y = q["labels"].index(q["gold"])
            conds, seeds = conditions(q)
            ref = conds["meanK"]
            key = {"type": q["type"], "family": row["family"], "source": row["source"]}.get(by, "all") if by else "all"
            for c, p in conds.items():
                top = max(range(len(p)), key=p.__getitem__)
                d = acc[key][c]
                d.append({"correct": top == y, "conf": p[top], "brier": brier(p, y),
                          "tv": tv(p, ref), "flip": top != max(range(len(ref)), key=ref.__getitem__)})
            sv = [tv(s, ref) for s in seeds]
            acc[key]["_seedvar"].append({"tv": sum(sv) / len(sv), "gt05": sum(v > 0.05 for v in sv) / len(sv)})
    return acc


def report(acc):
    for key in sorted(acc):
        d = acc[key]
        n = len(d["single"])
        sv = d["_seedvar"]
        print(f"\n== {key}  (n={n} slots)  single-read TV from meanK: mean {sum(s['tv'] for s in sv) / len(sv):.3f}, "
              f"share > 0.05: {sum(s['gt05'] for s in sv) / len(sv):.2f}")
        print(f"{'condition':10s} {'acc':>6s} {'brier':>6s} {'ece':>6s} {'tv->meanK':>9s} {'flip':>5s}")
        for c in [k for k in ("single", "auto4", "mean4", "meanK", "meanemb") if k in d] + sorted(k for k in d if k.startswith("query")):
            v = d[c]
            print(f"{c:10s} {sum(x['correct'] for x in v) / n:6.3f} {sum(x['brier'] for x in v) / n:6.3f} "
                  f"{ece([(x['conf'], x['correct']) for x in v]):6.3f} {sum(x['tv'] for x in v) / n:9.3f} "
                  f"{sum(x['flip'] for x in v) / n:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by", choices=["type", "family", "source"], default=None)
    a = ap.parse_args()
    rows = []
    for f in a.files:
        with open(f) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    report(score(rows, a.by))


if __name__ == "__main__":
    main()
