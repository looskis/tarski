"""Post-hoc calibration baselines for a stored read, so a learned query can be compared with what
plain calibration of the same read achieves.

  .venv/bin/python dlm/calibrate.py --read meanemb --train results/dlm/anatomy_typed_train.jsonl \
      --test results/dlm/anatomy_typed_test.jsonl
  .venv/bin/python dlm/calibrate.py --read query:gold_type --cv 2 --test results/dlm/anatomy_typed_test.gold_type.jsonl

Per question type: (a) the train label marginal alone, (b) temperature scaling of the read
(p ∝ p^(1/T), T fitted by NLL on gold labels), (c) temperature + mixing with the label marginal
(fitted jointly). With --train the fit uses the train anatomy; with --cv k (for reads that exist
only on test, e.g. a query) it is k-fold cross-fitted on the test file and every slot is scored by
a fit that did not see it. Temperature never changes the argmax, so accuracy differences between
the calibrated read and a query are ranking changes, not calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import brier, conditions, ece  # noqa: E402

TS = [0.25 * 1.15 ** i for i in range(40)]      # 0.25 .. ~59
ALPHAS = [i / 20 for i in range(11)]            # 0 .. 0.5


class Priors(dict):
    """label marginals keyed by label tuple; uniform for label sets unseen in the fit"""
    def __missing__(self, labels):
        return [1 / len(labels)] * len(labels)


def load(path, read):
    out = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for q in row["questions"]:
                if q["gold"] is None or q["gold"] not in q["labels"]:
                    continue
                conds, _ = conditions(q)
                if read not in conds:
                    continue
                out[q["type"]].append((conds[read], tuple(q["labels"]), q["labels"].index(q["gold"])))
    return out


def scale(p, T, alpha, prior):
    z = [max(x, 1e-9) ** (1.0 / T) for x in p]
    s = sum(z)
    return [(1 - alpha) * a / s + alpha * b for a, b in zip(z, prior)]


def fit(items):
    counts = defaultdict(lambda: defaultdict(int))
    for _, labels, y in items:
        counts[labels][y] += 1
    priors = Priors()
    for labels, c in counts.items():
        n = sum(c.values())
        priors[labels] = [(c[i] + 0.5) / (n + 0.5 * len(labels)) for i in range(len(labels))]

    def nll(T, alpha):
        return -sum(math.log(max(scale(p, T, alpha, priors[labels])[y], 1e-12)) for p, labels, y in items) / len(items)

    T1 = min(TS, key=lambda T: nll(T, 0.0))
    T2, a2 = min(((T, al) for T in TS for al in ALPHAS), key=lambda x: nll(*x))
    return priors, T1, (T2, a2)


def score(items_out):
    n = len(items_out)
    acc = sum(top == y for _, top, y in items_out) / n
    br = sum(brier(q, y) for q, _, y in items_out) / n
    return acc, br, ece([(q[top], top == y) for q, top, y in items_out])


def apply(items, fn):
    out = []
    for p, labels, y in items:
        q = fn(p, labels)
        out.append((q, max(range(len(q)), key=q.__getitem__), y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", default="meanemb")
    ap.add_argument("--train", default=None)
    ap.add_argument("--test", required=True)
    ap.add_argument("--cv", type=int, default=None, help="k-fold cross-fit on the test file instead of --train")
    a = ap.parse_args()
    if not a.train and not a.cv:
        ap.error("--train or --cv")
    test = load(a.test, a.read)
    train = load(a.train, a.read) if a.train else None
    how = f"fitted on {sum(map(len, train.values()))} train slots" if train else f"{a.cv}-fold cross-fitted on test"
    print(f"read {a.read}; {how}; scored on {sum(map(len, test.values()))} test slots")
    print(f"{'type':8s} {'condition':22s} {'acc':>6s} {'brier':>6s} {'ece':>6s}   fit")
    for t in sorted(test):
        folds = [(train[t], test[t])] if train else \
            [([x for i, x in enumerate(test[t]) if i % a.cv != k], [x for i, x in enumerate(test[t]) if i % a.cv == k])
             for k in range(a.cv)]
        outs = {"raw": [], "label marginal only": [], "temperature": [], "temperature + marginal": []}
        fits = []
        for fit_items, score_items in folds:
            priors, T1, (T2, a2) = fit(fit_items)
            fits.append(f"T={T1:.2f}; T={T2:.2f} alpha={a2:.2f}")
            outs["raw"] += apply(score_items, lambda p, l: p)
            outs["label marginal only"] += apply(score_items, lambda p, l: priors[l])
            outs["temperature"] += apply(score_items, lambda p, l: scale(p, T1, 0.0, priors[l]))
            outs["temperature + marginal"] += apply(score_items, lambda p, l: scale(p, T2, a2, priors[l]))
        for name, o in outs.items():
            acc, br, e = score(o)
            print(f"{t:8s} {name:22s} {acc:6.3f} {br:6.3f} {e:6.3f}   {' | '.join(fits) if name == 'temperature + marginal' else ''}")


if __name__ == "__main__":
    main()
