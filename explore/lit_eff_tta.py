"""Lit-scan idea 9: backprop-free, closed-form test-time refinement of Gaussian (LDA) branches from
unlabelled traffic (after ADAPT, Zhang et al., NeurIPS 2025, which does this for CLIP).

A few-label LDA branch (shared covariance, mean-pooled trunk states at depth k, additive sufficient
statistics as in geometry_sufficient.py) is deployed. Unlabelled test traffic then refines it, with no
gradients:
  none          labelled statistics only
  selftrain     one transductive round: the most confident fraction q of the test set is pseudo-labelled
                and added to the statistics, then the classifier is refitted (a self-training baseline)
  adapt_online  test messages arrive in random order; each is classified by the current branch; if its
                confidence clears tau it enters a bounded per-class bank (the M most confident per class);
                the classifier is refitted every U messages from labelled + bank statistics (bank weight w)
  adapt_2pass   adapt_online, then the whole stream is re-scored with the final banks (transductive)
Removing a bank is an exact subtraction of its statistics, so the refinement is reversible by design.

Labelled sets: 5 and 10 per class on average (random messages, 3 seeds; and one TypiClust selection),
no validation data (fixed shrinkage 0.5). Datasets: Banking77 intent (depth 8), CLINC150 intent and
domain (depth 11). CLINC intent also reports out-of-scope F1 (oos as its own class).

Prediction (lit_efficiency.md entry 9): +2-5 points at 5 per class, ~0 with many labels; watch oos F1
(confirmation bias).

Usage:
  .venv/bin/python explore/lit_eff_tta.py --smoke
  .venv/bin/python explore/lit_eff_tta.py --out results/tarski/explore_lit_tta.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_eff_common import GaussStats, Log, index, lda_logits, load_ds, onehot, pooled, save, select, texts, threads, ys

import numpy as np
import torch
from sklearn.metrics import f1_score

from tarski.train import FeatureCache

SPLIT = {"banking77": 8, "clinc150": 11}
TASKS = {"banking77": ["intent"], "clinc150": ["intent", "domain"]}


def probs(X, clf):
    return torch.softmax(lda_logits(X, clf).clamp_min(-1e4), -1)


def metrics(pred: np.ndarray, y: np.ndarray, oos: int = None) -> Dict:
    m = {"acc": float((pred == y).mean())}
    if oos is not None:
        m["oos_f1"] = float(f1_score(y == oos, pred == oos, zero_division=0))
        ins = y != oos
        m["in_scope_acc"] = float((pred[ins] == y[ins]).mean())
    return m


def adapt(base: GaussStats, Xte, gamma, tau, M, U, w, rng_seed, two_pass):
    C, D = base.s.shape
    order = np.random.default_rng(rng_seed).permutation(len(Xte))
    clf = base.classifier(gamma)
    bank: Dict[int, List] = {c: [] for c in range(C)}            # (conf, index)
    pred = np.zeros(len(Xte), dtype=int)
    for step, j in enumerate(order):
        p = probs(Xte[j:j + 1], clf)[0]
        c, conf = int(p.argmax()), float(p.max())
        pred[j] = c
        if conf >= tau:
            b = bank[c]
            if len(b) < M:
                b.append((conf, j))
            elif conf > min(b)[0]:
                b[b.index(min(b))] = (conf, j)
        if (step + 1) % U == 0:
            st = base.copy()
            for cc, b in bank.items():
                if b:
                    idx = [j for _, j in b]
                    P = torch.zeros(len(idx), C)
                    P[:, cc] = w
                    st.add(Xte[idx], P)
            clf = st.classifier(gamma)
    if two_pass:
        pred = probs(Xte, clf).argmax(-1).numpy()
    return pred, sum(len(b) for b in bank.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150"])
    ap.add_argument("--per-class", type=int, nargs="*", default=[5, 10])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, nargs="*", default=[0.9, 0.99])
    ap.add_argument("--bank", type=int, default=16, help="bank size per class (M)")
    ap.add_argument("--update-every", type=int, default=50)
    ap.add_argument("--bank-weight", type=float, default=0.5)
    ap.add_argument("--selftrain-frac", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.seeds, a.per_class, a.update_every, a.tau = "cpu", 1, [2], 10, [0.9]
        a.out = a.out or "results/tarski/explore_lit_tta_smoke.json"
    if not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    from tarski.trunk import Trunk
    trunk = Trunk(device=a.device)
    res = {"args": vars(a), "datasets": {}}
    for name in a.datasets:
        k = 4 if a.smoke else SPLIT[name]
        ds = load_ds(name, a.smoke, n_smoke=(300, 40, 120))
        allx, rng = index(ds)
        fc = FeatureCache(trunk, texts(ds), [k], ds.max_len)
        X = pooled(fc, k)
        log(f"== {name}: split {k}, cached in {fc.seconds:.0f}s")
        R = {}
        for task in TASKS[name]:
            C = len(ds.tasks[task].labels)
            oos = ds.tasks[task].labels.index("oos") if "oos" in ds.tasks[task].labels else None
            pool = [i for i in rng["train"] if task in allx[i].y]
            te = [i for i in rng["test"] if task in allx[i].y]
            Xs = ((X - X[pool].mean(0)) / X[pool].std(0).clamp_min(1e-4)).double()
            Xte = Xs[te]
            y_te = ys(allx, te, task)[0].numpy()
            R[task] = {}
            for pc in a.per_class:
                B = pc * C
                plans = [("random", s) for s in range(a.seeds)] + [("typiclust", 0)]
                runs = []
                for how, seed in plans:
                    t0 = time.time()
                    lab = [pool[j] for j in select(how, Xs[pool].float().numpy(), B, seed)]
                    y_lab = ys(allx, lab, task)[0]
                    base = GaussStats(C, Xs.shape[1]).add(Xs[lab], onehot(y_lab, C))
                    clf = base.classifier(a.gamma)
                    p0 = probs(Xte, clf)
                    r = {"selector": how, "seed": seed, "none": metrics(p0.argmax(-1).numpy(), y_te, oos)}
                    conf = p0.max(-1).values.numpy()
                    top = np.argsort(-conf)[: int(a.selftrain_frac * len(te))]
                    st = base.copy().add(Xte[top], onehot(p0.argmax(-1)[top], C))
                    r["selftrain"] = metrics(probs(Xte, st.classifier(a.gamma)).argmax(-1).numpy(), y_te, oos)
                    for tau in a.tau:
                        for two in (False, True):
                            pred, nb = adapt(base, Xte, a.gamma, tau, a.bank, a.update_every, a.bank_weight, seed, two)
                            r[f"adapt_{'2pass' if two else 'online'}_tau{tau}"] = {**metrics(pred, y_te, oos), "bank_size": nb}
                    r["s"] = round(time.time() - t0, 1)
                    runs.append(r)
                keys = [k_ for k_ in runs[0] if k_ not in ("selector", "seed", "s")]
                summ = {}
                for how in ("random", "typiclust"):
                    rr = [r for r in runs if r["selector"] == how]
                    summ[how] = {kk: {m: float(np.mean([r[kk][m] for r in rr])) for m in rr[0][kk] if m != "bank_size"}
                                 for kk in keys}
                R[task][pc] = {"runs": runs, "summary": summ}
                for how in ("random", "typiclust"):
                    log(f"   [{task}] {pc}/class {how:9s}: " + " | ".join(
                        f"{kk} {v['acc']:.4f}" + (f" (oosF1 {v['oos_f1']:.3f})" if "oos_f1" in v else "")
                        for kk, v in summ[how].items()))
                res["datasets"][name] = R
                save(res, a.out)
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
