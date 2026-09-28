"""Lit-scan idea 4: an in-context tabular foundation model as a zero-gradient branch.

"Add a decision" becomes one in-context-learning call instead of a training run: the labelled messages'
PCA-reduced mean-pooled trunk states (at depth k) are TabICL's context and prediction is a forward pass.
No gradient loop, no hyperparameters, no early-stopping pitfalls.

Model: TabICLv2 (Qu, Holzmueller, Varoquaux, Le Morvan; ICML 2026), package `tabicl` (BSD-3-Clause code and
checkpoint, jingang/TabICL on the Hugging Face Hub). >10 classes use its hierarchical many-class mode.
Not included: MotherNet (not on PyPI; installable only from its GitHub repo) and TabPFN-2.5/3 (installable,
but the weights are licensed for non-commercial use only).

Label budgets (random labelled subsets, 3 seeds; plus one TypiClust selection from lit_eff_coldstart):
  typed-decisions  16 / 32 / 64 / 128 / all (~240) labelled messages per task (20 tasks, <= 10 options)
  clinc150         domain (11) and oos (2): 32 / 64 / 128 / 256 / 1024
  many-class       banking77 intent (77) and clinc150 intent (151): 5 and 10 per class (--many-class)
Learners on the same labelled messages, no validation data (last epoch, T = 1):
  tabicl      TabICL on PCA-d features (PCA fitted on the unlabelled pool)
  logreg_pca  logistic regression on the same PCA features (L2 1e-2, 300 steps)
  logreg      logistic regression on all 768 standardised features (L2 1e-3, 300 steps)
  lda         shared-covariance Gaussian on 768 features (shrinkage 0.5)
  knn         5-NN (cosine) on 768 features
  probe       tarski ProbeBranch via train_branch (>= 300 steps)
Metrics: acc, macro-F1, ECE, NLL; plus CPU latency of TabICL fit (context build) and batch-1 predict.

Prediction (lit_efficiency.md entry 4): with <= 64 labels, +2-5 points over logreg/probe on typed-decisions
and CLINC domain/oos with lower ECE; parity or slightly worse with all labels; worse on 77/151-way intents.

Usage (the first run needs the Hub to fetch the 110 MB checkpoint once):
  .venv/bin/python explore/lit_eff_icl.py --smoke
  HF_HUB_OFFLINE=0 .venv/bin/python explore/lit_eff_icl.py --datasets typed-decisions --out results/tarski/explore_lit_icl_typed.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_eff_common import (GaussStats, Log, fit_logreg, index, lda_logits, load_ds, make, onehot, pooled, save,
                            select, texts, threads, train_eval, ys)

import numpy as np
import torch
import torch.nn.functional as F

from tarski.train import FeatureCache, evaluate

SPLIT = {"banking77": 8, "clinc150": 11, "typed-decisions": 11}
BUDGETS = {"typed-decisions": [16, 32, 64, 128, 10 ** 6], "clinc150": [32, 64, 128, 256, 1024]}
PER_CLASS = [5, 10]


def pca_fit(X: torch.Tensor, d: int):
    mu = X.mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(X - mu, q=d, center=False, niter=4)
    return mu, V[:, :d]


def expand(p: np.ndarray, classes: np.ndarray, C: int) -> np.ndarray:
    full = np.full((p.shape[0], C), 1e-9)
    full[:, classes.astype(int)] = p
    return full / full.sum(1, keepdims=True)


def tabicl_probs(Xl, yl, Xt, C, dev, n_est):
    from tabicl import TabICLClassifier
    classes = np.unique(yl)
    if len(classes) == 1:
        p = np.zeros((len(Xt), C))
        p[:, classes[0]] = 1.0
        return p, 0.0
    t0 = time.time()
    clf = TabICLClassifier(device=str(dev), n_estimators=n_est, random_state=0, allow_auto_download=True)
    clf.fit(Xl, yl)
    p = clf.predict_proba(Xt)
    return expand(p, clf.classes_, C), time.time() - t0


def knn_probs(Xl, yl, Xt, C, k=5):
    A = F.normalize(Xl.float(), dim=-1)
    B = F.normalize(Xt.float(), dim=-1)
    s = B @ A.T
    k = min(k, len(yl))
    top = s.topk(k, dim=-1)
    p = torch.zeros(len(Xt), C)
    p.scatter_add_(1, torch.as_tensor(yl)[top.indices], torch.ones_like(top.values))
    p = (p + 1e-3) / (p + 1e-3).sum(1, keepdim=True)
    return p.numpy()


def run_task(name, ds, allx, rng, task, fc, X, a, dev, trunk, log, budgets, per_class: bool):
    C = len(ds.tasks[task].labels)
    pool = [i for i in rng["train"] if task in allx[i].y]
    te = [i for i in rng["test"] if task in allx[i].y]
    y_pool = ys(allx, pool, task)[0].numpy()
    y_te = ys(allx, te, task)[0].numpy()
    soft_te = ys(allx, te, task)[1]
    soft_te = None if soft_te is None else soft_te.numpy()
    Xs = (X - X[pool].mean(0)) / X[pool].std(0).clamp_min(1e-4)
    mu, V = pca_fit(Xs[pool], a.pca)
    P = (Xs - mu) @ V
    P = P / P[pool].std(0).clamp_min(1e-6)
    out = {}
    for B in budgets:
        nB = min(len(pool), B * C if per_class else B)
        plans = [("random", s) for s in range(a.seeds)] + ([("typiclust", 0)] if nB < len(pool) else [])
        if nB >= len(pool):
            plans = [("all", 0)]
        recs = []
        for how, seed in plans:
            if how == "all":
                lab_j = np.arange(len(pool))
            else:
                lab_j = select(how, Xs[pool].numpy(), nB, seed)
            lab = [pool[j] for j in lab_j]
            y_lab, soft_lab = ys(allx, lab, task)
            T = soft_lab if soft_lab is not None else onehot(y_lab, C)
            m = {}
            ev = lambda p: evaluate(np.asarray(p, dtype=np.float64), y_te, soft_te)
            p_icl, t_icl = tabicl_probs(P[lab].numpy(), y_lab.numpy(), P[te].numpy(), C, dev, a.n_estimators)
            m["tabicl"] = {**ev(p_icl), "fit_predict_s": round(t_icl, 2)}
            r = fit_logreg(P[lab], T, [P[te]], dev, l2s=(1e-2,), steps=a.steps)
            m["logreg_pca"] = ev(torch.softmax(r["logits"][0][0], -1).numpy())
            r = fit_logreg(Xs[lab], T, [Xs[te]], dev, l2s=(1e-3,), steps=a.steps)
            m["logreg"] = ev(torch.softmax(r["logits"][0][0], -1).numpy())
            st = GaussStats(C, Xs.shape[1]).add(Xs[lab], T)
            m["lda"] = ev(torch.softmax(lda_logits(Xs[te], st.classifier(0.5)).clamp_min(-1e4), -1).numpy())
            m["knn"] = ev(knn_probs(Xs[lab], y_lab, Xs[te], C))
            if "probe" in a.learners:
                br = make("probe", SPLIT[name], ds.tasks[task].labels, trunk)
                pr = train_eval(br, fc, allx, {"train": lab, "val": [], "test": te}, task, "probe", use_val=False,
                                min_steps=a.min_steps, epochs=a.probe_epochs)
                m["probe"] = {k: v for k, v in pr["test"].items()}
            recs.append({"selector": how, "seed": seed, "n_labelled": len(lab),
                         "classes_covered": int(len(set(y_lab.tolist()))), "metrics": m})
        agg = {}
        for learner in recs[0]["metrics"]:
            rr = [r for r in recs if r["selector"] in ("random", "all")]
            agg[learner] = {k: float(np.mean([r["metrics"][learner][k] for r in rr])) for k in ("acc", "macro_f1", "ece", "nll")}
        out[str(B)] = {"n_labelled": recs[0]["n_labelled"], "runs": recs, "random_mean": agg}
        log(f"   [{task}] n={recs[0]['n_labelled']:5d}: " + " | ".join(
            f"{l} {v['acc']:.3f}/{v['ece']:.3f}" for l, v in agg.items()) + "   (acc/ECE, random-selection mean)")
    return out


def latency(X_ctx: np.ndarray, y_ctx: np.ndarray, X_q: np.ndarray, n_est: int) -> Dict:
    from tabicl import TabICLClassifier
    torch.set_num_threads(8)
    out = {}
    clf = TabICLClassifier(device="cpu", n_estimators=n_est, kv_cache=True, random_state=0)
    t0 = time.time()
    clf.fit(X_ctx, y_ctx)
    out["fit_s"] = time.time() - t0
    clf.predict_proba(X_q[:1])
    ts = []
    for i in range(min(20, len(X_q))):
        t = time.perf_counter()
        clf.predict_proba(X_q[i:i + 1])
        ts.append((time.perf_counter() - t) * 1000)
    out["predict_ms_batch1_median"] = float(np.median(ts))
    t = time.perf_counter()
    clf.predict_proba(X_q[:100])
    out["predict_ms_per_msg_batch100"] = (time.perf_counter() - t) * 1000 / min(100, len(X_q))
    out["context_rows"], out["features"], out["n_estimators"] = int(len(X_ctx)), int(X_ctx.shape[1]), n_est
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=["typed-decisions", "clinc150"])
    ap.add_argument("--many-class", action="store_true", help="also banking77 / clinc150 intent at 5, 10 per class")
    ap.add_argument("--learners", nargs="*", default=["tabicl", "logreg", "lda", "knn", "probe"])
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-estimators", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--latency", action="store_true", help="CPU latency of TabICL fit/predict")
    ap.add_argument("--latency-only", action="store_true", help="CPU latency on random features only (no trunk)")
    ap.add_argument("--many-only", action="store_true", help="only the per-class many-class runs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.min_steps, a.steps, a.probe_epochs, a.seeds, a.n_estimators = "cpu", 20, 40, 1, 1, 2
        a.pca, a.latency = 16, True
        BUDGETS["typed-decisions"], BUDGETS["clinc150"] = [16, 10 ** 6], [32]
        PER_CLASS[:] = [2]
        a.many_class = True
        a.out = a.out or "results/tarski/explore_lit_icl_smoke.json"
    if not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    if a.latency_only:
        rs = np.random.default_rng(0)
        res = {"args": vars(a), "latency_cpu": {}}
        for n_ctx, C in ((240, 5), (1024, 2), (1024, 10)):
            Xc = rs.standard_normal((n_ctx, a.pca)).astype(np.float32)
            yc = rs.integers(0, C, n_ctx)
            lat = latency(Xc, yc, rs.standard_normal((200, a.pca)).astype(np.float32), a.n_estimators)
            res["latency_cpu"][f"{n_ctx}x{a.pca}:{C}cls"] = lat
            log(f"   TabICL CPU latency, context {n_ctx} x {a.pca}, {C} classes: fit {lat['fit_s']:.2f}s, predict batch-1 "
                f"{lat['predict_ms_batch1_median']:.1f} ms, batch-100 {lat['predict_ms_per_msg_batch100']:.2f} ms/msg")
        save(res, a.out)
        return
    from tarski.trunk import Trunk
    trunk = Trunk(device=a.device)
    dev = trunk.device
    res = {"args": vars(a), "budgets": BUDGETS, "per_class": PER_CLASS, "datasets": {}}
    names = list(a.datasets) + (["banking77"] if a.many_class and "banking77" not in a.datasets else [])
    for name in names:
        ds = load_ds(name, a.smoke, n_smoke=(200, 40, 60))
        allx, rng = index(ds)
        k = 4 if a.smoke else SPLIT[name]
        SPLIT[name] = k
        fc = FeatureCache(trunk, texts(ds), [k], ds.max_len)
        X = pooled(fc, k)
        log(f"== {name}: split {k}, {len(allx)} messages, cached in {fc.seconds:.0f}s")
        R = {}
        if name == "typed-decisions" and not a.many_only:
            tasks = list(ds.tasks)[: (2 if a.smoke else None)]
            for t in tasks:
                R[t] = run_task(name, ds, allx, rng, t, fc, X, a, dev, trunk, log, BUDGETS[name], False)
                res["datasets"][name] = R
                save(res, a.out)
        if name == "clinc150" and name in a.datasets and not a.many_only:
            for t in ("domain", "oos"):
                R[t] = run_task(name, ds, allx, rng, t, fc, X, a, dev, trunk, log, BUDGETS[name], False)
                res["datasets"][name] = R
                save(res, a.out)
        if a.many_class and name in ("banking77", "clinc150"):
            R["intent_per_class"] = run_task(name, ds, allx, rng, "intent", fc, X, a, dev, trunk, log, PER_CLASS, True)
        res["datasets"][name] = R
        if a.latency and name in ("typed-decisions", "clinc150"):
            t = list(ds.tasks)[0] if name == "typed-decisions" else "oos"   # KV cache needs <= 10 classes
            pool = [i for i in rng["train"] if t in allx[i].y]
            Xs = (X - X[pool].mean(0)) / X[pool].std(0).clamp_min(1e-4)
            mu, V = pca_fit(Xs[pool], a.pca)
            P = ((Xs - mu) @ V).numpy()
            yp = ys(allx, pool, t)[0].numpy()
            te = [i for i in rng["test"] if t in allx[i].y]
            for n_ctx in ((len(pool),) if name == "typed-decisions" else (256, 1024)):
                sel = pool[:n_ctx]
                lat = latency(P[sel], yp[: len(sel)], P[te], a.n_estimators)
                res.setdefault("latency_cpu", {})[f"{name}:{t}:{n_ctx}"] = lat
                log(f"   TabICL CPU latency, context {lat['context_rows']} x {lat['features']}: fit {lat['fit_s']:.2f}s, "
                    f"predict batch-1 {lat['predict_ms_batch1_median']:.1f} ms, batch-100 {lat['predict_ms_per_msg_batch100']:.2f} ms/msg")
        save(res, a.out)
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
