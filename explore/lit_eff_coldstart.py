"""Lit-scan idea 3: cold-start selection. Which messages should a user label first for a new decision?

Before any label exists, the unlabelled training pool's mean-pooled trunk states (at the branch's split
depth) are clustered and the user labels B messages chosen by one of:
  random     B messages uniformly (3 seeds)
  typiclust  k-means with B clusters, the most typical (densest, 20-NN) message of each (Hacohen et al.,
             ICML 2022)
  kmedoid    k-means with B clusters, the message closest to each centroid
  probcover  greedy maximum coverage of delta-balls (Yehuda et al., NeurIPS 2022)
Every learner then trains on exactly those B labelled messages with NO validation data (the few-label
setting: last epoch, T = 1), and is scored on the full test split:
  logreg     pooled logistic regression (fixed L2 1e-3, 300 full-batch steps)
  lda        shared-covariance Gaussian (fixed shrinkage 0.5)
  probe      tarski ProbeBranch via tarski.train.train_branch (>= 300 steps)
  blocks:2   tarski BlockBranch (>= 300 steps), on a subset of budgets/selectors (--blocks-*)

Bundles: on CLINC150 one labelled message carries all three decisions (intent, domain, oos), so one
selection serves the bundle; on typed-decisions one selection per workflow serves its five questions.

Prediction (lit_efficiency.md entry 3): +8-12 points over random at <= 5 labels per class on intents
(CPU check on Banking77 depth 8 with logreg: 20.1/29.4/50.2 random vs 28.0/41.7/60.6 typiclust at
1/2/5 per class); smaller on typed-decisions, where typicality can over-sample the majority class.

Usage:
  .venv/bin/python explore/lit_eff_coldstart.py --smoke
  .venv/bin/python explore/lit_eff_coldstart.py --out results/tarski/explore_lit_coldstart.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_eff_common import (GaussStats, Log, fit_logreg, index, lda_logits, load_ds, make, onehot, pooled, save,
                            select, task_rows, texts, threads, train_eval, ys)

import numpy as np
import torch
from sklearn.metrics import f1_score

from tarski.train import FeatureCache

CONFIG = {  # split depth, budgets (labelled messages per pool), budgets that also train blocks:2
    "banking77": {"split": 8, "budgets": [77, 154, 385, 770], "blocks_budgets": [77, 385]},
    "clinc150": {"split": 11, "budgets": [150, 300, 750], "blocks_budgets": [150, 750]},
    "typed-decisions": {"split": 11, "budgets": [10, 40], "blocks_budgets": [40]},
}
SEEDS = {"random": [0, 1, 2], "typiclust": [0, 1], "kmedoid": [0, 1], "probcover": [0]}


def score(z: torch.Tensor, y: np.ndarray, C: int) -> Dict:
    pred = z.argmax(-1).numpy()
    return {"acc": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(C), zero_division=0))}


def run_pool(name, ds, allx, rng, tasks, fc, X, pool_rows, cfg, a, log, dev, trunk, blocks_tasks):
    """One unlabelled pool (all training messages of a dataset, or of one typed workflow)."""
    out = {}
    Xs = (X - X[pool_rows].mean(0)) / X[pool_rows].std(0).clamp_min(1e-4)
    feats = Xs[pool_rows].numpy()
    budgets = cfg["budgets"] if not a.smoke else cfg["budgets"][:2]
    for B in budgets:
        out[B] = {}
        for how in a.selectors:
            for seed in (SEEDS[how] if not a.smoke else SEEDS[how][:1]):
                t0 = time.time()
                chosen = [pool_rows[j] for j in select(how, feats, B, seed)]
                rec = {"selector": how, "seed": seed, "tasks": {}}
                for task in tasks:
                    C = len(ds.tasks[task].labels)
                    lab = [i for i in chosen if task in allx[i].y]
                    te = [i for i in rng["test"] if task in allx[i].y]
                    if not lab or not te:
                        continue
                    y_lab, soft = ys(allx, lab, task)
                    y_te = ys(allx, te, task)[0].numpy()
                    T = soft if soft is not None else onehot(y_lab, C)
                    r = {"n_labelled": len(lab), "classes_covered": int(len(set(y_lab.tolist()))), "n_classes": C}
                    lr = fit_logreg(Xs[lab], T, [Xs[te]], dev, l2s=(1e-3,), steps=a.steps)
                    r["logreg"] = score(lr["logits"][0][0], y_te, C)
                    st = GaussStats(C, Xs.shape[1]).add(Xs[lab], T)
                    r["lda"] = score(lda_logits(Xs[te], st.classifier(0.5)).float(), y_te, C)
                    sel = {"train": lab, "val": [], "test": te}
                    if "probe" in a.learners:
                        br = make("probe", cfg["split"], ds.tasks[task].labels, trunk)
                        pr = train_eval(br, fc, allx, sel, task, "probe", use_val=False, min_steps=a.min_steps,
                                        epochs=a.probe_epochs, seed=0)
                        r["probe"] = {"acc": pr["test"]["acc"], "macro_f1": pr["test"]["macro_f1"]}
                    if ("blocks:2" in a.learners and B in cfg["blocks_budgets"] and how in a.blocks_selectors
                            and seed in SEEDS[how][: a.blocks_seeds] and task in blocks_tasks):
                        bb = make("blocks:2", cfg["split"], ds.tasks[task].labels, trunk)
                        pb = train_eval(bb, fc, allx, sel, task, "blocks", use_val=False, min_steps=a.min_steps, seed=0)
                        r["blocks:2"] = {"acc": pb["test"]["acc"], "macro_f1": pb["test"]["macro_f1"]}
                    rec["tasks"][task] = r
                rec["s"] = round(time.time() - t0, 1)
                out[B].setdefault(how, []).append(rec)
            # summary line: mean over seeds and tasks
            recs = out[B][how]
            summ = {}
            for learner in ("logreg", "lda", "probe", "blocks:2"):
                v = [r[learner]["acc"] for rec in recs for r in rec["tasks"].values() if learner in r]
                if v:
                    summ[learner] = float(np.mean(v))
            cov = float(np.mean([r["classes_covered"] / r["n_classes"] for rec in recs for r in rec["tasks"].values()]))
            log(f"   B={B:4d} {how:9s}: " + " | ".join(f"{k} {v:.4f}" for k, v in summ.items()) + f" | class coverage {cov:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150", "typed-decisions"])
    ap.add_argument("--selectors", nargs="*", default=["random", "typiclust", "kmedoid", "probcover"])
    ap.add_argument("--learners", nargs="*", default=["logreg", "lda", "probe", "blocks:2"])
    ap.add_argument("--blocks-selectors", nargs="*", default=["random", "typiclust", "probcover"])
    ap.add_argument("--blocks-seeds", type=int, default=2, help="seeds per selector that also train blocks:2")
    ap.add_argument("--blocks-workflows", nargs="*", default=["customer_service"])
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.min_steps, a.steps, a.probe_epochs = "cpu", 20, 40, 1
        a.out = a.out or "results/tarski/explore_lit_coldstart_smoke.json"
        for c, b in (("banking77", [20, 40]), ("clinc150", [20, 40]), ("typed-decisions", [10, 20])):
            CONFIG[c]["budgets"], CONFIG[c]["blocks_budgets"] = b, b[:1]
            CONFIG[c]["split"] = 4
    if not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    from tarski.trunk import Trunk
    trunk = Trunk(device=a.device)
    dev = trunk.device
    res = {"args": vars(a), "config": CONFIG, "datasets": {}}
    for name in a.datasets:
        cfg = CONFIG[name]
        ds = load_ds(name, a.smoke, n_smoke=(200, 40, 80))
        allx, rng = index(ds)
        fc = FeatureCache(trunk, texts(ds), [cfg["split"]], ds.max_len)
        X = pooled(fc, cfg["split"])
        log(f"== {name}: pool {len(rng['train'])} messages, split {cfg['split']}, cached in {fc.seconds:.0f}s")
        t0 = time.time()
        if name == "typed-decisions":
            res["datasets"][name] = {}
            wfs = sorted({t.split(".")[0] for t in ds.tasks})
            for wf in wfs:
                tasks = [t for t in ds.tasks if t.startswith(wf + ".")]
                pool_rows = [i for i in rng["train"] if tasks[0] in allx[i].y]
                log(f"  workflow {wf}: pool {len(pool_rows)}")
                res["datasets"][name][wf] = run_pool(name, ds, allx, rng, tasks, fc, X, pool_rows, cfg, a, log, dev,
                                                     trunk, tasks if wf in a.blocks_workflows else [])
                save(res, a.out)
        else:
            tasks = list(ds.tasks)
            res["datasets"][name] = run_pool(name, ds, allx, rng, tasks, fc, X, rng["train"], cfg, a, log, dev, trunk,
                                             tasks)
        res.setdefault("wall_s", {})[name] = round(time.time() - t0, 1)
        save(res, a.out)
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
