"""A separate confidence head for tarski decisions (lit_decision_models #2, after Verdict 2.0).

Verdict 2.0 (github.com/Heman10x-NGU/openJev-verdict-2.0) reports that on typed-decisions its distribution
head has ECE 0.15, while a separate MLP "trained out-of-fold over prediction geometry" gets ECE 0.014 and
correctness AUROC 0.77. tarski's typed branches have ECE 0.13-0.17 against hard labels. Temperature scaling
(skeptic idea 3) cannot change the ranking of confidences, so it cannot raise AUROC; a learned correctness
model can.

The head here is one small model shared across all decisions. It predicts "is this branch's argmax right?"
from features a shared trunk gets for free:

  msp      max probability, margin, normalised entropy, number of options (of the main branch, T=1)
  qtype    question type (typed-decisions: yes/no, score, choice; CLINC: which task)
  depth    agreement of the main branch's argmax with probes at other depths, and their max probabilities
  bundle   the other decisions on the same message: mean/min max-probability and mean entropy; on CLINC also
           intent/domain consistency and intent-oos vs oos-binary agreement
  len      log token count

Main branch: blocks@14+2 (typed) / blocks@11+2 (CLINC); probes at --probe-depths. All trained with
tarski.train.train_branch (>= 300 optimiser steps). Protocols for the head:

  val->test   train the head on validation-split rows (all tasks pooled), score test rows
  lowo        typed only: train on validation rows of the other workflows, score the held-out workflow's test
              rows (does a head transfer to decisions it never saw?)
  oof         with --folds K: K-fold out-of-fold branch predictions on the training split train the head
              (costs K extra trainings of every branch; use --workflows to limit it)

Baselines: max probability at T=1, at the hard-label temperature fitted on validation data (skeptic 3), and
the deepest probe's max probability. Metrics on test rows: correctness AUROC, ECE, Brier, AURC and selective
accuracy at 100/80/60/50% coverage; plus feature-group ablations (msp only = a learned recalibration).

Usage:
  .venv/bin/python explore/lit_dm_confhead.py --smoke
  .venv/bin/python explore/lit_dm_confhead.py --dataset typed-decisions --out results/tarski/explore_confhead_typed.json
  .venv/bin/python explore/lit_dm_confhead.py --dataset clinc150 --out results/tarski/explore_confhead_clinc150.json
  .venv/bin/python explore/lit_dm_confhead.py --dataset typed-decisions --workflows customer_service --folds 5 \
      --out results/tarski/explore_confhead_typed_cs_oof.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lit_dm_common as C  # noqa: E402
from tarski.train import FeatureCache, predict_logits  # noqa: E402
from tarski.trunk import Trunk  # noqa: E402

GROUPS = ["msp", "qtype", "depth", "bundle", "len"]


def qtype_of(task: str, labels: Sequence[str], dataset: str) -> str:
    if dataset.startswith("clinc"):
        return task
    if list(labels) == ["false", "true"]:
        return "yesno"
    if all(l.isdigit() for l in labels):
        return "score"
    return "choice"


def softmax_np(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def entropy_norm(p: np.ndarray) -> float:
    return float(-(p * np.log(p.clip(1e-12))).sum() / np.log(max(len(p), 2)))


def train_models(trunk: Trunk, cache: FeatureCache, task: str, labels, tr, y_tr, soft_tr, va, y_va,
                 main: Tuple[int, int], probe_depths: Sequence[int], seed: int, log) -> Dict:
    """Main blocks branch + one probe per depth, trained on rows `tr`; returns the models and temperatures."""
    split, depth = main
    br = C.make("blocks", split, depth, labels, trunk)
    info = C.run_branch(br, cache, tr, y_tr, va, y_va, "blocks", seed=seed, soft_tr=soft_tr, log=log)
    probes = {}
    for d in probe_depths:
        pr = C.make("probe", d, 0, labels, trunk)
        C.run_branch(pr, cache, tr, y_tr, va, y_va, "probe", seed=seed, soft_tr=soft_tr, log=log)
        probes[d] = pr
    return {"main": br, "probes": probes, "T": float(br.temperature), "info": info}


def predict(models: Dict, cache: FeatureCache, idx: Sequence[int]) -> Dict:
    out = {"main": predict_logits(models["main"], cache, list(idx)).numpy(), "T": models["T"]}
    for d, pr in models["probes"].items():
        out[d] = predict_logits(pr, cache, list(idx)).numpy()
    return out


def build_rows(preds: Dict[str, Dict], rows_idx: Dict[str, List[int]], y: Dict[str, np.ndarray], tasks, labels_of,
               dataset: str, lengths, probe_depths, clinc_maps=None) -> List[Dict]:
    """One row per (message, task). preds[task] holds logits for rows_idx[task] (global message indices)."""
    pos = {t: {g: j for j, g in enumerate(rows_idx[t])} for t in tasks}
    msgs = sorted({g for t in tasks for g in rows_idx[t]})
    rows = []
    # per-message summaries of every task's main prediction, for bundle features
    summ = {}
    for t in tasks:
        P = softmax_np(preds[t]["main"])
        for g, j in pos[t].items():
            summ.setdefault(g, {})[t] = (P[j].max(), entropy_norm(P[j]), int(P[j].argmax()))
    for t in tasks:
        P = softmax_np(preds[t]["main"])
        PT = softmax_np(preds[t]["main"], preds[t]["T"])
        K = P.shape[1]
        qt = qtype_of(t, labels_of[t], dataset)
        for g, j in pos[t].items():
            p = np.sort(P[j])[::-1]
            am = int(P[j].argmax())
            f = {"msp.maxp": p[0], "msp.margin": p[0] - (p[1] if K > 1 else 0.0), "msp.ent": entropy_norm(P[j]),
                 "msp.logK": np.log(K), f"qtype.{qt}": 1.0}
            agree = []
            for d in probe_depths:
                Q = softmax_np(preds[t][d][j:j + 1])[0]
                a = float(Q.argmax() == am)
                agree.append(a)
                f[f"depth.agree{d}"] = a
                f[f"depth.maxp{d}"] = Q.max()
            f["depth.frac_agree"] = float(np.mean(agree)) if agree else 1.0
            others = [v for tt, v in summ[g].items() if tt != t]
            if others:
                f["bundle.mean_maxp"] = float(np.mean([o[0] for o in others]))
                f["bundle.min_maxp"] = float(np.min([o[0] for o in others]))
                f["bundle.mean_ent"] = float(np.mean([o[1] for o in others]))
            if clinc_maps is not None and all(k in summ[g] for k in ("intent", "domain", "oos")):
                intent_pred, dom_pred, oos_pred = summ[g]["intent"][2], summ[g]["domain"][2], summ[g]["oos"][2]
                f["bundle.intent_domain_agree"] = float(clinc_maps["dom_of_intent"][intent_pred] == dom_pred)
                f["bundle.intent_oos_agree"] = float((intent_pred == clinc_maps["oos_intent"]) == (oos_pred == 1))
            f["len.log"] = float(np.log(lengths[g]))
            rows.append({"g": g, "task": t, "qtype": qt, "f": f, "correct": float(am == y[t][j]),
                         "msp_T": float(PT[j].max()), "msp": float(p[0]),
                         "probe_deep": float(softmax_np(preds[t][max(probe_depths)][j:j + 1])[0].max())
                         if probe_depths else float(p[0])})
    return rows


def matrix(rows: List[Dict], names: List[str]) -> np.ndarray:
    return np.array([[r["f"].get(n, 0.0) for n in names] for r in rows], dtype=np.float64)


def score(conf: np.ndarray, correct: np.ndarray) -> Dict:
    from sklearn.metrics import roc_auc_score
    out = {"n": int(len(correct)), "acc": float(correct.mean())}
    out["auroc"] = float(roc_auc_score(correct, conf)) if 0 < correct.mean() < 1 else None
    out["ece"] = C.ece_score(conf, correct)
    out["brier"] = float(((conf - correct) ** 2).mean())
    out.update(C.selective(conf, correct))
    return out


def fit_heads(train_rows: List[Dict], test_rows: List[Dict], names_all: List[str], seed: int = 0) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_tr = np.array([r["correct"] for r in train_rows])
    y_te = np.array([r["correct"] for r in test_rows])
    out = {"n_train_rows": len(train_rows), "n_test_rows": len(test_rows)}
    if len(set(y_tr)) < 2:
        return out
    ablations = {"all": GROUPS, "msp": ["msp"], "msp+qtype": ["msp", "qtype"], "no_bundle": ["msp", "qtype", "depth", "len"],
                 "no_depth": ["msp", "qtype", "bundle", "len"]}
    for ab, groups in ablations.items():
        names = [n for n in names_all if n.split(".")[0] in groups]
        Xtr, Xte = matrix(train_rows, names), matrix(test_rows, names)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        lr.fit(Xtr, y_tr)
        out[f"lr[{ab}]"] = score(lr.predict_proba(Xte)[:, 1], y_te)
        if ab in ("all", "msp"):
            mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=3000,
                                                                 random_state=seed))
            mlp.fit(Xtr, y_tr)
            out[f"mlp[{ab}]"] = score(mlp.predict_proba(Xte)[:, 1], y_te)
    return out


def baselines(rows: List[Dict]) -> Dict:
    y = np.array([r["correct"] for r in rows])
    return {name: score(np.array([r[key] for r in rows]), y)
            for name, key in (("msp_T1", "msp"), ("msp_Thard", "msp_T"), ("probe_deep_maxp", "probe_deep"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="typed-decisions", choices=["typed-decisions", "clinc150"])
    ap.add_argument("--workflows", nargs="*", default=None, help="typed-decisions: only these workflows")
    ap.add_argument("--main-split", type=int, default=None, help="default 14 (typed) / 11 (clinc)")
    ap.add_argument("--main-depth", type=int, default=2)
    ap.add_argument("--probe-depths", type=int, nargs="*", default=None, help="default 6 <main-split> 22")
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.device = args.device or "cpu"
        args.base = args.base or C.SMOKE_BASE
        args.workflows = args.workflows or ["customer_service"]
        args.main_split, args.main_depth = 4, 2
        args.probe_depths = [2, 4, 7]
        args.folds = args.folds or 2
        args.out = args.out or "results/tarski/explore_confhead_smoke.json"
    log = C.Log(args.out)
    trunk = C.load_trunk(args.base, args.device)
    main_split = args.main_split or (14 if args.dataset == "typed-decisions" else 11)
    probe_depths = args.probe_depths or sorted({6, main_split, trunk.n_layers})
    main_cfg = (main_split, args.main_depth)

    ds = C.load_dataset_by_name(args.dataset)
    if args.workflows and args.dataset == "typed-decisions":
        ds = C.restrict_tasks(ds, [t for t in ds.tasks if t.split(".")[0] in args.workflows])
    if args.smoke:                       # 2 decisions per message (bundle features need >= 2), short inputs
        if args.dataset == "typed-decisions":
            ds = C.restrict_tasks(ds, list(ds.tasks)[:2])
        ds = C.subsample(ds, 70, 40, 40)
        ds.max_len = 128
    tasks = list(ds.tasks)
    log(f"== confidence head on {ds.summary()} | base {trunk.base} on {trunk.device} | main blocks@{main_split}"
        f"+{args.main_depth}, probes {probe_depths}, folds {args.folds}")

    allx = ds.train + ds.val + ds.test
    n_tr, n_va = len(ds.train), len(ds.val)
    split_of = ["train"] * n_tr + ["val"] * n_va + ["test"] * len(ds.test)
    cache = FeatureCache(trunk, [e.text for e in allx], sorted({main_split, *probe_depths}), ds.max_len)
    log(f"   cached {len(allx)} messages at {sorted({main_split, *probe_depths})} in {cache.seconds:.1f}s "
        f"({cache.bytes() / 1e6:.0f} MB)")
    idx = {s: {t: [i for i in range(len(allx)) if split_of[i] == s and t in allx[i].y] for t in tasks}
           for s in ("train", "val", "test")}
    Y = {s: {t: np.array([allx[i].y[t] for i in idx[s][t]]) for t in tasks} for s in idx}

    def soft_of(ids, t):
        if all(t in allx[i].soft for i in ids) and ids:
            return torch.tensor(np.stack([allx[i].soft[t] for i in ids]))
        return None

    clinc_maps = None
    if args.dataset == "clinc150":
        with open(os.path.join(C.REPO, "tarski", "resources", "clinc_domains.json")) as f:
            domains = json.load(f)
        intents, dom_labels = ds.tasks["intent"].labels, ds.tasks["domain"].labels
        dom_of = {i: d for d, its in domains.items() for i in its}
        clinc_maps = {"dom_of_intent": [dom_labels.index(dom_of.get(n, "oos")) for n in intents],
                      "oos_intent": intents.index("oos")}

    preds = {s: {} for s in ("val", "test", "oof")}
    branch_info = {}
    t0 = time.time()
    for t in tasks:
        labels = ds.tasks[t].labels
        log(f"-- {t} ({len(labels)} options, {len(idx['train'][t])} train, {len(idx['val'][t])} val)")
        tr, va = idx["train"][t], idx["val"][t]
        m = train_models(trunk, cache, t, labels, tr, torch.tensor(Y["train"][t]), soft_of(tr, t), va,
                         torch.tensor(Y["val"][t]), main_cfg, probe_depths, args.seed, log)
        branch_info[t] = m["info"]
        preds["val"][t] = predict(m, cache, va)
        preds["test"][t] = predict(m, cache, idx["test"][t])
        test_acc = float((preds["test"][t]["main"].argmax(-1) == Y["test"][t]).mean())
        branch_info[t]["test_acc"] = test_acc
        log(f"   main test acc {test_acc:.4f}")
        del m
        C.gpu_gc()
        if args.folds:
            rng = np.random.default_rng(args.seed)
            order = rng.permutation(len(tr))
            fold_of = np.empty(len(tr), int)
            for f_, chunk in enumerate(np.array_split(order, args.folds)):
                fold_of[chunk] = f_
            oof = {}
            for f_ in range(args.folds):
                fit_ids = [tr[j] for j in range(len(tr)) if fold_of[j] != f_]
                held = [tr[j] for j in range(len(tr)) if fold_of[j] == f_]
                yf = torch.tensor([allx[i].y[t] for i in fit_ids])
                mf = train_models(trunk, cache, t, labels, fit_ids, yf, soft_of(fit_ids, t), va,
                                  torch.tensor(Y["val"][t]), main_cfg, probe_depths, args.seed + 1 + f_, log)
                pf = predict(mf, cache, held)
                for key in ["main", *probe_depths]:
                    oof.setdefault(key, {}).update({g: pf[key][j] for j, g in enumerate(held)})
                oof.setdefault("T", []).append(pf["T"])
                del mf
                C.gpu_gc()
            preds["oof"][t] = {key: np.stack([oof[key][g] for g in tr]) for key in ["main", *probe_depths]}
            preds["oof"][t]["T"] = float(np.mean(oof["T"]))
    log(f"   branches trained in {time.time() - t0:.0f}s")

    labels_of = {t: ds.tasks[t].labels for t in tasks}
    lengths = cache.lengths
    rows = {s: build_rows(preds[s], idx[s], Y[s], tasks, labels_of, args.dataset, lengths, probe_depths, clinc_maps)
            for s in ("val", "test")}
    if args.folds:
        rows["oof"] = build_rows(preds["oof"], idx["train"], Y["train"], tasks, labels_of, args.dataset, lengths,
                                 probe_depths, clinc_maps)
    names_all = sorted({n for s in rows for r in rows[s] for n in r["f"]})
    res = {"config": vars(args) | {"base": trunk.base, "main": f"blocks@{main_split}+{args.main_depth}",
                                   "probe_depths": probe_depths, "features": names_all},
           "branches": branch_info, "results": {}}

    def report(name, test_rows, train_rows):
        r = {"baselines": baselines(test_rows), "heads": fit_heads(train_rows, test_rows, names_all, args.seed)}
        res["results"][name] = r
        b, h = r["baselines"], r["heads"]
        fmt = lambda d: (f"auroc {d['auroc']:.3f} ece {d['ece']:.3f} acc@80 {d['acc@80']:.3f} aurc {d['aurc']:.3f}"
                         if d and d.get("auroc") is not None else "n/a")
        log(f"== {name}: {len(train_rows)} head-training rows -> {len(test_rows)} test rows (acc {b['msp_T1']['acc']:.3f})")
        for k in ("msp_T1", "msp_Thard", "probe_deep_maxp"):
            log(f"   {k:18s} {fmt(b[k])}")
        for k in [k for k in h if k.startswith(("lr[", "mlp["))]:
            log(f"   {k:18s} {fmt(h[k])}")
        C.dump(res, args.out)

    report("val->test", rows["test"], rows["val"])
    if args.dataset == "typed-decisions":
        wfs = sorted({t.split(".")[0] for t in tasks})
        if len(wfs) > 1:
            y_all, p_all = [], {}
            per = {}
            for w in wfs:
                tr_rows = [r for r in rows["val"] if not r["task"].startswith(w + ".")]
                te_rows = [r for r in rows["test"] if r["task"].startswith(w + ".")]
                per[w] = {"baselines": baselines(te_rows), "heads": fit_heads(tr_rows, te_rows, names_all, args.seed)}
            res["results"]["lowo"] = per
            keys = ["lr[all]", "lr[msp]", "mlp[all]"]
            log("== leave-one-workflow-out (head never saw the workflow's decisions): AUROC per workflow")
            for w in wfs:
                log(f"   {w:28s} msp_Thard {per[w]['baselines']['msp_Thard']['auroc']:.3f} | " +
                    " ".join(f"{k} {per[w]['heads'][k]['auroc']:.3f}" for k in keys if k in per[w]["heads"]))
            C.dump(res, args.out)
    if args.folds:
        report("oof->test", rows["test"], rows["oof"])
    # per-task view of the pooled val->test head vs the temperature baseline
    by_task = {}
    for t in tasks:
        te = [r for r in rows["test"] if r["task"] == t]
        by_task[t] = {"acc": float(np.mean([r["correct"] for r in te])), "msp_Thard": baselines(te)["msp_Thard"]}
    res["by_task"] = by_task
    C.dump(res, args.out)
    log("done")


if __name__ == "__main__":
    main()
