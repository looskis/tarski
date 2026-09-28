"""Lit-scan idea 1: a "shadow trunk". Static token vectors that reproduce the trunk's state at the split
depth give every existing probe a zero-trunk tier; risk-controlled thresholds decide when to skip the trunk.

Two shadows per split depth k, both fitted on the UNLABELLED training messages only:
  tokmean   one vector per token id = the mean of that token's depth-k states over its occurrences
            (a per-token regression target). A message's shadow token states are the lookups, so tarski's
            ProbeBranch (norm per token, then mean pool) and even a blocks branch run on them unchanged.
  poolfit   vectors E fitted by ridge least squares so that mean_t E[t] ~= the mean-pooled depth-k state
            (Tokenlearn's objective with tarski's own trunk as the teacher). Used with pooled heads.

Served branches (trained on real trunk states, as today): probe@k (tarski ProbeBranch), a pooled logistic
head (logreg@k), and blocks:2@k (optional). Tier-0 candidates, all trunk-free at serving time:
  same_probe_tokmean   the served probe run on tokmean shadow tokens (no new head)
  same_logreg_poolfit  the served logistic head on the poolfit shadow (no new head)
  logreg_poolfit_rt    a logistic head retrained on shadow features of the training messages
  tfidf                TF-IDF (1-2 grams) + logistic regression, a plain bag-of-words control
Cascade: a message's bundle (all its decisions) exits at tier 0 only if every decision's tier-0
confidence clears tau. tau is chosen on validation with Learn-then-Test so that P(exit and any decision
differs from the served branch) <= delta with probability 1 - eps (Jazbec et al., NeurIPS 2024).
Reported: bundle exit rate, realised flip rate and accuracy on test, plus fixed-tau rows, and CPU
latency per message for tier 0 vs the trunk to depth k.

Prediction (lit_efficiency.md entry 1): Banking77 60-75% exits at <= 1% flips vs the probe (fewer vs
blocks); CLINC lower (the oos decision); typed-decisions low (< 20%).

Usage:
  .venv/bin/python explore/lit_eff_shadow.py --smoke
  .venv/bin/python explore/lit_eff_shadow.py --out results/tarski/explore_lit_shadow.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_eff_common import (Log, cache_from_states, fit_logreg, index, load_ds, ltt_threshold, make, onehot, pooled,
                            save, standardize, task_rows, texts, threads, train_eval, ys)

import numpy as np
import torch

from tarski.train import FeatureCache, predict_logits
from tarski.trunk import Trunk

DEFAULT_DEPTHS = {"banking77": [4, 8], "clinc150": [6, 11], "typed-decisions": [11]}


# ---------------------------------------------------------------------------------------------------
# Shadows
# ---------------------------------------------------------------------------------------------------

def fit_tokmean(fc: FeatureCache, ids: List[List[int]], rows: List[int], depth: int, V: int) -> torch.Tensor:
    """(V, D) mean depth-k state of each token id over `rows`; unseen ids get the global token mean."""
    D = fc.h[depth][rows[0]].shape[1]
    s = torch.zeros(V, D, dtype=torch.float64)
    n = torch.zeros(V, dtype=torch.float64)
    for i in rows:
        t = torch.tensor(ids[i])
        s.index_add_(0, t, fc.h[depth][i].double())
        n.index_add_(0, t, torch.ones(len(t), dtype=torch.float64))
    glob = s.sum(0) / n.sum()
    E = torch.where(n[:, None] > 0, s / n.clamp_min(1)[:, None], glob[None])
    return E.float(), n > 0


def count_matrix(ids: List[List[int]], rows: List[int], vocab: Dict[int, int]) -> torch.Tensor:
    """Dense (len(rows), |vocab|) matrix of token frequencies normalised over in-vocabulary tokens."""
    C = torch.zeros(len(rows), len(vocab))
    for r, i in enumerate(rows):
        toks = [vocab[t] for t in ids[i] if t in vocab]
        if toks:
            idx = torch.tensor(toks)
            C[r].index_add_(0, idx, torch.full((len(toks),), 1.0 / len(toks)))
    return C


def fit_poolfit(ids, rows, P: torch.Tensor, dev, lam: float = 1e-3):
    """Ridge least squares for static vectors E with C E ~= P (P: pooled states of `rows`)."""
    vocab = {t: j for j, t in enumerate(sorted({t for i in rows for t in ids[i]}))}
    C = count_matrix(ids, rows, vocab).to(dev)
    A = C.T @ C
    A += lam * A.diagonal().mean() * torch.eye(len(vocab), device=dev)
    B = C.T @ P.to(dev)
    E = torch.linalg.solve(A.double(), B.double()).float()
    return E.cpu(), vocab


def shadow_pooled_poolfit(ids, rows, E, vocab) -> torch.Tensor:
    return count_matrix(ids, rows, vocab) @ E


# ---------------------------------------------------------------------------------------------------
# Heads on pooled features
# ---------------------------------------------------------------------------------------------------

def logreg_head(X, allx, sel, task, C, dev, steps):
    """Logistic head on standardised pooled features; L2 chosen on validation. Returns (mu, sd, W, b)."""
    mu, sd = X[sel["train"]].mean(0, keepdim=True), X[sel["train"]].std(0, keepdim=True).clamp_min(1e-4)
    Z = lambda A: (A - mu) / sd
    y_tr, soft = ys(allx, sel["train"], task)
    T = soft if soft is not None else onehot(y_tr, C)
    r = fit_logreg(Z(X[sel["train"]]), T, [Z(X[sel["val"]])], dev, steps=steps)
    y_va = ys(allx, sel["val"], task)[0].numpy()
    g = int(np.argmax((r["logits"][0].argmax(-1).numpy() == y_va[None]).mean(-1)))
    return mu, sd, r["W"][g], r["b"][g]


def head_probs(head, X) -> np.ndarray:
    mu, sd, W, b = head
    return torch.softmax(((X - mu) / sd) @ W + b, -1).numpy()


def tfidf_probs(allx, sel, task):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Xtr = vec.fit_transform([allx[i].text for i in sel["train"]])
    ytr = ys(allx, sel["train"], task)[0].numpy()
    yva = ys(allx, sel["val"], task)[0].numpy()
    best, clf = -1, None
    for Cc in (1.0, 10.0):
        m = LogisticRegression(C=Cc, max_iter=2000).fit(Xtr, ytr)
        a = (m.predict(vec.transform([allx[i].text for i in sel["val"]])) == yva).mean()
        if a > best:
            best, clf = a, m
    out = {}
    for s in ("val", "test"):
        p = clf.predict_proba(vec.transform([allx[i].text for i in sel[s]]))
        out[s] = (p, clf.classes_)
    return out


def expand(p: np.ndarray, classes: np.ndarray, C: int) -> np.ndarray:
    full = np.zeros((p.shape[0], C))
    full[:, classes] = p
    return full


# ---------------------------------------------------------------------------------------------------
# Cascade evaluation
# ---------------------------------------------------------------------------------------------------

def bundle_eval(tasks, served: Dict, tier0: Dict, gold: Dict, deltas, eps, taus=(0.5, 0.7, 0.9, 0.95)):
    """served/tier0/gold: {split: {task: {row: value}}} (served & tier0 hold (pred, conf)).
    Bundles = messages; a bundle is every task that message carries."""
    def arrays(split):
        rows = sorted({r for t in tasks for r in gold[split][t]})
        conf, flip, n_pairs, c_served, c_cascade_parts, c_t0 = [], [], 0, 0, [], 0
        per = []
        for r in rows:
            ts = [t for t in tasks if r in gold[split][t]]
            conf.append(min(tier0[split][t][r][1] for t in ts))
            flip.append(any(tier0[split][t][r][0] != served[split][t][r][0] for t in ts))
            per.append([(tier0[split][t][r][0] == gold[split][t][r], served[split][t][r][0] == gold[split][t][r]) for t in ts])
        return np.array(conf), np.array(flip), per

    cv, fv, _ = arrays("val")
    ct, ft, per = arrays("test")
    n_pairs = sum(len(p) for p in per)
    acc_served = sum(s for p in per for _, s in p) / n_pairs
    acc_t0 = sum(t for p in per for t, _ in p) / n_pairs

    def at(tau):
        ex = ct >= tau
        acc = sum((t if e else s) for e, p in zip(ex, per) for t, s in p) / n_pairs
        return {"tau": float(tau), "exit_rate": float(ex.mean()), "flip_rate": float((ex & ft).mean()),
                "acc": float(acc)}

    out = {"n_bundles_test": int(len(ct)), "n_bundles_val": int(len(cv)), "acc_served": float(acc_served),
           "acc_tier0_only": float(acc_t0), "bundle_agreement_test": float(1 - ft.mean()), "fixed": [at(t) for t in taus],
           "ltt": {}}
    for d in deltas:
        tau = ltt_threshold(cv, fv, d, eps)
        out["ltt"][str(d)] = at(tau) if tau is not None else {"tau": None, "exit_rate": 0.0, "flip_rate": 0.0,
                                                               "acc": float(acc_served), "note": "not certifiable on validation"}
    return out


# ---------------------------------------------------------------------------------------------------
# Latency (CPU, batch 1)
# ---------------------------------------------------------------------------------------------------

@torch.no_grad()
def latency(msgs: List[str], depth: int, E_tok: torch.Tensor, head, reps: int = 5) -> Dict:
    import statistics
    trunk = Trunk(device="cpu")
    ids = trunk.token_ids(msgs, 512)
    En = E_tok.numpy()
    mu, sd, W, b = [x.numpy() for x in head]
    t_trunk, t_t0 = [], []
    for m, idl in zip(msgs, ids):
        x = torch.tensor([idl])
        att = torch.ones_like(x)
        trunk.taps(x, att, [depth])
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            h, _ = trunk.taps(x, att, [depth])
            h[depth].mean(1)
            ts.append((time.perf_counter() - t) * 1000)
        t_trunk.append(statistics.median(ts))
        arr = np.array(idl)
        ts = []
        for _ in range(reps * 20):
            t = time.perf_counter()
            v = En[arr].mean(0)
            z = ((v - mu[0]) / sd[0]) @ W + b[0]
            z.argmax()
            ts.append((time.perf_counter() - t) * 1000)
        t_t0.append(statistics.median(ts))
    return {"depth": depth, "threads": torch.get_num_threads(), "n_msgs": len(msgs),
            "trunk_to_depth_ms_median": float(np.median(t_trunk)), "tier0_ms_median": float(np.median(t_t0)),
            "speedup": float(np.median(t_trunk) / max(np.median(t_t0), 1e-6)),
            "mean_tokens": float(np.mean([len(i) for i in ids]))}


# ---------------------------------------------------------------------------------------------------

def run_dataset(name, trunk, a, log, dev):
    ds = load_ds(name, a.smoke)
    allx, rng = index(ds)
    tasks = list(ds.tasks)
    depths = a.depths or DEFAULT_DEPTHS[name]
    if a.smoke:
        depths = depths[:1]
        tasks = tasks[:3]
    t0 = time.time()
    fc = FeatureCache(trunk, texts(ds), depths, ds.max_len)
    ids = trunk.token_ids(texts(ds), ds.max_len)
    V = len(trunk.tok)
    log(f"== {name}: {len(allx)} messages, tasks {len(tasks)}, depths {depths}, cached in {fc.seconds:.0f}s")
    sels = {t: task_rows(allx, rng, t) for t in tasks}
    train_rows = rng["train"]
    res = {"depths": {}, "n_tasks": len(tasks)}
    blocks_here = a.blocks and name in a.blocks_datasets
    tfidf_cache: Dict[str, Dict] = {}
    for k in depths:
        out = {"tasks": {}}
        # --- shadows (unlabelled training messages only) ---
        E_tok, seen = fit_tokmean(fc, ids, train_rows, k, V)
        P = pooled(fc, k)
        E_pf, vocab = fit_poolfit(ids, train_rows, P[train_rows], dev, a.lam)
        all_rows = list(range(len(allx)))
        S_pf = shadow_pooled_poolfit(ids, all_rows, E_pf, vocab)
        S_tm = torch.stack([E_tok[torch.tensor(ids[i])].mean(0) for i in all_rows])
        shadow_fc = cache_from_states(trunk, {k: [E_tok[torch.tensor(ids[i])] for i in all_rows]})
        te = rng["test"]
        cos = lambda A, B: float(torch.nn.functional.cosine_similarity(A[te], B[te], dim=-1).mean())
        oov = float(np.mean([np.mean([not bool(seen[t]) for t in ids[i]]) for i in te]))
        centred = lambda A: A - P[train_rows].mean(0)
        out["shadow"] = {"vocab_seen": int(seen.sum()), "test_oov_token_rate": oov,
                         "cos_pooled_test_poolfit": cos(S_pf, P), "cos_pooled_test_tokmean": cos(S_tm, P),
                         "cos_centred_test_poolfit": cos(centred(S_pf), centred(P)),
                         "cos_centred_test_tokmean": cos(centred(S_tm), centred(P))}
        log(f"  depth {k}: shadow fitted ({out['shadow']['vocab_seen']} tokens seen, test OOV "
            f"{oov:.3f}); centred cos poolfit {out['shadow']['cos_centred_test_poolfit']:.3f} "
            f"tokmean {out['shadow']['cos_centred_test_tokmean']:.3f}")
        served = {m: {"val": {}, "test": {}} for m in ("probe", "logreg", "blocks")}
        tier0 = {m: {"val": {}, "test": {}} for m in ("same_probe_tokmean", "same_logreg_poolfit", "logreg_poolfit_rt", "tfidf")}
        gold = {"val": {}, "test": {}}
        for task in tasks:
            sel = sels[task]
            C = len(ds.tasks[task].labels)
            rec = {}
            for s in ("val", "test"):
                gold[s][task] = dict(zip(sel[s], ys(allx, sel[s], task)[0].tolist()))

            def put(store, s, probs):
                store[s][task] = {r: (int(p.argmax()), float(p.max())) for r, p in zip(sel[s], probs)}

            # served probe (real states) and the same probe on shadow tokens
            br = make("probe", k, ds.tasks[task].labels, trunk)
            tr = train_eval(br, fc, allx, sel, task, "probe", min_steps=a.min_steps, epochs=a.probe_epochs)
            T = float(br.temperature)
            for s in ("val", "test"):
                put(served["probe"], s, torch.softmax(predict_logits(br, fc, sel[s]) / T, -1).numpy())
                put(tier0["same_probe_tokmean"], s, torch.softmax(predict_logits(br, shadow_fc, sel[s]) / T, -1).numpy())
            rec["probe_real"] = tr["test"]["acc"]
            rec["probe_on_shadow"] = float(np.mean([tier0["same_probe_tokmean"]["test"][task][r][0] == gold["test"][task][r] for r in sel["test"]]))
            # pooled logistic heads
            head = logreg_head(P, allx, sel, task, C, dev, a.steps)
            head_rt = logreg_head(S_pf, allx, sel, task, C, dev, a.steps)
            for s in ("val", "test"):
                put(served["logreg"], s, head_probs(head, P[sel[s]]))
                put(tier0["same_logreg_poolfit"], s, head_probs(head, S_pf[sel[s]]))
                put(tier0["logreg_poolfit_rt"], s, head_probs(head_rt, S_pf[sel[s]]))
            acc_of = lambda st: float(np.mean([st["test"][task][r][0] == gold["test"][task][r] for r in sel["test"]]))
            rec["logreg_real"] = acc_of(served["logreg"])
            rec["same_logreg_on_poolfit"] = acc_of(tier0["same_logreg_poolfit"])
            rec["logreg_on_tokmean_pooled"] = float(np.mean(head_probs(head, S_tm[sel["test"]]).argmax(-1) ==
                                                            np.array([gold["test"][task][r] for r in sel["test"]])))
            rec["logreg_retrained_on_poolfit"] = acc_of(tier0["logreg_poolfit_rt"])
            if task not in tfidf_cache:
                tfidf_cache[task] = tfidf_probs(allx, sel, task)
            tf = tfidf_cache[task]
            for s in ("val", "test"):
                put(tier0["tfidf"], s, expand(tf[s][0], tf[s][1], C))
            rec["tfidf"] = acc_of(tier0["tfidf"])
            if blocks_here:
                bb = make("blocks:2", k, ds.tasks[task].labels, trunk)
                trb = train_eval(bb, fc, allx, sel, task, "blocks", min_steps=a.min_steps)
                Tb = float(bb.temperature)
                for s in ("val", "test"):
                    put(served["blocks"], s, torch.softmax(predict_logits(bb, fc, sel[s]) / Tb, -1).numpy())
                rec["blocks_real"] = trb["test"]["acc"]
                zb = predict_logits(bb, shadow_fc, sel["test"]).argmax(-1).numpy()
                rec["blocks_on_shadow_tokens"] = float(np.mean(zb == np.array([gold["test"][task][r] for r in sel["test"]])))
            out["tasks"][task] = rec
            log(f"   [{task}] " + " | ".join(f"{m} {v:.4f}" for m, v in rec.items()))
        # --- cascades ---
        out["cascade"] = {}
        pairs = [("probe", "same_probe_tokmean"), ("probe", "logreg_poolfit_rt"), ("probe", "tfidf"),
                 ("logreg", "same_logreg_poolfit"), ("logreg", "logreg_poolfit_rt"), ("logreg", "tfidf")]
        if blocks_here:
            pairs += [("blocks", "same_probe_tokmean"), ("blocks", "logreg_poolfit_rt"), ("blocks", "tfidf")]
        for sv, t0n in pairs:
            ev = bundle_eval(tasks, served[sv], tier0[t0n], gold, a.deltas, a.eps)
            out["cascade"][f"{sv}<-{t0n}"] = ev
            l1 = ev["ltt"][str(a.deltas[min(1, len(a.deltas) - 1)])]
            log(f"   cascade served={sv:6s} tier0={t0n:20s}: served {ev['acc_served']:.4f} tier0-only "
                f"{ev['acc_tier0_only']:.4f} | LTT delta={a.deltas[min(1, len(a.deltas) - 1)]}: exit "
                f"{l1['exit_rate']:.3f} flips {l1['flip_rate']:.4f} acc {l1['acc']:.4f} | tau=0.9: exit "
                f"{ev['fixed'][2]['exit_rate']:.3f} flips {ev['fixed'][2]['flip_rate']:.4f}")
        if a.latency:
            msgs = [allx[i].text for i in te[: a.latency_msgs]]
            head = logreg_head(P, allx, sels[tasks[0]], tasks[0], len(ds.tasks[tasks[0]].labels), dev, a.steps)
            out["latency_cpu"] = latency(msgs, k, E_tok, head)
            L = out["latency_cpu"]
            log(f"   CPU latency (batch 1, {L['threads']} threads, {L['mean_tokens']:.0f} tokens): trunk to depth {k} "
                f"{L['trunk_to_depth_ms_median']:.2f} ms vs tier 0 {L['tier0_ms_median']:.4f} ms")
        res["depths"][k] = out
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150", "typed-decisions"])
    ap.add_argument("--depths", type=int, nargs="*", default=None, help="override the per-dataset depths")
    ap.add_argument("--blocks", action="store_true", help="also serve blocks:2 branches")
    ap.add_argument("--blocks-datasets", nargs="*", default=["banking77", "clinc150"])
    ap.add_argument("--deltas", type=float, nargs="*", default=[0.005, 0.01, 0.02])
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=300, help="logistic-head optimiser steps (>= 300)")
    ap.add_argument("--probe-epochs", type=int, default=8)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--latency", action="store_true", help="CPU batch-1 latency of trunk-to-k vs tier 0")
    ap.add_argument("--latency-msgs", type=int, default=30)
    ap.add_argument("--latency-only", action="store_true",
                    help="CPU latency only, with random shadow vectors and head (cost does not depend on the fit)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.min_steps, a.steps, a.probe_epochs, a.latency_msgs = "cpu", 20, 30, 1, 3
        a.blocks, a.latency = True, True
        a.out = a.out or "results/tarski/explore_lit_shadow_smoke.json"
    if not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    if a.latency_only:
        res = {"args": vars(a), "latency_cpu": {}}
        for name in a.datasets:
            ds = load_ds(name)
            msgs = [e.text for e in ds.test[: a.latency_msgs]]
            for k in (a.depths or DEFAULT_DEPTHS[name]):
                C = len(next(iter(ds.tasks.values())).labels)
                head = (torch.zeros(1, 768), torch.ones(1, 768), torch.randn(768, C), torch.zeros(1, C))
                L = latency(msgs, k, torch.randn(50368, 768), head)
                res["latency_cpu"][f"{name}:{k}"] = L
                log(f"  {name} depth {k}: CPU batch 1 ({L['threads']} threads, {L['mean_tokens']:.0f} tokens): trunk "
                    f"{L['trunk_to_depth_ms_median']:.2f} ms vs tier 0 {L['tier0_ms_median']:.4f} ms ({L['speedup']:.0f}x)")
                save(res, a.out)
        return
    trunk = Trunk(device=a.device)
    dev = trunk.device
    res = {"args": vars(a), "datasets": {}}
    for name in a.datasets:
        res["datasets"][name] = run_dataset(name, trunk, a, log, dev)
        save(res, a.out)
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
