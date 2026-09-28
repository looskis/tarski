"""Idea 6: speculate from a decision cache, verify with a shallow slice of the trunk.

Every processed message leaves a cache entry: its mean-pooled trunk state at a shallow depth v and its
whole decision vector (all N branch outputs). A new message runs the trunk only to depth v and finds its
nearest cached neighbour; if the similarity clears the threshold it reuses the neighbour's decisions
(a hit: the rest of the trunk and every branch are skipped). Otherwise it continues the pass (nothing
is wasted) and is inserted. Variants:

  shallow@v            nearest neighbour by cosine of the pooled depth-v state (v in 2, 4, 6)
  lexical              nearest neighbour by token-set overlap (Ochiai, a SimHash-style key, no trunk at all)
  lexical+verify@v     lexical candidate, accepted only if its depth-v cosine is also >= tau

Stream: the training messages form the initial cache (decisions = the branches' own outputs on them),
then the test set arrives in random order. Reported: hit rate, overall accuracy (hits use the reused
decisions), accuracy on hits vs the branches' own accuracy on those messages, and compute saved
(hit rate x (1 - v / (k + branch layers))).

Tested prediction: a knee where 20-40% of messages hit at < 0.3 points accuracy loss.

Usage:
  python explore/systems_speccache.py --smoke
  python explore/systems_speccache.py --out results/tarski/explore_speccache.json
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explore.systems_common import Log, all_texts, save, split_index, subset, threads, train_tasks

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from tarski import data
from tarski.autosplit import pooled_by_depth
from tarski.train import FeatureCache, predict_logits
from tarski.trunk import Trunk


def dense_sims(e_new: torch.Tensor, e_hist: torch.Tensor, chunk: int = 2048):
    """Best/argmax similarity of each new row to the history, and the new x new similarity matrix."""
    best, arg = [], []
    for s in range(0, len(e_new), chunk):
        m, a = (e_new[s:s + chunk] @ e_hist.T).max(-1)
        best.append(m.cpu())
        arg.append(a.cpu())
    return torch.cat(best).numpy(), torch.cat(arg).numpy(), (e_new @ e_new.T).float().cpu().numpy()


def lexical(trunk: Trunk, texts: List[str]) -> sp.csr_matrix:
    """Binary bag of token ids, rows L2-normalised: row . row = |A & B| / sqrt(|A||B|)."""
    ids = [sorted(set(x)) for x in trunk.token_ids(texts, 128)]
    rows = np.concatenate([np.full(len(x), i) for i, x in enumerate(ids)])
    cols = np.concatenate([np.array(x) for x in ids])
    vals = np.concatenate([np.full(len(x), 1 / np.sqrt(len(x))) for x in ids])
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(ids), len(trunk.tok) + 8), dtype=np.float32)


def stream(order: np.ndarray, h_best, h_arg, S_new, theta: float,
           verify: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None):
    """Sequential cache. Returns hit mask and source per message (-1-j: history row j; j >= 0: test row)."""
    n = len(order)
    hit, src = np.zeros(n, bool), np.zeros(n, int)
    inserted: List[int] = []
    for i in order:
        b, j = float(h_best[i]), -1 - int(h_arg[i])
        if inserted:
            s = S_new[i, inserted]
            a = int(s.argmax())
            if s[a] > b:
                b, j = float(s[a]), inserted[a]
        ok = b >= theta
        if ok and verify is not None:
            e_hist, e_new, tau = verify
            other = e_hist[-1 - j] if j < 0 else e_new[j]
            ok = float(other @ e_new[i]) >= tau
        if ok:
            hit[i], src[i] = True, j
        else:
            inserted.append(i)
    return hit, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["clinc150", "banking77"])
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--kind", default="probe", help="branch kind giving the decisions (probe | blocks:D)")
    ap.add_argument("--vs", type=int, nargs="*", default=[2, 4, 6])
    ap.add_argument("--taus", type=float, nargs="*", default=[0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98])
    ap.add_argument("--verify-taus", type=float, nargs="*", default=[0.5, 0.7, 0.8, 0.9])
    ap.add_argument("--thetas", type=float, nargs="*", default=[0.6, 0.7, 0.8, 0.9, 0.999])
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--no-center", dest="center", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.split, a.vs, a.min_steps, a.epochs, a.datasets = "cpu", 4, [2], 5, 1, ["clinc150"]
        a.taus, a.thetas, a.verify_taus = [0.5, 0.9], [0.7, 0.999], [0.5]
    if not a.smoke and not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    trunk = Trunk(device=a.device)
    res = {"split": a.split, "kind": a.kind, "sets": {}}
    for name in a.datasets:
        ds = data.load(name)
        if a.smoke:
            ds = subset(ds, list(ds.tasks), 200, 40, 60, stride=23)
        tasks = list(ds.tasks)
        texts = all_texts(ds)
        allx, rng = split_index(ds)
        cache = FeatureCache(trunk, texts, [a.split], ds.max_len)
        br = train_tasks(trunk, ds, cache, tasks, a.kind, a.split, seed=a.seed, epochs=a.epochs,
                         min_steps=a.min_steps, log=log)
        tr_idx, te_idx = list(rng["train"]), list(rng["test"])
        dec_hist = {t: predict_logits(br[t]["branch"], cache, tr_idx).argmax(-1).numpy() for t in tasks}
        dec_new = {t: predict_logits(br[t]["branch"], cache, te_idx).argmax(-1).numpy() for t in tasks}
        del cache
        y = {t: np.array([allx[i].y[t] for i in te_idx]) for t in tasks}
        own_acc = float(np.mean([(dec_new[t] == y[t]).mean() for t in tasks]))
        order = np.random.default_rng(a.seed).permutation(len(te_idx))
        branch_layers = 0 if a.kind == "probe" else int(a.kind.split(":")[1]) * len(tasks)
        full_cost = a.split + branch_layers
        norm = lambda s: " ".join(s.lower().split())
        exact = len({norm(allx[i].text) for i in te_idx} & {norm(allx[i].text) for i in tr_idx})
        entry = {"own_acc": own_acc, "n_test": len(te_idx), "exact_test_texts_in_train": exact, "settings": {}}
        log(f"== {ds.summary()} | decisions from {a.kind}@{a.split}: acc {own_acc:.4f} | "
            f"exact test texts also in train: {exact}")

        def record(key, hit, src, v):
            reused = {t: dec_new[t].copy() for t in tasks}
            for i in np.nonzero(hit)[0]:
                j = src[i]
                for t in tasks:
                    reused[t][i] = dec_hist[t][-1 - j] if j < 0 else dec_new[t][j]
            acc = float(np.mean([(reused[t] == y[t]).mean() for t in tasks]))
            h = hit
            hit_acc = float(np.mean([(reused[t][h] == y[t][h]).mean() for t in tasks])) if h.any() else None
            own_hits = float(np.mean([(dec_new[t][h] == y[t][h]).mean() for t in tasks])) if h.any() else None
            agree = float(np.mean([(reused[t][h] == dec_new[t][h]).mean() for t in tasks])) if h.any() else None
            saved = float(h.mean() * (1 - v / full_cost))
            entry["settings"][key] = {"hit_rate": float(h.mean()), "acc": acc, "acc_delta": acc - own_acc,
                                      "acc_on_hits": hit_acc, "own_acc_on_hits": own_hits,
                                      "agree_on_hits": agree, "compute_saved": saved}
            fmt = lambda x: "-" if x is None else f"{x:.4f}"
            log(f"   {key:28s} hit {h.mean():.3f} acc {acc:.4f} ({acc - own_acc:+.4f}) | on hits reused "
                f"{fmt(hit_acc)} vs own {fmt(own_hits)} agree {fmt(agree)} | compute saved {saved:.3f}")

        pooled = pooled_by_depth(trunk, texts, a.vs, ds.max_len)
        embs = {}
        for v in a.vs:
            x = pooled[v]
            if a.center:                          # pooled states are anisotropic: centre on the history mean
                x = x - x[tr_idx].mean(0, keepdim=True)
            e = F.normalize(x, dim=-1).to(trunk.device)
            e_hist, e_new = e[tr_idx], e[te_idx]
            embs[v] = (e_hist.float().cpu().numpy(), e_new.float().cpu().numpy())
            hb, ha, S = dense_sims(e_new, e_hist)
            for tau in a.taus:
                hit, src = stream(order, hb, ha, S, tau)
                record(f"shallow@{v} tau={tau}", hit, src, v)
        lex = lexical(trunk, texts)
        L_hist, L_new = lex[tr_idx], lex[te_idx]
        LH = (L_new @ L_hist.T).tocsr()
        hb = np.asarray(LH.max(axis=1).todense()).ravel()
        ha = np.asarray(LH.argmax(axis=1)).ravel()
        S = (L_new @ L_new.T).toarray()
        v = min(a.vs)
        for th in a.thetas:
            hit, src = stream(order, hb, ha, S, th)
            record(f"lexical theta={th}", hit, src, 0)
            for tau in a.verify_taus:
                hit, src = stream(order, hb, ha, S, th, verify=(embs[v][0], embs[v][1], tau))
                record(f"lexical {th}+verify@{v} tau={tau}", hit, src, v)
        res["sets"][name] = entry
        save(res, a.out)
    log("== summary: largest compute saving with accuracy within 0.3 points of the branches")
    for name, e in res["sets"].items():
        ok = [(k_, s) for k_, s in e["settings"].items() if s["acc_delta"] >= -0.003]
        for k_, s in sorted(ok, key=lambda kv: -kv[1]["compute_saved"])[:4]:
            log(f"  {name:10s} {k_:28s} hit {s['hit_rate']:.3f} acc Δ {s['acc_delta']:+.4f} saved {s['compute_saved']:.3f}")
    save(res, a.out)
    log("done")


if __name__ == "__main__":
    main()
