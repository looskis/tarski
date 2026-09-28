"""Closed-form geometry scan of the frozen trunk: what each depth makes easy, from one trunk pass.

One pass over every message taps all 22 depths and keeps, per message and depth, the mean-pooled state,
the [CLS] state and a *token-identity-centred* mean pool (below). Everything after that is closed form
(ridge classifiers, Gaussian class statistics), so the scan costs about one trunk pass per dataset.

It tests four hypotheses from research_notes/explore/geometry.md:

  centring    Mean pooling a long JSON state is dominated by what every state shares (keys, braces,
              punctuation, the same field names in the same order). Subtracting from each token state the
              mean state of its token id (estimated on training messages, no labels) leaves the part of
              the state that depends on this message's context:  pool_c(x) = mean_t (h_t - mu[id_t]).
              mu is shrunk towards the global mean token state by --shrink pseudo-counts, so rare
              (content) tokens are not centred on themselves. Because mean_t mu[id_t] depends only on the
              token ids, this is one subtraction per message at inference.
  global/local ModernBERT runs a global-attention layer every third layer (layers 0,3,...,21) and 128-token
              sliding windows otherwise. Do probe curves show a period-3 sawtooth, i.e. are depths right
              after a global layer better split points?
  OOD         Out-of-scope detection on CLINC150 with no out-of-scope training examples: class-conditional
              Mahalanobis distance (shared covariance) to the 150 in-scope intents at every depth, a
              multi-depth sum, and cross-depth disagreement of nearest-class predictions, against a
              supervised ridge oos probe.
  transfer    Do decisions that mean the same thing in different schemas share a direction? A ridge
              regression on the expected urgency (soft labels) of one typed-decisions workflow is applied
              to the other workflows' states (Spearman correlation with their expected urgency).

Probes are ridge classifiers on standardized features (one eigendecomposition per depth serves every task
and every L2 strength, which is chosen on validation accuracy). The tarski probe results in
results/tarski/sweep_*.json are the reference for "probes collapse on typed-decisions".

Usage (from the repo root):
  .venv/bin/python explore/geometry_depthscan.py --smoke
  .venv/bin/python explore/geometry_depthscan.py --out results/tarski/explore_depthscan.json   # ~5-8 min on the M6 GPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from tarski import data as tdata
from tarski import train as ttrain
from tarski.trunk import Trunk


# ---------------------------------------------------------------------------------------------------
# One trunk pass: pooled, [CLS] and token-identity-centred pooled states at every depth
# ---------------------------------------------------------------------------------------------------

@torch.no_grad()
def scan_features(trunk: Trunk, texts: Sequence[str], train_mask: np.ndarray, depths: List[int],
                  center_depths: List[int], max_len: int, bs: int = 64, shrink: float = 20.0) -> Dict:
    dev, H = trunk.device, trunk.hidden
    ids = trunk.token_ids(texts, max_len)
    vocab = sorted({t for i, x in enumerate(ids) if train_mask[i] for t in x})
    vmap = torch.full((max(trunk.cfg.vocab_size, len(trunk.tok)) + 1,), -1, dtype=torch.long)
    vmap[torch.tensor(vocab)] = torch.arange(len(vocab))
    vmap = vmap.to(dev)
    N = len(texts)
    mean = {d: torch.zeros(N, H, dtype=torch.float16) for d in depths}     # fp16 on CPU bounds RAM (CLINC: ~1.6 GB)
    cls = {d: torch.zeros(N, H, dtype=torch.float16) for d in depths}
    tsum = {d: torch.zeros(len(vocab), H, device=dev) for d in center_depths}
    tcnt = torch.zeros(len(vocab), device=dev)
    order = sorted(range(N), key=lambda i: len(ids[i]))
    t0 = time.time()
    for s in range(0, N, bs):
        idx = order[s:s + bs]
        L = max(len(ids[i]) for i in idx)
        x = torch.full((len(idx), L), trunk.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            x[j, : len(ids[i])] = torch.tensor(ids[i])
            att[j, : len(ids[i])] = 1
        x, att = x.to(dev), att.to(dev)
        with ttrain.autocast(dev):
            taps, _ = trunk.taps(x, att, depths)
        m = att.float()[..., None]
        tr = torch.tensor([bool(train_mask[i]) for i in idx], device=dev)
        sel = att.bool() & tr[:, None]
        rows = vmap[x[sel]]
        if sel.any():
            tcnt.index_add_(0, rows, torch.ones(len(rows), device=dev))
        for d in depths:
            h = taps[d].float()
            mean[d][idx] = ((h * m).sum(1) / m.sum(1)).half().cpu()
            cls[d][idx] = h[:, 0].half().cpu()
            if d in tsum and sel.any():
                tsum[d].index_add_(0, rows, h[sel])
    secs = time.time() - t0
    # centred pool = mean pool - mean over the message's tokens of mu[token id]. mu is shrunk towards the
    # global mean token state g by `shrink` pseudo-counts, so frequent ids (keys, braces, punctuation) are
    # centred on their own mean while rare, content-bearing ids are centred on g and keep their signal;
    # ids never seen in training messages use g.
    centred = {}
    for d in center_depths:
        g = (tsum[d].sum(0) / tcnt.sum())[None]
        mu = ((tsum[d] + shrink * g) / (tcnt[:, None] + shrink)).cpu()
        mu = torch.cat([mu, g.cpu()])                                    # row -1 -> g
        vm = vmap.cpu()
        sub = torch.stack([mu[vm[torch.tensor(x)]].mean(0) for x in ids])
        centred[d] = (mean[d].float() - sub).half()
    return {"mean": mean, "cls": cls, "centred": centred, "seconds": secs, "vocab": len(vocab),
            "lengths": [len(x) for x in ids]}


# ---------------------------------------------------------------------------------------------------
# Closed-form probes
# ---------------------------------------------------------------------------------------------------

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)


class Ridge:
    """Standardized ridge regression; one eigendecomposition per feature matrix serves every target."""

    def __init__(self, Xtr: torch.Tensor):
        X = Xtr.double()
        self.mu, self.sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
        self.X = (X - self.mu) / self.sd
        s, V = torch.linalg.eigh(self.X.T @ self.X / len(X))
        self.s, self.V = s.clamp_min(0), V

    def fit_predict(self, T: torch.Tensor, others: Sequence[torch.Tensor], lam: float) -> List[torch.Tensor]:
        T = T.double()
        tm = T.mean(0, keepdim=True)
        W = self.V @ ((self.V.T @ (self.X.T @ (T - tm) / len(T))) / (self.s + lam)[:, None])
        return [((Z.double() - self.mu) / self.sd) @ W + tm for Z in others]


def probe(F: torch.Tensor, tr, va, te, Ttr: torch.Tensor, yva: np.ndarray, yte: np.ndarray, rid=None) -> Dict:
    rid = rid or Ridge(F[tr])
    best = None
    for lam in LAMBDAS:
        zv, zt = rid.fit_predict(Ttr, [F[va], F[te]], lam)
        acc_v = float((zv.argmax(-1).numpy() == yva).mean())
        if best is None or acc_v > best["val_acc"]:
            best = {"val_acc": acc_v, "acc": float((zt.argmax(-1).numpy() == yte).mean()), "l2": lam}
    return best


def task_arrays(ds: tdata.Dataset, allx, task: str, n_tr: int, n_va: int):
    idx = [i for i, e in enumerate(allx) if task in e.y]
    tr = [i for i in idx if i < n_tr]
    va = [i for i in idx if n_tr <= i < n_tr + n_va]
    te = [i for i in idx if i >= n_tr + n_va]
    C = len(ds.tasks[task].labels)
    if all(task in allx[i].soft for i in tr):
        T = torch.tensor(np.stack([allx[i].soft[task] for i in tr]))
    else:
        T = torch.nn.functional.one_hot(torch.tensor([allx[i].y[task] for i in tr]), C).double()
    y = lambda s: np.array([allx[i].y[task] for i in s])
    return tr, va, te, T, y(va), y(te)


def is_after_global(cfg, d: int) -> bool:
    return d >= 1 and cfg.layer_types[d - 1] == "full_attention"


def sawtooth(curve: Dict[int, float], cfg) -> Dict[str, float]:
    """Mean accuracy and mean local bump acc(d) - (acc(d-1)+acc(d+1))/2 for depths right after a global
    layer vs after a local layer."""
    out = {}
    for name, want in (("after_global", True), ("after_local", False)):
        ds_ = [d for d in curve if is_after_global(cfg, d) == want]
        bumps = [curve[d] - (curve[d - 1] + curve[d + 1]) / 2 for d in ds_ if d - 1 in curve and d + 1 in curve]
        out[f"{name}_mean_acc"] = float(np.mean([curve[d] for d in ds_]))
        out[f"{name}_mean_bump"] = float(np.mean(bumps)) if bumps else float("nan")
    return out


# ---------------------------------------------------------------------------------------------------
# OOD on CLINC150 without out-of-scope training data
# ---------------------------------------------------------------------------------------------------

def gaussian_stats(X: torch.Tensor, y: np.ndarray, n_cls: int, eps: float = 1e-3):
    X = X.double()
    mu = torch.stack([X[torch.tensor(y == c)].mean(0) for c in range(n_cls)])
    R = X - mu[torch.tensor(y)]
    cov = R.T @ R / len(X)
    cov = cov + eps * cov.diagonal().mean() * torch.eye(X.shape[1], dtype=cov.dtype)
    return mu, torch.linalg.inv(cov)


def maha(X: torch.Tensor, mu: torch.Tensor, P: torch.Tensor):
    X = X.double()
    XP = X @ P
    d = (XP * X).sum(-1, keepdim=True) - 2 * XP @ mu.T + ((mu @ P) * mu).sum(-1)[None]
    m, c = d.min(-1)
    return m.numpy(), c.numpy()


def ood_metrics(score_in: np.ndarray, score_out: np.ndarray, val_in: np.ndarray, val_out: np.ndarray) -> Dict:
    """score: higher = more out-of-scope. AUROC, fraction of oos missed when 95% of in-scope is kept, and
    oos-binary test accuracy at the threshold that maximises validation accuracy."""
    y = np.r_[np.zeros(len(score_in)), np.ones(len(score_out))]
    s = np.r_[score_in, score_out]
    thr95 = np.quantile(score_in, 0.95)
    cand = np.unique(np.r_[val_in, val_out])
    yv = np.r_[np.zeros(len(val_in)), np.ones(len(val_out))]
    sv = np.r_[val_in, val_out]
    accs = [((sv > t) == yv).mean() for t in cand]
    t = cand[int(np.argmax(accs))]
    return {"auroc": float(roc_auc_score(y, s)), "oos_missed_at_95_inscope": float((score_out <= thr95).mean()),
            "binary_acc_valthr": float(((s > t) == y).mean()), "majority_acc": float(1 - y.mean())}


def ood_clinc(ds: tdata.Dataset, feats: Dict, depths: List[int], log) -> Dict:
    allx = ds.train + ds.val + ds.test
    n_tr, n_va = len(ds.train), len(ds.val)
    names = ds.tasks["intent"].labels
    oos_id = names.index("oos")
    y = np.array([e.y["intent"] for e in allx])
    tr_in = [i for i in range(n_tr) if y[i] != oos_id]
    va = list(range(n_tr, n_tr + n_va))
    te = list(range(n_tr + n_va, len(allx)))
    ins = sorted(set(y[tr_in]))
    remap = {c: j for j, c in enumerate(ins)}
    ytr = np.array([remap[c] for c in y[tr_in]])
    split = lambda idx: ([i for i in idx if y[i] != oos_id], [i for i in idx if y[i] == oos_id])
    va_in, va_out = split(va)
    te_in, te_out = split(te)
    out = {"per_depth": {}, "n_test_in": len(te_in), "n_test_oos": len(te_out)}
    zs_val, zs_test, preds = {}, {}, {}
    for d in depths:
        X = feats["mean"][d].double()
        m_, s_ = X[tr_in].mean(0, keepdim=True), X[tr_in].std(0, keepdim=True).clamp_min(1e-6)
        X = (X - m_) / s_                       # standardize first: a few massive dimensions dominate raw scale
        mu, P = gaussian_stats(X[tr_in], ytr, len(ins))
        s_all, c_all = maha(X, mu, P)
        m = ood_metrics(s_all[te_in], s_all[te_out], s_all[va_in], s_all[va_out])
        acc_ncm = float((np.array([ins[c] for c in c_all[te_in]]) == y[te_in]).mean())
        m["in_scope_ncm_acc"] = acc_ncm
        out["per_depth"][d] = m
        ref = s_all[va_in]
        z = (s_all - ref.mean()) / ref.std()
        zs_val[d], zs_test[d] = z, z
        preds[d] = c_all
    best = max(depths, key=lambda d: ood_metrics(zs_val[d][va_in], zs_val[d][va_out], zs_val[d][va_in], zs_val[d][va_out])["auroc"])
    out["best_depth_by_val_auroc"] = best
    out["best_depth"] = out["per_depth"][best]
    s = sum(zs_test[d] for d in depths) / len(depths)
    out["multi_depth_mean_z"] = ood_metrics(s[te_in], s[te_out], s[va_in], s[va_out])
    top = [d for d in depths if d >= 6]
    s = sum(zs_test[d] for d in top) / len(top)
    out["multi_depth_mean_z_ge6"] = ood_metrics(s[te_in], s[te_out], s[va_in], s[va_out])
    # cross-depth disagreement: how many depths' nearest-class predictions differ from the modal one
    P_ = np.stack([preds[d] for d in top], 1)
    mode = np.array([np.bincount(r).argmax() for r in P_])
    dis = (P_ != mode[:, None]).mean(1) + 1e-3 * s          # tie-break by the mean Mahalanobis z
    out["cross_depth_disagreement"] = ood_metrics(dis[te_in], dis[te_out], dis[va_in], dis[va_out])
    # supervised reference: ridge on the oos binary label (uses the 100 oos training examples)
    tr_all = list(range(n_tr))
    yb = (y == oos_id).astype(int)
    Xb = feats["mean"][best]
    rid = Ridge(Xb[tr_all])
    T = torch.nn.functional.one_hot(torch.tensor(yb[tr_all]), 2).double()
    zv, zt = rid.fit_predict(T, [Xb[va], Xb[te]], 1.0)
    sv, st = (zv[:, 1] - zv[:, 0]).numpy(), (zt[:, 1] - zt[:, 0]).numpy()
    va_pos = {i: j for j, i in enumerate(va)}
    te_pos = {i: j for j, i in enumerate(te)}
    out["supervised_ridge_oos_probe"] = ood_metrics(st[[te_pos[i] for i in te_in]], st[[te_pos[i] for i in te_out]],
                                                    sv[[va_pos[i] for i in va_in]], sv[[va_pos[i] for i in va_out]])
    log("  OOD best single depth %d: %s" % (best, json.dumps({k: round(v, 4) for k, v in out["best_depth"].items()})))
    for k in ("multi_depth_mean_z", "multi_depth_mean_z_ge6", "cross_depth_disagreement", "supervised_ridge_oos_probe"):
        log(f"  OOD {k}: " + json.dumps({a: round(b, 4) for a, b in out[k].items()}))
    return out


# ---------------------------------------------------------------------------------------------------
# Cross-schema transfer of a shared decision (typed-decisions urgency)
# ---------------------------------------------------------------------------------------------------

def urgency_transfer(ds: tdata.Dataset, feats: Dict, depths: List[int], pool: str) -> Dict:
    allx = ds.train + ds.val + ds.test
    n_tr, n_va = len(ds.train), len(ds.val)
    tasks = [t for t in ds.tasks if t.endswith(".urgency")]
    out = {}
    for d in depths:
        X = feats[pool][d]
        tab = {}
        for a in tasks:
            tr = [i for i, e in enumerate(allx[:n_tr]) if a in e.y]
            ya = torch.tensor([float(np.dot(allx[i].soft[a], np.arange(len(allx[i].soft[a])))) for i in tr])[:, None]
            rid = Ridge(X[tr])
            for b in tasks:
                te = [i for i in range(n_tr + n_va, len(allx)) if b in allx[i].y]
                yb = np.array([float(np.dot(allx[i].soft[b], np.arange(len(allx[i].soft[b])))) for i in te])
                (z,) = rid.fit_predict(ya, [X[te]], 10.0)
                tab[f"{a.split('.')[0]}->{b.split('.')[0]}"] = float(spearmanr(z[:, 0].numpy(), yb).correlation)
        diag = [v for k, v in tab.items() if k.split("->")[0] == k.split("->")[1]]
        off = [v for k, v in tab.items() if k.split("->")[0] != k.split("->")[1]]
        out[d] = {"pairs": tab, "within_mean": float(np.mean(diag)), "across_mean": float(np.mean(off))}
    return out


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------

def run_dataset(trunk: Trunk, ds: tdata.Dataset, depths: List[int], center_depths: List[int], log,
                shrink: float = 20.0) -> Dict:
    allx = ds.train + ds.val + ds.test
    n_tr, n_va = len(ds.train), len(ds.val)
    train_mask = np.array([i < n_tr for i in range(len(allx))])
    feats = scan_features(trunk, [e.text for e in allx], train_mask, depths, center_depths, ds.max_len,
                          shrink=shrink)
    log(f"  one trunk pass over {len(allx)} messages, {len(depths)} depths: {feats['seconds']:.1f}s "
        f"(train vocab {feats['vocab']} token ids, median {np.median(feats['lengths']):.0f} tokens)")
    res = {"trunk_pass_s": feats["seconds"], "curves": {}, "summary": {}}
    ridges = {}
    for pool in ("mean", "cls", "centred"):
        pdepths = center_depths if pool == "centred" else depths
        for d in pdepths:
            X = feats[pool][d]
            for t in ds.tasks:
                tr, va, te, T, yva, yte = task_arrays(ds, allx, t, n_tr, n_va)
                key = (pool, d, tuple(tr))
                if key not in ridges:
                    ridges = {k: v for k, v in ridges.items() if k[:2] == (pool, d)}   # bound memory
                    ridges[key] = Ridge(X[tr])
                r = probe(X, tr, va, te, T, yva, yte, ridges[key])
                res["curves"].setdefault(pool, {}).setdefault(t, {})[d] = r
    # multi-tap: concatenated mean pools at a few depths
    taps = [d for d in (6, 11, 16, 22) if d in depths]
    Xc = torch.cat([feats["mean"][d] for d in taps], -1)
    res["curves"]["multitap_" + "_".join(map(str, taps))] = {}
    for t in ds.tasks:
        tr, va, te, T, yva, yte = task_arrays(ds, allx, t, n_tr, n_va)
        res["curves"]["multitap_" + "_".join(map(str, taps))][t] = {0: probe(Xc, tr, va, te, T, yva, yte)}
    # summaries: mean over tasks
    cfg = trunk.cfg
    for pool, per_task in res["curves"].items():
        ds_ = sorted({d for c in per_task.values() for d in c})
        mean_curve = {d: float(np.mean([per_task[t][d]["acc"] for t in per_task])) for d in ds_}
        val_best = {t: max(c, key=lambda d: c[d]["val_acc"]) for t, c in per_task.items()}
        summ = {"mean_acc_by_depth": mean_curve,
                "mean_acc_at_val_best_depth": float(np.mean([per_task[t][val_best[t]]["acc"] for t in per_task])),
                "val_best_depths": val_best}
        if pool in ("mean", "cls") and len(ds_) > 3:
            summ["sawtooth"] = sawtooth(mean_curve, cfg)
        res["summary"][pool] = summ
        log(f"  [{pool}] mean acc at val-best depth {summ['mean_acc_at_val_best_depth']:.4f}; by depth " +
            " ".join(f"{d}:{a:.3f}" for d, a in mean_curve.items()))
        if "sawtooth" in summ:
            log(f"  [{pool}] sawtooth: " + json.dumps({k: round(v, 4) for k, v in summ["sawtooth"].items()}))
    if "centred" in res["summary"]:
        c, m = res["summary"]["centred"]["mean_acc_by_depth"], res["summary"]["mean"]["mean_acc_by_depth"]
        res["summary"]["centred_minus_mean"] = {d: c[d] - m[d] for d in c}
        log("  centred - mean by depth: " + " ".join(f"{d}:{v:+.3f}" for d, v in res["summary"]["centred_minus_mean"].items()))
    if ds.name == "clinc150":
        res["ood"] = ood_clinc(ds, feats, depths, log)
    if ds.name == "typed-decisions":
        res["urgency_transfer"] = {pool: urgency_transfer(ds, feats, [d for d in (11, 22) if d in center_depths], pool)
                                   for pool in ("mean", "centred")}
        for pool, v in res["urgency_transfer"].items():
            for d, r in v.items():
                log(f"  urgency transfer [{pool}@{d}] within {r['within_mean']:.3f} across {r['across_mean']:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=["typed-decisions", "clinc150", "banking77"])
    ap.add_argument("--center-depths", type=int, nargs="*", default=[3, 6, 9, 11, 12, 16, 22])
    ap.add_argument("--shrink", type=float, default=20.0, help="pseudo-counts shrinking token-id means to the global mean")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = args.out or ("results/tarski/explore_depthscan_smoke.json" if args.smoke
                            else "results/tarski/explore_depthscan.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    logf = open(out_path.replace(".json", ".log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    trunk = Trunk(device="cpu" if args.smoke else None)
    depths = list(range(1, trunk.n_layers + 1))
    log(f"== depth scan | base {trunk.base} on {trunk.device} | smoke={args.smoke} | layer types "
        + "".join("G" if t == "full_attention" else "l" for t in trunk.cfg.layer_types))
    results = {"args": vars(args), "device": str(trunk.device), "datasets": {}}
    for name in args.datasets:
        ds = tdata.load(name)
        if args.smoke:
            if name == "clinc150":
                rng = np.random.default_rng(0)
                pick = lambda xs, k: [xs[i] for i in sorted(rng.choice(len(xs), k, replace=False))]
                inscope = lambda xs: [e for e in xs if e.y["oos"] == 0]
                oos = lambda xs: [e for e in xs if e.y["oos"] == 1]
                ds.train = pick(inscope(ds.train), 300) + pick(oos(ds.train), 20)
                ds.val = pick(inscope(ds.val), 60) + pick(oos(ds.val), 20)
                ds.test = pick(inscope(ds.test), 60) + pick(oos(ds.test), 20)
            else:
                rng = np.random.default_rng(0)
                pick = lambda xs, k: [xs[i] for i in sorted(rng.choice(len(xs), min(k, len(xs)), replace=False))]
                ds.train, ds.val, ds.test = pick(ds.train, 120), pick(ds.val, 60), pick(ds.test, 80)
            ds.max_len = min(ds.max_len, 128)
        t0 = time.time()
        log(f"-- {ds.summary()}")
        r = run_dataset(trunk, ds, depths, [d for d in args.center_depths if d in depths], log, args.shrink)
        r["wall_s"] = round(time.time() - t0, 1)
        results["datasets"][name] = r
        json.dump(results, open(out_path, "w"), indent=1)
    log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
