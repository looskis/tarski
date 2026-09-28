"""Far-field idea 1: syndrome decoding of decision bundles (coding theory / danger theory) on CLINC150.

Several branches answer several decisions about one message. The decisions are redundant (domain is a
function of intent), so a bundle of branch outputs is a noisy received word from a code whose valid
codewords are the consistent bundles. Two consequences are tested here, using only in-scope training data:

  * syndrome as an out-of-scope score: disagreement between the domain branch and the domain implied by
    the intent branch (JS divergence, agreement probability, joint-decoded evidence);
  * consistent joint decoding: argmax_i log P_int(i) + lam * log P_dom(dom(i)) instead of separate argmaxes.

Baselines computed in the same run: max-softmax and entropy of the intent probe, the supervised OOS
binary probe and the 151-way probe's P(oos) (both use OOS training labels, the references), kNN distance
and Mahalanobis distance (idea 3, negative selection, reduces to these), depth-disagreement (idea 4), and
rank/z-score combinations. All heads are linear probes on mean-pooled trunk states at several depths, so the
whole study is one trunk pass plus seconds of full-batch training per probe.

Usage:
  .venv/bin/python explore/farfield_syndrome.py --smoke
  .venv/bin/python explore/farfield_syndrome.py --out results/tarski/explore_syndrome.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, so `tarski` imports

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from tarski import data
from tarski.autosplit import pooled_by_depth
from tarski.train import fit_temperature
from tarski.trunk import Trunk

EPS = 1e-12


# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------

def log(msg: str, f=None):
    print(msg, flush=True)
    if f is not None:
        f.write(msg + "\n")
        f.flush()


def fit_linear(x: torch.Tensor, y: torch.Tensor, n_labels: int, device, steps: int = 300, lr: float = 1e-2,
               wd: float = 1e-4, seed: int = 0) -> torch.nn.Linear:
    """Full-batch linear probe on standardised features (the autosplit recipe, returning the model)."""
    torch.manual_seed(seed)
    lin = torch.nn.Linear(x.shape[1], n_labels).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    x, y = x.to(device), y.to(device)
    for _ in range(steps):
        loss = F.cross_entropy(lin(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lin.eval()


@torch.no_grad()
def logits_of(lin: torch.nn.Linear, x: torch.Tensor, device) -> torch.Tensor:
    return lin(x.to(device)).float().cpu()


def softmax_np(z: torch.Tensor, t: float = 1.0) -> np.ndarray:
    return torch.softmax(z / t, -1).numpy().astype(np.float64)


def js_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence per row (natural log)."""
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p + EPS) - np.log(m + EPS))).sum(-1)
    kl_qm = (q * (np.log(q + EPS) - np.log(m + EPS))).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(p + EPS)).sum(-1)


def ood_metrics(score: np.ndarray, is_oos: np.ndarray) -> Dict[str, float]:
    """Higher score = more out-of-scope. OOS is the positive class."""
    auroc = float(roc_auc_score(is_oos, score))
    aupr = float(average_precision_score(is_oos, score))
    fpr, tpr, _ = roc_curve(is_oos, score)
    fpr95 = float(fpr[np.searchsorted(tpr, 0.95, side="left")]) if (tpr >= 0.95).any() else 1.0
    return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}


def best_threshold_acc(score_val: np.ndarray, oos_val: np.ndarray, score_test: np.ndarray,
                       oos_test: np.ndarray) -> Dict[str, float]:
    """Binary OOS accuracy at the threshold that maximises validation accuracy (the sweep's oos task)."""
    cands = np.unique(np.quantile(score_val, np.linspace(0, 1, 401)))
    accs = [((score_val > c).astype(int) == oos_val).mean() for c in cands]
    thr = float(cands[int(np.argmax(accs))])
    pred = (score_test > thr).astype(int)
    tp = int(((pred == 1) & (oos_test == 1)).sum())
    return {"threshold": thr, "acc": float((pred == oos_test).mean()),
            "oos_recall": tp / max(1, int(oos_test.sum())),
            "oos_precision": tp / max(1, int(pred.sum()))}


def zscore_by(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (x - ref.mean()) / (ref.std() + 1e-9)


@torch.no_grad()
def knn_scores(train: torch.Tensor, test: torch.Tensor, ks: Sequence[int], device, chunk: int = 1024) -> Dict[int, np.ndarray]:
    """Distance to the k-th nearest in-scope training feature (L2-normalised features)."""
    tr = F.normalize(train.to(device), dim=-1)
    out = {k: np.zeros(len(test)) for k in ks}
    kmax = max(ks)
    for s in range(0, len(test), chunk):
        te = F.normalize(test[s:s + chunk].to(device), dim=-1)
        sim = te @ tr.T
        top = torch.topk(sim, kmax, dim=-1).values          # descending similarity
        for k in ks:
            out[k][s:s + chunk] = torch.sqrt((2 - 2 * top[:, k - 1]).clamp_min(0)).cpu().numpy()
    return out


@torch.no_grad()
def mahalanobis_scores(train: torch.Tensor, y: torch.Tensor, n_cls: int, test: torch.Tensor, device,
                       shrink: float = 0.1) -> np.ndarray:
    """min_c (x-mu_c)^T S^-1 (x-mu_c) with a shared, shrunk covariance (Podolskiy et al. 2021 recipe).
    Runs on the CPU in float64 (MPS has no float64); 15k x 768 is a fraction of a second."""
    device = torch.device("cpu")
    x = train.to(device, torch.float64)
    mus = torch.stack([x[y == c].mean(0) for c in range(n_cls)])          # (C, D)
    centred = x - mus[y.to(device)]
    cov = centred.T @ centred / len(x)
    cov = (1 - shrink) * cov + shrink * torch.diag(torch.diag(cov)).mean() * torch.eye(cov.shape[0], device=device, dtype=cov.dtype)
    prec = torch.linalg.pinv(cov)
    te = test.to(device, torch.float64)
    a = te @ prec                                                          # (N, D)
    xpx = (a * te).sum(-1, keepdim=True)                                   # (N, 1)
    xpm = a @ mus.T                                                        # (N, C)
    mpm = ((mus @ prec) * mus).sum(-1)[None]                               # (1, C)
    d2 = xpx - 2 * xpm + mpm
    return d2.min(-1).values.clamp_min(0).cpu().numpy()


def subsample(rows: List, n: int, seed: int, key=None) -> List:
    """Deterministic subsample; with `key`, stratified by key(row) with about n rows in total."""
    rng = random.Random(seed)
    rows = list(rows)
    if key is None:
        rng.shuffle(rows)
        return rows[:n]
    groups: Dict = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    per = max(1, n // len(groups))
    out = []
    for g in groups.values():
        rng.shuffle(g)
        out.extend(g[:per])
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="CPU, small subsets, two depths")
    ap.add_argument("--device", default=None)
    ap.add_argument("--depths", type=int, nargs="*", default=[4, 8, 12, 16, 22])
    ap.add_argument("--steps", type=int, default=300, help="full-batch probe steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.device = args.device or "cpu"
        args.depths = args.depths if args.depths != [4, 8, 12, 16, 22] else [4, 12]
        args.out = args.out or "results/tarski/explore_syndrome_smoke.json"
    else:
        args.out = args.out or "results/tarski/explore_syndrome.json"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logf = open(args.out.replace(".json", ".log"), "a")
    L = lambda m: log(m, logf)

    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    trunk = Trunk(device=args.device)
    dev = trunk.device
    ds = data.load_clinc()
    intents, domains = ds.tasks["intent"].labels, ds.tasks["domain"].labels
    oos_i, oos_d = intents.index("oos"), domains.index("oos")

    train, val, test = ds.train, ds.val, ds.test
    if args.smoke:
        train = subsample(train, 1200, args.seed, key=lambda e: e.y["intent"])
        val = subsample(val, 300, args.seed + 1, key=lambda e: e.y["intent"])
        test_in = subsample([e for e in test if e.y["oos"] == 0], 600, args.seed + 2, key=lambda e: e.y["intent"])
        test_oos = subsample([e for e in test if e.y["oos"] == 1], 150, args.seed + 3)
        test = test_in + test_oos
    L(f"== syndrome decoding on {ds.name}: {len(train)}/{len(val)}/{len(test)} train/val/test, depths {args.depths}, "
      f"device {dev}, smoke={args.smoke}")

    # intent -> domain map from the data itself (a deterministic function; oos -> oos)
    dom_of = np.full(len(intents), -1, dtype=int)
    for e in train + val + test:
        i, d = e.y["intent"], e.y["domain"]
        assert dom_of[i] in (-1, d), "intent -> domain is not a function"
        dom_of[i] = d
    assert (dom_of >= 0).all()
    in_intents = [i for i in range(len(intents)) if i != oos_i]              # 150 in-scope intents
    in_domains = [d for d in range(len(domains)) if d != oos_d]              # 10 in-scope domains
    imap = {i: j for j, i in enumerate(in_intents)}                          # 151-index -> 150-index
    dmap = {d: j for j, d in enumerate(in_domains)}
    M = np.zeros((len(in_intents), len(in_domains)))                        # P_int150 -> implied P_dom10
    for i in in_intents:
        M[imap[i], dmap[dom_of[i]]] = 1.0
    dom150 = np.array([dmap[dom_of[i]] for i in in_intents])                 # 150-index -> 10-index

    # one trunk pass over everything, mean-pooled at each depth
    allx = train + val + test
    n_tr, n_va = len(train), len(val)
    idx = {"train": np.arange(0, n_tr), "val": np.arange(n_tr, n_tr + n_va), "test": np.arange(n_tr + n_va, len(allx))}
    t0 = time.time()
    feats = pooled_by_depth(trunk, [e.text for e in allx], args.depths, ds.max_len)
    t_trunk = time.time() - t0
    L(f"  trunk pass over {len(allx)} messages at {len(args.depths)} depths: {t_trunk:.1f}s")

    y_int = np.array([e.y["intent"] for e in allx])
    y_oos = np.array([e.y["oos"] for e in allx])
    in_mask = y_oos == 0
    y_int150 = np.array([imap.get(i, -1) for i in y_int])
    y_dom10 = np.array([dmap.get(dom_of[i], -1) for i in y_int])
    tr_in = idx["train"][in_mask[idx["train"]]]
    va_in = idx["val"][in_mask[idx["val"]]]
    te, te_in = idx["test"], idx["test"][in_mask[idx["test"]]]
    oos_te, oos_va = y_oos[te], y_oos[idx["val"]]
    always_in_acc = float(1 - oos_te.mean())

    results: Dict = {"config": vars(args), "device": str(dev), "n": {k: int(len(v)) for k, v in idx.items()},
                     "n_test_oos": int(oos_te.sum()), "always_in_scope_acc": always_in_acc,
                     "trunk_seconds": t_trunk, "depths": {}}
    # reference numbers from the sweep, when present
    sweep_path = "results/tarski/sweep_clinc150.json"
    if os.path.exists(sweep_path):
        sw = json.load(open(sweep_path))
        results["sweep_reference"] = {k: {t: round(m["acc"], 4) for t, m in v["tasks"].items()} for k, v in sw.items()}

    p_int150_by_depth: Dict[int, np.ndarray] = {}
    for d in args.depths:
        t0 = time.time()
        x = feats[d]
        mu, sd = x[idx["train"]].mean(0, keepdim=True), x[idx["train"]].std(0, keepdim=True).clamp_min(1e-4)
        z = (x - mu) / sd

        # --- probes ------------------------------------------------------------------------------
        probes = {}
        # in-scope-only heads (no OOS labels anywhere): intent150, domain10
        for name, rows_tr, rows_va, y_all, n_cls in (("int150", tr_in, va_in, y_int150, len(in_intents)),
                                                      ("dom10", tr_in, va_in, y_dom10, len(in_domains))):
            lin = fit_linear(z[rows_tr], torch.tensor(y_all[rows_tr]), n_cls, dev, args.steps, seed=args.seed)
            zv = logits_of(lin, z[rows_va], dev)
            T = fit_temperature(zv, torch.tensor(y_all[rows_va])) if len(rows_va) >= 30 else 1.0
            probes[name] = (lin, T)
        # references that use OOS training labels: 151-way intent (the sweep's task) and the binary oos probe
        for name, y_all, n_cls in (("int151", y_int, len(intents)), ("oos2", y_oos, 2)):
            lin = fit_linear(z[idx["train"]], torch.tensor(y_all[idx["train"]]), n_cls, dev, args.steps, seed=args.seed)
            zv = logits_of(lin, z[idx["val"]], dev)
            T = fit_temperature(zv, torch.tensor(y_all[idx["val"]]))
            probes[name] = (lin, T)

        P = {s: {} for s in ("val", "test")}
        for name, (lin, T) in probes.items():
            for s, rows in (("val", idx["val"]), ("test", te)):
                P[s][name] = softmax_np(logits_of(lin, z[rows], dev), T)
        p_int150_by_depth[d] = P["test"]["int150"]

        # --- accuracies --------------------------------------------------------------------------
        te_in_local = in_mask[te]                     # in-scope rows within the test block
        acc = {}
        acc["int151_all"] = float((P["test"]["int151"].argmax(-1) == y_int[te]).mean())
        acc["int150_inscope"] = float((P["test"]["int150"][te_in_local].argmax(-1) == y_int150[te_in]).mean())
        acc["dom10_inscope"] = float((P["test"]["dom10"][te_in_local].argmax(-1) == y_dom10[te_in]).mean())
        acc["dom_implied_by_int150_inscope"] = float((dom150[P["test"]["int150"][te_in_local].argmax(-1)] == y_dom10[te_in]).mean())
        acc["oos2_binary"] = float((P["test"]["oos2"].argmax(-1) == oos_te).mean())

        # consistent joint decoding: argmax_i log P_int(i) + lam * log P_dom(dom(i)); lam chosen on in-scope val
        def joint_decode(p_int, p_dom, lam):
            return (np.log(p_int + EPS) + lam * np.log(p_dom[:, dom150] + EPS)).argmax(-1)

        va_in_local = in_mask[idx["val"]]
        lams = [0.0, 0.25, 0.5, 1.0, 2.0]
        val_acc = {lam: float((joint_decode(P["val"]["int150"][va_in_local], P["val"]["dom10"][va_in_local], lam)
                               == y_int150[va_in]).mean()) for lam in lams}
        lam_star = max(lams, key=lambda l: (val_acc[l], -l))
        acc["joint_decode_val_acc_by_lambda"] = val_acc
        acc["joint_decode_lambda"] = lam_star
        for lam in lams:
            pred = joint_decode(P["test"]["int150"][te_in_local], P["test"]["dom10"][te_in_local], lam)
            acc[f"int150_joint_lam{lam}"] = float((pred == y_int150[te_in]).mean())
        acc["int150_joint_inscope"] = acc[f"int150_joint_lam{lam_star}"]

        # --- OOS scores (higher = more out of scope) ------------------------------------------------
        def scores_for(Pd: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[str, np.ndarray]:
            pi, pd_ = Pd["int150"], Pd["dom10"]
            implied = pi @ M
            s = {"msp_int": 1 - pi.max(-1), "ent_int": entropy(pi), "msp_dom": 1 - pd_.max(-1),
                 "syn_js": js_div(pd_, implied),
                 "syn_agree": 1 - (pi * pd_[:, dom150]).sum(-1),
                 "syn_joint": -(np.log(pi + EPS) + np.log(pd_[:, dom150] + EPS)).max(-1),
                 "ref_p_oos151": Pd["int151"][:, oos_i],
                 "ref_p_oos2": Pd["oos2"][:, 1]}
            return s

        S_val, S_test = scores_for(P["val"], idx["val"]), scores_for(P["test"], te)
        knn_va = knn_scores(z[tr_in], z[idx["val"]], (1, 10), dev)
        knn_te = knn_scores(z[tr_in], z[te], (1, 10), dev)
        for k in (1, 10):
            S_val[f"knn{k}"], S_test[f"knn{k}"] = knn_va[k], knn_te[k]
        S_val["maha"] = mahalanobis_scores(z[tr_in], torch.tensor(y_int150[tr_in]), len(in_intents), z[idx["val"]], dev)
        S_test["maha"] = mahalanobis_scores(z[tr_in], torch.tensor(y_int150[tr_in]), len(in_intents), z[te], dev)

        # label-free combinations: z-scored against in-scope validation messages, summed
        for combo in (("syn_js", "msp_int"), ("syn_js", "maha"), ("syn_js", "knn10"), ("syn_js", "msp_int", "maha"),
                      ("msp_int", "maha")):
            name = "sum_" + "+".join(combo)
            S_val[name] = sum(zscore_by(S_val[c][va_in_local], S_val[c]) for c in combo)
            S_test[name] = sum(zscore_by(S_val[c][va_in_local], S_test[c]) for c in combo)
        # supervised stacker (uses OOS validation labels): logistic regression over the label-free scores
        stack_feats = ["syn_js", "syn_agree", "msp_int", "ent_int", "msp_dom", "knn10", "maha"]
        Xv = np.stack([zscore_by(S_val[c][va_in_local], S_val[c]) for c in stack_feats], 1)
        Xt = np.stack([zscore_by(S_val[c][va_in_local], S_test[c]) for c in stack_feats], 1)
        if oos_va.sum() >= 5:
            lr_model = LogisticRegression(C=1.0, max_iter=1000).fit(Xv, oos_va)
            S_val["stack_supervised"], S_test["stack_supervised"] = lr_model.decision_function(Xv), lr_model.decision_function(Xt)
            stack_coef = dict(zip(stack_feats, [round(float(c), 3) for c in lr_model.coef_[0]]))
        else:
            stack_coef = None

        ood = {}
        for name in S_test:
            m = ood_metrics(S_test[name], oos_te)
            m.update({"binary_" + k: v for k, v in best_threshold_acc(S_val[name], oos_va, S_test[name], oos_te).items()})
            ood[name] = m
        # syndrome gate + consistent decoding as a 151-way decision, comparable with the sweep's intent task
        gate = best_threshold_acc(S_val["sum_syn_js+msp_int"], oos_va, S_test["sum_syn_js+msp_int"], oos_te)["threshold"]
        pred150 = joint_decode(P["test"]["int150"], P["test"]["dom10"], lam_star)
        pred151 = np.array([in_intents[j] for j in pred150])
        pred151[S_test["sum_syn_js+msp_int"] > gate] = oos_i
        acc["int151_syndrome_gate_plus_joint"] = float((pred151 == y_int[te]).mean())
        gate_ref = best_threshold_acc(S_val["ref_p_oos2"], oos_va, S_test["ref_p_oos2"], oos_te)["threshold"]
        pred151b = np.array([in_intents[j] for j in P["test"]["int150"].argmax(-1)])
        pred151b[S_test["ref_p_oos2"] > gate_ref] = oos_i
        acc["int151_supervised_gate_plus_argmax"] = float((pred151b == y_int[te]).mean())

        results["depths"][str(d)] = {"acc": acc, "oos": ood, "stack_coef": stack_coef,
                                     "temperatures": {k: float(v[1]) for k, v in probes.items()},
                                     "seconds": round(time.time() - t0, 1)}
        L(f"-- depth {d} ({time.time() - t0:.1f}s)")
        L(f"   acc: int151 {acc['int151_all']:.4f} | in-scope int150 {acc['int150_inscope']:.4f} -> joint(lam={lam_star}) "
          f"{acc['int150_joint_inscope']:.4f} | dom10 {acc['dom10_inscope']:.4f} (implied by intent "
          f"{acc['dom_implied_by_int150_inscope']:.4f}) | oos2 binary {acc['oos2_binary']:.4f} (always in-scope {always_in_acc:.4f})")
        L(f"   151-way with gate: syndrome gate + joint {acc['int151_syndrome_gate_plus_joint']:.4f} vs supervised gate + argmax "
          f"{acc['int151_supervised_gate_plus_argmax']:.4f}")
        order = sorted(ood, key=lambda n: -ood[n]["auroc"])
        for name in order:
            m = ood[name]
            L(f"   OOS {name:>26}: AUROC {m['auroc']:.4f}  AUPR {m['aupr']:.4f}  FPR@95 {m['fpr95']:.3f}  "
              f"binary acc {m['binary_acc']:.4f} (recall {m['binary_oos_recall']:.3f})")

    # idea 4: disagreement of the intent decision across depths as an OOS score
    if len(args.depths) >= 2:
        ds_sorted = sorted(args.depths)
        js_adj = np.mean([js_div(p_int150_by_depth[a], p_int150_by_depth[b]) for a, b in zip(ds_sorted[:-1], ds_sorted[1:])], 0)
        mean_p = np.mean([p_int150_by_depth[d] for d in ds_sorted], 0)
        mi = entropy(mean_p) - np.mean([entropy(p_int150_by_depth[d]) for d in ds_sorted], 0)
        results["depth_disagreement"] = {"js_adjacent": ood_metrics(js_adj, oos_te), "ensemble_mi": ood_metrics(mi, oos_te),
                                         "ensemble_msp": ood_metrics(1 - mean_p.max(-1), oos_te),
                                         "ensemble_int150_inscope_acc": float((mean_p[in_mask[te]].argmax(-1) == y_int150[te_in]).mean())}
        L(f"-- across depths {ds_sorted}: OOS AUROC js_adjacent {results['depth_disagreement']['js_adjacent']['auroc']:.4f}, "
          f"ensemble MI {results['depth_disagreement']['ensemble_mi']['auroc']:.4f}, ensemble MSP "
          f"{results['depth_disagreement']['ensemble_msp']['auroc']:.4f}; depth-ensemble in-scope acc "
          f"{results['depth_disagreement']['ensemble_int150_inscope_acc']:.4f}")

    # summary across depths: best label-free score vs baselines
    summary = {}
    for name in results["depths"][str(args.depths[0])]["oos"]:
        by_d = {d: results["depths"][str(d)]["oos"][name]["auroc"] for d in args.depths}
        best_d = max(by_d, key=by_d.get)
        summary[name] = {"best_depth": best_d, "auroc": by_d[best_d], "auroc_by_depth": by_d}
    results["summary_auroc"] = summary
    results["wall_seconds"] = round(time.time() - t_start, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    L(f"== done in {results['wall_seconds']}s; wrote {args.out}")
    L("   best AUROC per score: " + ", ".join(f"{n} {s['auroc']:.3f}@{s['best_depth']}" for n, s in
                                              sorted(summary.items(), key=lambda kv: -kv[1]['auroc'])))


if __name__ == "__main__":
    main()
