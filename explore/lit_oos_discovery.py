"""Entry 10 (lit_incremental_oos.md): mine the out-of-scope inbox for new-route proposals.

  - Song et al., "Continual Generalized Intent Discovery" (CGID / PLRD, EMNLP Findings 2023): discover
    out-of-domain intent clusters from a stream and add them to the classifier incrementally.
  - Aida & Formentin, "Uncertainty-Aware Continual Learning for Open-World Intent Discovery Under an
    Evolving Label Space" (arXiv 2609.17866, 2026): flag unknowns, cluster them by density, promote only
    confident clusters. They report near-zero NMI/ARI, so the evidence is weak.
This script tests the product loop rather than a new method: out-of-scope-flagged messages pile up, the
frozen trunk clusters them, a dense tight cluster is proposed as "a new route may be forming", and an
accepted cluster becomes a route through the additive LDA statistics of entry 1 / geometry idea 10.

Setup (CLINC150): 15 intents held out as "routes that do not exist yet"; the known routes are the other 135.
  flag      Mahalanobis++ at --flag-depth, threshold keeping 95% of known-route validation messages
  inbox     flagged messages among: held-out intents' train+val messages, real out-of-scope messages
            (train + val + half of test) and known-route validation messages
  cluster   HDBSCAN (min cluster size 10 / 20) on L2-normalised, centred trunk states (PCA to 50 dims)
            at depths 4 and 22
  proposal  a cluster with >= 10 members; "recovered" = majority is a held-out intent with purity >= 0.8
  promote   add the cluster as a class to LDA statistics at --route-depth (z-scored), no retraining
Metrics: flag recall on held-out-intent messages; recovered intents out of 15; junk proposals (majority
out-of-scope or known, or purity < 0.8); ARI on flagged held-out messages; recall of promoted routes on the
held-out intents' TEST messages vs routes built from 10 true-labelled inbox messages and from all of them,
and overall accuracy on known + recovered test messages.
Expected: 50-70% of held-out intents recovered, 1-3 junk clusters from real out-of-scope, promoted routes
within 5-10 points of the 10-label routes.

Usage:
  .venv/bin/python explore/lit_oos_discovery.py --smoke
  .venv/bin/python explore/lit_oos_discovery.py --out results/tarski/explore_lit_discovery.json   # ~3 min on an A10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_oos_common import REPO, Gauss, Logger, get_trunk, l2n, load_ds, pooled, seed_all, split_index, zscore

import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-heldout", type=int, default=15)
    ap.add_argument("--flag-depth", type=int, default=4)
    ap.add_argument("--cluster-depths", type=int, nargs="*", default=[4, 22])
    ap.add_argument("--route-depth", type=int, default=22)
    ap.add_argument("--min-sizes", type=int, nargs="*", default=[10, 20])
    ap.add_argument("--min-proposal", type=int, default=10)
    ap.add_argument("--purity", type=float, default=0.8)
    ap.add_argument("--labels-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.n_heldout, args.min_sizes, args.min_proposal, args.labels_k = 10, [3], 3, 3
    out = args.out or ("results/tarski/explore_lit_discovery_smoke.json" if args.smoke
                       else "results/tarski/explore_lit_discovery.json")
    log = Logger(out)
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    trunk = get_trunk(args.smoke)
    t0 = time.time()
    ds = load_ds("clinc150", args.smoke, args.seed)
    allx, idx = split_index(ds)
    intents = ds.tasks["intent"].labels
    oos_i = intents.index("oos")
    with open(os.path.join(REPO, "tarski", "resources", "clinc_domains.json")) as f:
        domains = json.load(f)
    doms = sorted(domains)
    held = []
    k = 0
    while len(held) < args.n_heldout:                           # round-robin over domains
        its = sorted(set(domains[doms[k % len(doms)]]) - set(held))
        held.append(str(rng.choice(its)))
        k += 1
    held_ids = sorted(intents.index(h) for h in held)
    known = [i for i in range(len(intents)) if i != oos_i and i not in held_ids]
    y_int = np.array([e.y["intent"] for e in allx])
    is_held = np.isin(y_int, held_ids)
    is_oos = y_int == oos_i
    is_known = ~(is_held | is_oos)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    # route label space: known routes first, then (possibly) promoted held-out routes
    lab = {i: j for j, i in enumerate(known)}
    for j, h in enumerate(held_ids):
        lab[h] = len(known) + j
    y = np.array([lab.get(int(i), -1) for i in y_int])           # -1 = out-of-scope
    Ctot = len(known) + len(held_ids)
    depths = sorted(set([args.flag_depth, args.route_depth] + args.cluster_depths))
    F_ = pooled(trunk, [e.text for e in allx], depths, ds.max_len, log=log)
    log(f"== OOS-inbox route discovery | {len(known)} known routes, {len(held_ids)} held out: {', '.join(held)}")

    # flag
    tr_k, va_k = tr[is_known[tr]], va[is_known[va]]
    cen = F_[args.flag_depth][tr_k].mean(0, keepdim=True)
    g = Gauss(l2n(F_[args.flag_depth][tr_k] - cen), y[tr_k], Ctot, 0.1)
    flag_score = g.min_dist(l2n(F_[args.flag_depth] - cen))
    thr = float(np.quantile(flag_score[va_k], 0.95))
    flagged = flag_score > thr
    te_oos = te[is_oos[te]]
    te_oos_inbox = te_oos[: len(te_oos) // 2]
    cand = np.concatenate([tr[is_held[tr]], va[is_held[va]], tr[is_oos[tr]], va[is_oos[va]], te_oos_inbox, va_k])
    inbox = cand[flagged[cand]]
    held_cand = cand[is_held[cand]]
    res = {"args": vars(args), "held_out": held,
           "flag": {"threshold": thr, "recall_held_out": float(flagged[held_cand].mean()),
                    "recall_oos": float(flagged[cand[is_oos[cand]]].mean()),
                    "known_val_flagged": float(flagged[va_k].mean()), "inbox_size": int(len(inbox)),
                    "inbox_composition": {"held_out": int(is_held[inbox].sum()), "oos": int(is_oos[inbox].sum()),
                                          "known": int(is_known[inbox].sum())}}}
    log(f"   flag: held-out recall {res['flag']['recall_held_out']:.3f}, OOS recall {res['flag']['recall_oos']:.3f}, "
        f"inbox {len(inbox)} = {res['flag']['inbox_composition']}")

    # LDA routes at route depth (z-scored with known-route training statistics)
    Z = zscore(F_[args.route_depth], tr_k)
    te_eval = te[~is_oos[te]]                                    # known + held-out test messages

    def eval_routes(extra_rows: dict) -> dict:
        """extra_rows: held-out route label -> dataset rows used as that route's examples."""
        rows = [tr_k]
        ys = [y[tr_k]]
        for lbl, r in extra_rows.items():
            if len(r):
                rows.append(np.asarray(r))
                ys.append(np.full(len(r), lbl))
        R, Y = np.concatenate(rows), np.concatenate(ys)
        gg = Gauss(Z[R], Y, Ctot, 0.1)
        pred = gg.lda_logits(Z[te_eval]).argmax(-1).numpy()
        yt = y[te_eval]
        promoted = list(extra_rows)
        m_h = np.isin(yt, promoted)
        m_k = yt < len(known)
        return {"recall_promoted": float((pred == yt)[m_h].mean()) if m_h.any() else None,
                "acc_known": float((pred == yt)[m_k].mean()),
                "acc_known_plus_promoted": float((pred == yt)[m_h | m_k].mean())}

    runs = {}
    for d in args.cluster_depths:
        Xc = l2n(F_[d][inbox] - F_[d][tr_k].mean(0, keepdim=True)).numpy()
        Xp = PCA(n_components=min(50, len(inbox) - 1, Xc.shape[1]), random_state=args.seed).fit_transform(Xc)
        for mcs in args.min_sizes:
            cl = HDBSCAN(min_cluster_size=mcs, min_samples=5 if not args.smoke else 2).fit_predict(Xp)
            truth = np.where(is_oos[inbox], -1, y[inbox])
            proposals, recovered, junk = [], {}, 0
            for c in sorted(set(cl) - {-1}):
                mem = inbox[cl == c]
                if len(mem) < args.min_proposal:
                    continue
                vals, cnt = np.unique(truth[cl == c], return_counts=True)
                maj, pur = int(vals[np.argmax(cnt)]), float(cnt.max() / cnt.sum())
                kind = "held_out" if maj >= len(known) else ("oos" if maj == -1 else "known")
                proposals.append({"size": int(len(mem)), "majority": kind, "purity": pur,
                                  "intent": intents[held_ids[maj - len(known)]] if kind == "held_out" else None})
                if kind == "held_out" and pur >= args.purity:
                    if maj not in recovered or len(mem) > len(recovered[maj]):
                        recovered[maj] = mem
                else:
                    junk += 1
            mh = is_held[inbox]
            ari = float(adjusted_rand_score(truth[mh], cl[mh])) if mh.sum() > 1 else None
            # promotion: cluster members as the route's examples, vs 10 / all true-labelled inbox messages
            promo = eval_routes({lbl: mem for lbl, mem in recovered.items()})
            ten = {lbl: rng.choice(inbox[truth == lbl], min(args.labels_k, int((truth == lbl).sum())), replace=False)
                   for lbl in recovered}
            ten_r = eval_routes(ten)
            all_r = eval_routes({lbl: inbox[truth == lbl] for lbl in recovered})
            key = f"depth{d}_mcs{mcs}"
            runs[key] = {"n_clusters": int(len(set(cl) - {-1})), "noise_share": float((cl == -1).mean()),
                         "n_proposals": len(proposals), "recovered": len(recovered), "junk_proposals": junk,
                         "recovered_intents": [intents[held_ids[l - len(known)]] for l in recovered],
                         "ari_held_out": ari, "proposals": proposals,
                         "promoted_from_cluster": promo, f"route_from_{args.labels_k}_labels": ten_r,
                         "route_from_all_inbox_labels": all_r}
            log(f"-- depth {d}, min size {mcs}: {runs[key]['n_clusters']} clusters, {len(proposals)} proposals, "
                f"recovered {len(recovered)}/{len(held_ids)}, junk {junk}, ARI {ari if ari is None else round(ari, 3)} | "
                f"promoted recall {promo['recall_promoted']} vs {args.labels_k}-label {ten_r['recall_promoted']} vs "
                f"all-label {all_r['recall_promoted']}; known acc {promo['acc_known']:.4f}")
            log.dump({**res, "runs": runs})
    res["runs"] = runs
    res["no_promotion"] = eval_routes({})
    # upper bound: every held-out intent present in the inbox promoted with its true inbox labels
    truth_all = np.where(is_oos[inbox], -1, y[inbox])
    present = {int(l): inbox[truth_all == l] for l in np.unique(truth_all) if l >= len(known)}
    res["upper_all_present_true_labels"] = {"n_routes": len(present), **eval_routes(present)}
    log(f"   upper bound (all {len(present)} held-out intents in the inbox, true labels): "
        f"{res['upper_all_present_true_labels']}")
    res["wall_s"] = round(time.time() - t0, 1)
    log.dump(res)
    log(f"done in {res['wall_s']}s -> {out}")


if __name__ == "__main__":
    main()
