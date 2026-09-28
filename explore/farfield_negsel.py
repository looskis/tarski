"""Far-field idea 3: negative selection (artificial immune system) for out-of-scope detection, CLINC150.

T-cells with random receptors are deleted if they bind self; the survivors detect anything that is not
self. Real-valued negative selection (V-detector, Ji & Dasgupta 2004): sample candidate detector centres,
delete any within the self radius of an in-scope training point, give the survivors a radius reaching to
the nearest self point, and flag a test point covered by any detector as out of scope.

Testable prediction (stated in docs/research/notes/explore/farfield.md): in a 64-d PCA of the trunk's pooled
state the surviving detectors tile the complement of the self set, so "covered by a detector" reduces to
"far from the nearest in-scope training point": NSA AUROC <= 1-NN distance AUROC on the same features and
the two scores are rank-correlated above 0.9. A quick replication of Sun et al. (2022) kNN is the baseline.

Usage:
  .venv/bin/python explore/farfield_negsel.py --smoke
  .venv/bin/python explore/farfield_negsel.py --out results/tarski/explore_negsel.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import numpy as np
import torch
from scipy.stats import spearmanr

from explore.farfield_common import (Logger, Standardiser, dump, knn_distance, ood_metrics, out_paths, split_indices,
                                     subsample)
from tarski import data
from tarski.autosplit import pooled_by_depth
from tarski.trunk import Trunk


@torch.no_grad()
def min_dist(q: torch.Tensor, ref: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    out = torch.empty(len(q), device=q.device)
    for s in range(0, len(q), chunk):
        out[s:s + chunk] = torch.cdist(q[s:s + chunk], ref).min(-1).values
    return out


@torch.no_grad()
def negative_selection(self_pts: torch.Tensor, n_candidates: int, self_radius: float, proposal_scale: float,
                       seed: int, max_detectors: int):
    """V-detector style: candidates are self points displaced by isotropic noise whose expected length is
    {1, 2, 3, 5} x `proposal_scale` x self_radius (per-dimension sigma = that / sqrt(d), so in 64-d the
    candidates land just outside the self set rather than far away), plus 10% from a broad Gaussian.
    Negative selection keeps those farther than `self_radius` from every self point; a survivor's radius is
    its distance to the nearest self point minus self_radius.

    By the triangle inequality max_j (r_j - ||x - c_j||) <= d(x, self) - self_radius, so the NSA margin is
    a lower bound on the 1-NN distance; equality needs a detector on the ray beyond x."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, d = self_pts.shape
    n_broad = n_candidates // 10
    per = (n_candidates - n_broad) // 4
    parts = []
    for mult in (1.0, 2.0, 3.0, 5.0):
        idx = torch.randint(0, n, (per,), generator=g)
        sigma = mult * proposal_scale * self_radius / d ** 0.5
        parts.append(self_pts[idx].cpu() + sigma * torch.randn(per, d, generator=g))
    mu, sd = self_pts.mean(0).cpu(), self_pts.std(0).cpu()
    parts.append(mu + 1.5 * sd * torch.randn(n_broad, d, generator=g))
    cand = torch.cat(parts).to(self_pts.device)
    dist = min_dist(cand, self_pts)
    keep = dist > self_radius                      # negative selection: delete anything that binds self
    centres, radii = cand[keep], (dist[keep] - self_radius)
    order = torch.argsort(radii, descending=True)[:max_detectors]
    return centres[order], radii[order], float(keep.float().mean())


@torch.no_grad()
def nsa_scores(x: torch.Tensor, centres: torch.Tensor, radii: torch.Tensor, chunk: int = 2048):
    """max_j (r_j - ||x - c_j||): positive when some detector covers x. Also the number of covering detectors."""
    best = torch.empty(len(x), device=x.device)
    count = torch.empty(len(x), device=x.device)
    for s in range(0, len(x), chunk):
        m = radii[None] - torch.cdist(x[s:s + chunk], centres)
        best[s:s + chunk] = m.max(-1).values
        count[s:s + chunk] = (m > 0).float().sum(-1)
    return best.cpu().numpy(), count.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--depths", type=int, nargs="*", default=[4, 12])
    ap.add_argument("--pca", type=int, nargs="*", default=[8, 16, 64], help="PCA dimensions to try (0 = no PCA, all 768 standardised dims)")
    ap.add_argument("--candidates", type=int, default=40000)
    ap.add_argument("--max-detectors", type=int, default=4000)
    ap.add_argument("--self-radius-quantile", type=float, nargs="*", default=[0.25, 0.5], help="self radius = this quantile of self 1-NN distances")
    ap.add_argument("--proposal-scale", type=float, default=1.0, help="displacement of candidates in units of the self radius")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.device = args.device or "cpu"
        args.depths = [4] if args.depths == [4, 12] else args.depths
        args.candidates, args.max_detectors = 4000, 500
        args.pca = [8, 0] if args.pca == [8, 16, 64] else args.pca
    out, logp = out_paths(args, "negsel")
    L = Logger(logp)
    t_start = time.time()

    trunk = Trunk(device=args.device)
    dev = trunk.device
    ds = data.load_clinc()
    train, val, test = ds.train, ds.val, ds.test
    if args.smoke:
        train = subsample(train, 1500, args.seed, key=lambda e: e.y["intent"])
        val = subsample(val, 300, args.seed + 1, key=lambda e: e.y["intent"])
        test = subsample([e for e in test if e.y["oos"] == 0], 600, args.seed + 2, key=lambda e: e.y["intent"]) + \
            subsample([e for e in test if e.y["oos"] == 1], 150, args.seed + 3)
    allx = train + val + test
    idx = split_indices(len(train), len(val), len(allx))
    y_oos = np.array([e.y["oos"] for e in allx])
    tr_in = idx["train"][y_oos[idx["train"]] == 0]
    te = idx["test"]
    L(f"== negative selection on clinc150: {len(train)}/{len(val)}/{len(test)}, depths {args.depths}, pca {args.pca}, "
      f"{args.candidates} candidates, device {dev}, smoke={args.smoke}")
    t0 = time.time()
    feats = pooled_by_depth(trunk, [e.text for e in allx], args.depths, ds.max_len)
    L(f"  trunk pass: {time.time() - t0:.1f}s")

    results = {"config": vars(args), "device": str(dev), "n_test": int(len(te)), "n_test_oos": int(y_oos[te].sum()), "depths": {}}
    for d, n_pca, q in [(d, k, q) for d in args.depths for k in args.pca for q in args.self_radius_quantile]:
        t0 = time.time()
        z = Standardiser(feats[d][tr_in])(feats[d])
        # PCA fitted on in-scope training rows (SVD of the centred matrix)
        xc = z[tr_in] - z[tr_in].mean(0, keepdim=True)
        _, S, V = torch.linalg.svd(xc, full_matrices=False)
        if n_pca > 0:
            P = V[:n_pca].T                                                  # (768, pca)
            explained = float((S[:n_pca] ** 2).sum() / (S ** 2).sum())
            zp = ((z - z[tr_in].mean(0, keepdim=True)) @ P).to(dev)
        else:
            explained, zp = 1.0, (z - z[tr_in].mean(0, keepdim=True)).to(dev)
        self_pts = zp[tr_in]
        # self radius from the self set's own 1-NN distances (leave-one-out on a sample)
        samp = self_pts[torch.randperm(len(self_pts), generator=torch.Generator().manual_seed(args.seed))[:2000]]
        dd = torch.cdist(samp, self_pts)
        dd[dd == 0] = float("inf")
        self_r = float(torch.quantile(dd.min(-1).values, q))
        centres, radii, survive = negative_selection(self_pts, args.candidates, self_r, args.proposal_scale, args.seed, args.max_detectors)
        best, count = nsa_scores(zp[te], centres, radii)
        coverage_in = float((best[y_oos[te] == 0] > 0).mean())
        coverage_oos = float((best[y_oos[te] == 1] > 0).mean())
        # baselines on the same PCA features and on the full features
        knn1_pca = min_dist(zp[te], self_pts).cpu().numpy()
        knn1_full = knn_distance(z[tr_in], z[te], 1, dev)
        knn10_full = knn_distance(z[tr_in], z[te], 10, dev)
        scores = {"nsa_max_margin": best, "nsa_count": count, "knn1_pca": knn1_pca, "knn1_full": knn1_full, "knn10_full": knn10_full}
        m = {k: ood_metrics(v, y_oos[te]) for k, v in scores.items()}
        rho = float(spearmanr(best, knn1_pca).correlation)
        bound_ok = float((best <= knn1_pca - self_r + 1e-4).mean())        # triangle-inequality bound, should be 1.0
        bound_gap = float(np.median(knn1_pca - self_r - best))               # how far the margin sits below the bound
        results["depths"][f"{d}_pca{n_pca}_q{q}"] = {"depth": d, "pca": n_pca, "self_radius_quantile": q, "ood": m, "n_detectors": int(len(radii)), "survival_rate": survive, "self_radius": self_r,
                                     "median_detector_radius": float(radii.median()) if len(radii) else None,
                                     "pca_explained_variance": explained, "coverage_inscope_test": coverage_in,
                                     "coverage_oos_test": coverage_oos, "spearman_nsa_vs_knn1_pca": rho,
                                     "bound_holds_fraction": bound_ok, "median_gap_to_knn1_bound": bound_gap,
                                     "seconds": round(time.time() - t0, 1)}
        L(f"-- depth {d}, PCA {n_pca or 768}, self-radius quantile {q}: {len(radii)} detectors survive ({survive:.1%} of candidates), self radius {self_r:.2f}, "
          f"PCA explains {explained:.2f}; covered: in-scope {coverage_in:.3f}, OOS {coverage_oos:.3f}")
        for k in scores:
            L(f"   {k:>16}: AUROC {m[k]['auroc']:.4f}  AUPR {m[k]['aupr']:.4f}  FPR@95 {m[k]['fpr95']:.3f}")
        L(f"   NSA margin <= 1-NN distance - self radius holds for {bound_ok:.1%} of test rows (median gap {bound_gap:.2f})")
        L(f"   Spearman(NSA margin, 1-NN distance in PCA space) = {rho:.3f}  "
          f"[prediction: NSA <= knn1 and rho > 0.9 -> {'holds' if m['nsa_max_margin']['auroc'] <= m['knn1_pca']['auroc'] + 0.005 and rho > 0.9 else 'fails'}]")
    results["wall_seconds"] = round(time.time() - t_start, 1)
    dump(results, out)
    L(f"== done in {results['wall_seconds']}s; wrote {out}")


if __name__ == "__main__":
    main()
