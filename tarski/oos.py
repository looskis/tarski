"""Out-of-scope detection without out-of-scope labels.

A branch answers only the labels it was trained on. To notice a message that fits none of them, each
branch carries class-conditional Gaussians with a tied covariance over the L2-normalised, mean-pooled
trunk states at its own split (Lee et al., 2018; Müller and Hein, 2025), plus one background Gaussian
over all its training messages, and scores a message by the relative Mahalanobis distance
(Ren et al., 2021):

    RMD(x) = min_c MD_c(x) - MD_0(x)

High RMD means the message sits far from every label's cluster, relative to the data as a whole. The
threshold is a quantile of RMD over the in-scope validation messages, so about (1 - quantile) of normal
traffic is flagged; on CLINC150 this reaches AUROC 0.957 with no out-of-scope labels (see
docs/research/THESIS.md, section 6.2). Covariances are shrunk towards a scaled identity, which matters
when a task has fewer messages than the trunk has dimensions (768). Scoring reads the tap the branch
already receives, so it adds nothing to the trunk's cost.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

FILE = "oos.safetensors"
META = "oos.json"


def _shrunk_precision(cov: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of (1-a) cov + a (tr(cov)/D) I, with a growing as n falls below D."""
    d = cov.shape[0]
    a = min(0.95, max(0.05, d / (n + d)))
    target = torch.eye(d, dtype=cov.dtype) * (torch.trace(cov) / d).clamp_min(1e-8)
    return torch.linalg.inv((1 - a) * cov + a * target)


class OOSStats:
    def __init__(self, means: torch.Tensor, prec: torch.Tensor, bg_mean: torch.Tensor, bg_prec: torch.Tensor,
                 tau: float, quantile: float, n: int):
        self.means, self.prec, self.bg_mean, self.bg_prec = means, prec, bg_mean, bg_prec
        self.tau, self.quantile, self.n = float(tau), float(quantile), int(n)

    # -- fitting ------------------------------------------------------------------------------------

    @staticmethod
    def fit(feats: torch.Tensor, y: torch.Tensor, n_labels: int, quantile: float = 0.95,
            val_feats: Optional[torch.Tensor] = None, min_val: int = 30) -> "OOSStats":
        """`feats`: pooled trunk states of the training messages [n, D]; `y`: their labels. The threshold
        comes from `val_feats` when there are at least `min_val` of them, else from the training rows."""
        z = F.normalize(feats.detach().float().cpu().double(), dim=-1)
        n, d = z.shape
        y = y.long().cpu()
        means = torch.stack([z[y == c].mean(0) if bool((y == c).any()) else z.mean(0) for c in range(n_labels)])
        centred = z - means[y]
        cov = centred.T @ centred / max(n - 1, 1)
        bg_mean = z.mean(0)
        bg_c = z - bg_mean
        bg_cov = bg_c.T @ bg_c / max(n - 1, 1)
        stats = OOSStats(means, _shrunk_precision(cov, n), bg_mean, _shrunk_precision(bg_cov, n), 0.0, quantile, n)
        if val_feats is not None and len(val_feats) >= min_val:
            ref = stats.score(val_feats)
        else:
            ref = stats._loo_scores(z, y)
        stats.tau = float(torch.quantile(ref, quantile))
        return stats

    def _loo_scores(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RMD of each training row with its own contribution removed from its class mean and from the
        background mean, so a threshold set on training rows reflects unseen in-scope messages rather
        than rows the means were fitted on."""
        n, c = z.shape[0], self.means.shape[0]
        diff = z[:, None, :] - self.means[None]
        d = torch.einsum("ncd,de,nce->nc", diff, self.prec, diff)
        n_c = torch.bincount(y, minlength=c).to(z.dtype)[y]
        own = torch.where((n_c > 1)[:, None], (self.means[y] * n_c[:, None] - z) / (n_c - 1).clamp_min(1)[:, None],
                          self.means[y])
        dd = z - own
        d[torch.arange(n), y] = torch.einsum("nd,de,ne->n", dd, self.prec, dd)
        bg = z - (self.bg_mean * n - z) / max(n - 1, 1)
        d_0 = torch.einsum("nd,de,ne->n", bg, self.bg_prec, bg)
        return (d.min(1).values - d_0).float().cpu()

    # -- scoring ------------------------------------------------------------------------------------

    @torch.no_grad()
    def score(self, feats: torch.Tensor) -> torch.Tensor:
        """Relative Mahalanobis distance per row; higher means further from every label. Computed on the
        CPU in float64 (MPS has no float64; one pooled vector per message makes the copy negligible)."""
        z = F.normalize(feats.detach().float().cpu().double(), dim=-1)
        diff = z[:, None, :] - self.means[None]                         # [n, C, D]
        d_c = torch.einsum("ncd,de,nce->nc", diff, self.prec, diff).min(1).values
        bg = z - self.bg_mean
        d_0 = torch.einsum("nd,de,ne->n", bg, self.bg_prec, bg)
        return (d_c - d_0).float().cpu()

    def flags(self, feats: torch.Tensor) -> torch.Tensor:
        return self.score(feats) > self.tau

    # -- persistence --------------------------------------------------------------------------------

    def config(self) -> Dict:
        return {"tau": self.tau, "quantile": self.quantile, "n": self.n, "labels": int(self.means.shape[0])}

    def save(self, path: str) -> None:
        save_file({"means": self.means.contiguous(), "prec": self.prec.contiguous(),
                   "bg_mean": self.bg_mean.contiguous(), "bg_prec": self.bg_prec.contiguous()},
                  os.path.join(path, FILE))
        with open(os.path.join(path, META), "w") as f:
            json.dump(self.config(), f, indent=1)

    @staticmethod
    def load(path: str) -> Optional["OOSStats"]:
        if not os.path.exists(os.path.join(path, FILE)):
            return None
        t = load_file(os.path.join(path, FILE))
        with open(os.path.join(path, META)) as f:
            m = json.load(f)
        return OOSStats(t["means"], t["prec"], t["bg_mean"], t["bg_prec"], m["tau"], m["quantile"], m["n"])
