"""Pick each task's split depth from one trunk pass.

Mean-pooled trunk states at every depth come out of a single forward pass. A linear probe per depth,
trained for a few hundred full-batch steps on the GPU, gives a validation-accuracy curve over depth.
The chosen split is the shallowest depth within `tol` of the best probe: decisions that are easy to read
off early layers branch early, so a message whose decisions all branch early never runs the upper trunk.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from tarski.branches import mean_pool
from tarski.data import Dataset
from tarski.train import autocast
from tarski.trunk import Trunk


@torch.no_grad()
def pooled_by_depth(trunk: Trunk, texts: Sequence[str], depths: Sequence[int], max_len: int,
                    bs: int = 64) -> Dict[int, torch.Tensor]:
    ids = trunk.token_ids(texts, max_len)
    out = {d: torch.zeros(len(texts), trunk.hidden) for d in depths}
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(ids[i]) for i in idx)
        x = torch.full((len(idx), L), trunk.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            x[j, : len(ids[i])] = torch.tensor(ids[i])
            att[j, : len(ids[i])] = 1
        x, att = x.to(trunk.device), att.to(trunk.device)
        with autocast(trunk.device):
            taps, _ = trunk.taps(x, att, depths)
        for d in depths:
            out[d][idx] = mean_pool(taps[d].float(), att).cpu()
    return out


def probe_accuracy(xtr, ytr, xva, yva, n_labels: int, device, steps: int = 300, lr: float = 1e-2,
                   wd: float = 1e-4) -> float:
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-4)
    xtr, xva = ((xtr - mu) / sd).to(device), ((xva - mu) / sd).to(device)
    ytr, yva = ytr.to(device), yva.to(device)
    lin = torch.nn.Linear(xtr.shape[1], n_labels).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        loss = F.cross_entropy(lin(xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((lin(xva).argmax(-1) == yva).float().mean())


def layer_curves(trunk: Trunk, ds: Dataset, tasks: Optional[Sequence[str]] = None,
                 depths: Optional[Sequence[int]] = None) -> Dict[str, Dict[int, float]]:
    tasks = list(tasks or ds.tasks)
    depths = list(depths or range(1, trunk.n_layers + 1))
    texts = [e.text for e in ds.train + ds.val]
    feats = pooled_by_depth(trunk, texts, depths, ds.max_len)
    n_tr = len(ds.train)
    curves = {}
    for task in tasks:
        tr = [i for i, e in enumerate(ds.train) if task in e.y]
        va = [n_tr + i for i, e in enumerate(ds.val) if task in e.y]
        ytr = torch.tensor([ds.train[i].y[task] for i in tr])
        yva = torch.tensor([ds.val[i - n_tr].y[task] for i in va])
        curves[task] = {d: probe_accuracy(feats[d][tr], ytr, feats[d][va], yva, len(ds.tasks[task].labels),
                                          trunk.device) for d in depths}
    return curves


def choose_split(curve: Dict[int, float], tol: float = 0.01) -> int:
    best = max(curve.values())
    return min(d for d, acc in curve.items() if acc >= best - tol)


def branch_curves(trunk: Trunk, ds: Dataset, tasks: Optional[Sequence[str]] = None,
                  depths: Sequence[int] = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20), steps: int = 120,
                  branch_depth: int = 1, seed: int = 0) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Validation accuracy and NLL by split depth from short 1-layer branch runs (a cheap proxy for the
    branch actually deployed). Linear probes can be flat across depth while branches are not (Banking77,
    CLINC150), so ranking depths by probes picks splits that are too shallow."""
    from tarski.train import FeatureCache, _targets, make_branch, predict_logits, train_branch

    tasks = list(tasks or ds.tasks)
    depths = [d for d in depths if d + branch_depth <= trunk.n_layers]
    fc = FeatureCache(trunk, [e.text for e in ds.train + ds.val], depths, ds.max_len)
    n_tr = len(ds.train)
    curves: Dict[str, Dict[int, Dict[str, float]]] = {}
    for task in tasks:
        tr = [i for i, e in enumerate(ds.train) if task in e.y]
        va = [n_tr + i for i, e in enumerate(ds.val) if task in e.y]
        y_tr, soft_tr = _targets([ds.train[i] for i in tr], task)
        y_va, _ = _targets([ds.val[i - n_tr] for i in va], task)
        curves[task] = {}
        for k in depths:
            br = make_branch("blocks", k, ds.tasks[task].labels, trunk, branch_depth)
            train_branch(br, fc, tr, y_tr, soft_tr, va, y_va, epochs=1, min_steps=steps, min_val=10 ** 9, seed=seed)
            z = predict_logits(br, fc, va)
            curves[task][k] = {"acc": float((z.argmax(-1) == y_va).float().mean()),
                               "nll": float(F.cross_entropy(z, y_va))}
    return curves


def choose_split_branch(curve: Dict[int, Dict[str, float]], tol: float = 0.01, by: str = "acc") -> int:
    """Shallowest depth within `tol` of the best proxy score (accuracy, or NLL in nats when by="nll")."""
    if by == "nll":
        best = min(v["nll"] for v in curve.values())
        return min(d for d, v in curve.items() if v["nll"] <= best + tol)
    best = max(v["acc"] for v in curve.values())
    return min(d for d, v in curve.items() if v["acc"] >= best - tol)
