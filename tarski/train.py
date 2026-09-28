"""Train branches on a frozen trunk.

The trunk runs once over the training messages and its hidden states at the split depth are cached
(fp16, CPU). Every task's branch then trains on that cache, so adding a decision never re-runs the
base model over the data. Each branch is calibrated with one temperature fitted on validation data.
"""

from __future__ import annotations

import contextlib
import math
import copy
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from tarski.branches import Branch, BlockBranch, ProbeBranch
from tarski.data import Dataset, Example
from tarski.trunk import Trunk


def autocast(device: torch.device):
    if device.type == "mps":
        return torch.autocast("mps", dtype=torch.float16)
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------------------------------

class FeatureCache:
    """Token-level trunk states at one or more depths for a list of texts."""

    def __init__(self, trunk: Trunk, texts: Sequence[str], depths: Sequence[int], max_len: int, bs: int = 64):
        self.trunk, self.depths = trunk, sorted(set(depths))
        ids = trunk.token_ids(texts, max_len)
        self.lengths = [len(x) for x in ids]
        self.h: Dict[int, List[torch.Tensor]] = {d: [None] * len(texts) for d in self.depths}
        order = sorted(range(len(ids)), key=lambda i: self.lengths[i])
        t0 = time.time()
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            L = max(self.lengths[i] for i in idx)
            x = torch.full((len(idx), L), trunk.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                x[j, : self.lengths[i]] = torch.tensor(ids[i])
                att[j, : self.lengths[i]] = 1
            with autocast(trunk.device):
                taps, _ = trunk.taps(x.to(trunk.device), att.to(trunk.device), self.depths)
            for d in self.depths:
                hd = taps[d].to("cpu", torch.float16)
                for j, i in enumerate(idx):
                    self.h[d][i] = hd[j, : self.lengths[i]].clone()
        self.seconds = time.time() - t0

    def batch(self, depth: int, idx: Sequence[int], dtype=torch.float32):
        L = max(self.lengths[i] for i in idx)
        h = torch.zeros(len(idx), L, self.trunk.hidden, dtype=torch.float16)
        att = torch.zeros(len(idx), L, dtype=torch.long)
        for j, i in enumerate(idx):
            h[j, : self.lengths[i]] = self.h[depth][i]
            att[j, : self.lengths[i]] = 1
        h, att = h.to(self.trunk.device, dtype), att.to(self.trunk.device)
        return h, self.trunk.context(h, att)

    def bytes(self) -> int:
        return sum(t.numel() * 2 for d in self.depths for t in self.h[d])


# ---------------------------------------------------------------------------------------------------
# Metrics and calibration
# ---------------------------------------------------------------------------------------------------

def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += abs(conf[sel].mean() - correct[sel].mean()) * sel.mean()
    return float(e)


def evaluate(probs: np.ndarray, y: np.ndarray, soft: Optional[np.ndarray] = None) -> Dict[str, float]:
    pred = probs.argmax(-1)
    correct = (pred == y).astype(np.float64)
    out = {"acc": float(correct.mean()),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(probs.shape[1]), zero_division=0)),
           "nll": float(-np.log(probs[np.arange(len(y)), y].clip(1e-12)).mean()),
           "ece": ece_score(probs.max(-1), correct), "n": int(len(y))}
    if soft is not None:
        out["brier_soft"] = float(((probs - soft) ** 2).sum(-1).mean())
    return out


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, soft: Optional[torch.Tensor] = None) -> float:
    target = soft if soft is not None else F.one_hot(y, logits.shape[-1]).float()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = -(target * F.log_softmax(logits / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp())
    # LBFGS can diverge to NaN/inf on near-separable logits; clamp() passes NaN through, so guard it
    return min(max(t, 0.05), 20.0) if math.isfinite(t) else 1.0


# ---------------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------------

def _targets(examples: List[Example], task: str):
    y = torch.tensor([e.y[task] for e in examples])
    soft = None
    if all(task in e.soft for e in examples):
        soft = torch.tensor(np.stack([e.soft[task] for e in examples]))
    return y, soft


@torch.no_grad()
def predict_logits(branch: Branch, cache: FeatureCache, idx: Sequence[int], bs: int = 128) -> torch.Tensor:
    branch.eval()
    order = sorted(range(len(idx)), key=lambda j: cache.lengths[idx[j]])
    out = torch.zeros(len(idx), branch.n_labels)
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        h, ctx = cache.batch(branch.split, [idx[j] for j in sel])
        out[sel] = branch.logits(h, ctx).float().cpu()
    return out


@torch.no_grad()
def pooled_states(cache: FeatureCache, depth: int, idx: Sequence[int], bs: int = 128) -> torch.Tensor:
    """Mean-pooled trunk states at `depth` for the rows `idx`, in order (what the out-of-scope detector reads)."""
    from tarski.branches import mean_pool

    out = torch.zeros(len(idx), cache.trunk.hidden)
    for s in range(0, len(idx), bs):
        sel = list(range(s, min(s + bs, len(idx))))
        h, ctx = cache.batch(depth, [idx[j] for j in sel])
        out[sel] = mean_pool(h.float(), ctx.attention_mask).float().cpu()
    return out


def make_branch(kind: str, split: int, labels: List[str], trunk: Trunk, depth: int = 2, init: str = "next") -> Branch:
    if kind == "probe":
        return ProbeBranch(split, labels, trunk.hidden)
    if kind == "blocks":
        return BlockBranch(split, labels, trunk.hidden, depth, trunk, init=init)
    raise ValueError(kind)


def train_branch(branch: Branch, cache: FeatureCache, train_idx: List[int], y: torch.Tensor,
                 soft: Optional[torch.Tensor], val_idx: List[int], y_val: torch.Tensor,
                 epochs: int = 8, bs: int = 32, lr_layers: float = 1e-4, lr_head: float = 1e-3,
                 patience: int = 3, seed: int = 0, log: Callable[[str], None] = lambda s: None,
                 min_val: int = 50, min_steps: int = 300) -> Dict:
    """Train on cached trunk states.

    With at least `min_val` validation examples, keep the epoch with the best validation accuracy and
    stop after `patience` epochs without improvement, but never before half the schedule has run (the
    early epochs sit in the learning-rate warm-up). With fewer, validation accuracy is too noisy to select
    or stop on, so the full schedule runs and the final, annealed weights are kept.
    """
    # small datasets get more epochs: a few hundred examples at batch 32 is only ~10 steps per epoch
    per_epoch = (len(train_idx) + bs - 1) // bs
    epochs = max(epochs, -(-min_steps // per_epoch))
    select = len(val_idx) >= min_val
    min_epochs = max(1, (epochs + 1) // 2)
    torch.manual_seed(seed)
    dev = cache.trunk.device
    branch.to(dev).train()
    layer_params = [p for n, p in branch.named_parameters() if n.startswith("layers.")]
    head_params = [p for n, p in branch.named_parameters() if not n.startswith("layers.")]
    groups = [{"params": head_params, "lr": lr_head}]
    if layer_params:
        groups.append({"params": layer_params, "lr": lr_layers})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = epochs * ((len(train_idx) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(epochs):
        branch.train()
        t0, tot = time.time(), 0.0
        # length-bucketed batches, shuffled
        order = sorted(range(len(train_idx)), key=lambda j: cache.lengths[train_idx[j]] + rng.random() * 8)
        chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
        rng.shuffle(chunks)
        for sel in chunks:
            h, ctx = cache.batch(branch.split, [train_idx[j] for j in sel])
            z = branch.logits(h, ctx).float()
            if soft is not None:
                loss = -(soft[sel].to(dev) * F.log_softmax(z, -1)).sum(-1).mean()
            else:
                loss = F.cross_entropy(z, y[sel].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(sel)
        va = float((predict_logits(branch, cache, val_idx).argmax(-1) == y_val).float().mean())
        hist.append({"epoch": ep + 1, "loss": tot / len(train_idx), "val_acc": va, "s": round(time.time() - t0, 1)})
        log(f"    epoch {ep + 1}/{epochs} loss {tot / len(train_idx):.4f} val acc {va:.4f} ({time.time() - t0:.1f}s)")
        if not select:
            best = va
            continue
        if va > best:
            best, best_state, bad = va, copy.deepcopy(branch.state_dict()), 0
        else:
            bad += 1
            if bad >= patience and ep + 1 >= min_epochs:
                break
    if select:
        branch.load_state_dict(best_state)
    branch.eval()
    return {"val_acc": best, "history": hist, "selection": "best_val_acc" if select else "last_epoch"}


def fit(trunk: Trunk, ds: Dataset, tasks: Optional[Sequence[str]] = None, kind: str = "blocks", split: int = 11,
        depth: int = 2, epochs: int = 8, lr_layers: float = 1e-4, lr_head: float = 1e-3, seed: int = 0,
        store: Optional[str] = None, log: Callable[[str], None] = print, cache: Optional[Dict] = None,
        init: str = "next", oos: bool = True, oos_quantile: float = 0.95) -> Dict:
    """Train one branch per task at one split depth, sharing a single trunk pass over the data.

    Returns per-task test metrics. With `store`, saves each branch to `<store>/<task>/`.
    Pass the same `cache` dict across calls to reuse trunk states between runs at the same depth.
    """
    import os

    tasks = list(tasks or ds.tasks)
    cache = cache if cache is not None else {}
    key = (ds.name, split)
    if key not in cache:
        texts = [e.text for e in ds.train + ds.val + ds.test]
        cache[key] = FeatureCache(trunk, texts, [split], ds.max_len)
        log(f"  cached trunk depth {split} for {len(texts)} messages in {cache[key].seconds:.1f}s "
            f"({cache[key].bytes() / 1e6:.0f} MB)")
    fc = cache[key]
    n_tr, n_va = len(ds.train), len(ds.val)
    split_idx = {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va), "test": range(n_tr + n_va, len(fc.lengths))}
    allx = ds.train + ds.val + ds.test
    results = {}
    for task in tasks:
        sel = {s: [i for i in split_idx[s] if task in allx[i].y] for s in split_idx}
        ex = {s: [allx[i] for i in sel[s]] for s in sel}
        y, soft = {}, {}
        for s in sel:
            y[s], soft[s] = _targets(ex[s], task)
        labels = ds.tasks[task].labels
        branch = make_branch(kind, split, labels, trunk, depth, init)
        t0 = time.time()
        log(f"  [{task}] {kind} split={split}" + (f" depth={depth}" if kind == "blocks" else "") +
            f", {len(labels)} labels, {len(sel['train'])} train")
        info = train_branch(branch, fc, sel["train"], y["train"], soft["train"], sel["val"], y["val"],
                            epochs=epochs, lr_layers=lr_layers, lr_head=lr_head, seed=seed, log=log)
        train_s = time.time() - t0
        # a temperature fitted on a handful of rows is noise; below 30 validation rows keep T = 1
        t = fit_temperature(predict_logits(branch, fc, sel["val"]), y["val"], soft["val"]) \
            if len(sel["val"]) >= 30 else 1.0
        branch.temperature.fill_(t)
        z = predict_logits(branch, fc, sel["test"])
        probs = torch.softmax(z / t, -1).numpy()
        m = evaluate(probs, y["test"].numpy(), None if soft["test"] is None else soft["test"].numpy())
        m.update({"val_acc": info["val_acc"], "temperature": t, "train_s": round(train_s, 1),
                  "epochs_run": len(info["history"]), "params": sum(p.numel() for p in branch.parameters())})
        results[task] = m
        log(f"  [{task}] test acc {m['acc']:.4f} macro-F1 {m['macro_f1']:.4f} ECE {m['ece']:.3f} "
            f"(T={t:.2f}, {train_s:.0f}s, {m['params'] / 1e6:.2f}M params)")
        stats = None
        if oos:
            # out-of-scope detector on the same tap the branch reads; thresholded so that about
            # (1 - quantile) of in-scope validation messages are flagged
            from tarski.oos import OOSStats

            feats_val = pooled_states(fc, split, sel["val"]) if sel["val"] else None
            stats = OOSStats.fit(pooled_states(fc, split, sel["train"]), y["train"], len(labels), oos_quantile,
                                 feats_val)
            branch.oos = stats
            if sel["test"]:
                m["oos_flag_rate"] = float(stats.flags(pooled_states(fc, split, sel["test"])).float().mean())
        if store:
            branch.save(os.path.join(store, task), trunk.fingerprint(), trunk.base,
                        {"task": task, "dataset": ds.name, "metrics": m, "max_len": ds.max_len})
            if stats is not None:
                stats.save(os.path.join(store, task))
    return results
