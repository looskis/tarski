"""k-shot curve for tarski branches, plus FastFit-style late-interaction class anchors (lit_decision_models #5).

The user's Slack routes will have a handful of labels each, and no experiment so far measures accuracy as a
function of labels per class. This script does, on Banking77 (77 intents) or CLINC150 (150 in-scope intents,
out-of-scope messages held out of training and used only to score oos detection), for k = 0/1/5/10/25/all
labelled messages per class:

  proto      nearest class centroid of mean-pooled trunk states (no training). k=0: centroids are the
             embeddings of the humanised label names ("card arrival").
  li         late interaction (FastFit, Yehudai & Bandel, NAACL 2024 demo): each class is a bag of trunk
             token states (its name + up to `li_cap` support messages); a message scores against a class by
             ColBERT MaxSim (mean over its tokens of the best cosine in the bag), after a small trained
             projection (LayerNorm + 768->128) learned with a contrastive cross-entropy over classes. A
             training message is masked out of its own class bag. `li_untrained` is the same scorer with no
             training (identity projection); at k=0 the bag is the label name only.
  probe      tarski ProbeBranch at --probe-depth.
  blocks     tarski BlockBranch at --block-split + --block-depth (the "branch" of the thesis).
  full       full fine-tune reference (BlockBranch at split 0 with all layers, embeddings frozen), as in
             experiments/sweep.py.

Protocol. Support sets are sampled per seed (3 seeds for k <= 10, 2 for k = 25, 1 for all). For k < all the
validation set is also k-shot (min(k, 10) per class, from the validation split): it is used only to fit the
temperature, never to pick an epoch (train_branch runs its full schedule, >= 300 optimiser steps, last
epoch kept), because a few-label user has no large validation set. For k = all, the standard tarski
protocol (full validation split, best epoch after half the schedule). The late-interaction loop also
trains >= 300 steps. Metrics on the test split: accuracy, macro-F1, ECE, NLL; on CLINC also oos AUROC and
FPR95 from the max probability (in-scope test vs the 1,000 oos test messages).

Usage:
  .venv/bin/python explore/lit_dm_kshot.py --smoke
  .venv/bin/python explore/lit_dm_kshot.py --dataset banking77 --out results/tarski/explore_kshot_banking77.json
  .venv/bin/python explore/lit_dm_kshot.py --dataset clinc150 --out results/tarski/explore_kshot_clinc150.json
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

import lit_dm_common as C  # noqa: E402
from tarski.train import FeatureCache, fit_temperature  # noqa: E402
from tarski.trunk import Trunk  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# Late interaction (FastFit-style) on frozen trunk token states
# ---------------------------------------------------------------------------------------------------

class LateInteraction(nn.Module):
    def __init__(self, hidden: int, dim: int = 128, trainable: bool = True):
        super().__init__()
        self.trainable = trainable
        self.norm = nn.LayerNorm(hidden, elementwise_affine=trainable)
        self.proj = nn.Linear(hidden, dim, bias=False) if trainable else None
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)), requires_grad=trainable)

    def embed(self, h: torch.Tensor) -> torch.Tensor:
        z = self.norm(h.float())
        if self.proj is not None:
            z = self.proj(z)
        return F.normalize(z, dim=-1)


class Bag:
    """All token states of the class documents (label names + support messages), flattened."""

    def __init__(self, cache: FeatureCache, depth: int, docs: Sequence[int], doc_class: Sequence[int],
                 n_classes: int, device):
        hs, cls, own = [], [], []
        for d, c in zip(docs, doc_class):
            h = cache.h[depth][d]
            hs.append(h)
            cls.append(torch.full((len(h),), int(c), dtype=torch.long))
            own.append(torch.full((len(h),), int(d), dtype=torch.long))
        self.h = torch.cat(hs).to(device)                      # [T, hidden] fp16
        self.cls = torch.cat(cls).to(device)                   # [T]
        self.owner = torch.cat(own).to(device)                 # [T] cache index of the document
        self.n_classes = n_classes

    def scores(self, model: LateInteraction, h: torch.Tensor, att: torch.Tensor,
               qidx: Optional[torch.Tensor] = None, bag_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[b, C] mean-over-query-tokens of the max cosine within each class bag."""
        b = bag_emb if bag_emb is not None else model.embed(self.h)          # [T, D]
        q = model.embed(h)                                                    # [b, L, D]
        sim = torch.einsum("bld,td->blt", q, b)                               # [b, L, T]
        if qidx is not None:                                                  # a message never matches itself
            sim = sim.masked_fill((self.owner[None, :] == qidx[:, None])[:, None, :], -2.0)
        B, L, T = sim.shape
        out = torch.full((B, L, self.n_classes), -2.0, device=sim.device, dtype=sim.dtype)
        out = out.scatter_reduce(2, self.cls.view(1, 1, T).expand(B, L, T), sim, reduce="amax", include_self=True)
        out = out.clamp_min(-1.0)
        m = att.to(out.dtype)[..., None]
        return (out * m).sum(1) / m.sum(1).clamp_min(1.0)


def li_train(model: LateInteraction, bag: Bag, cache: FeatureCache, depth: int, tr: Sequence[int],
             y: np.ndarray, epochs: int = 10, bs: int = 32, lr: float = 1e-3, min_steps: int = 300,
             seed: int = 0) -> Dict:
    dev = cache.trunk.device
    model.to(dev).train()
    per_epoch = (len(tr) + bs - 1) // bs
    epochs = max(epochs, -(-min_steps // per_epoch))
    steps = epochs * per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    t0, tot = time.time(), 0.0
    for ep in range(epochs):
        order = rng.permutation(len(tr))
        for s in range(0, len(order), bs):
            sel = [tr[j] for j in order[s:s + bs]]
            h, ctx = cache.batch(depth, sel)
            logits = model.log_scale.exp() * bag.scores(model, h, ctx.attention_mask,
                                                        torch.tensor(sel, device=dev))
            loss = F.cross_entropy(logits, torch.tensor(y[sel], device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
    model.eval()
    return {"opt_steps": steps, "epochs_run": epochs, "train_s": round(time.time() - t0, 1),
            "final_loss": tot / max(steps, 1)}


@torch.no_grad()
def li_logits(model: LateInteraction, bag: Bag, cache: FeatureCache, depth: int, idx: Sequence[int],
              bs: int = 64) -> torch.Tensor:
    model.to(cache.trunk.device).eval()
    b = model.embed(bag.h)
    out = torch.zeros(len(idx), bag.n_classes)
    for s in range(0, len(idx), bs):
        sel = list(idx[s:s + bs])
        h, ctx = cache.batch(depth, sel)
        out[s:s + len(sel)] = (model.log_scale.exp() * bag.scores(model, h, ctx.attention_mask, bag_emb=b)).float().cpu()
    return out


# ---------------------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------------------

def evaluate_logits(z_te: torch.Tensor, y_te: np.ndarray, z_va: Optional[torch.Tensor], y_va: Optional[np.ndarray],
                    z_oos: Optional[torch.Tensor]) -> Dict:
    t = 1.0
    if z_va is not None and len(y_va) >= 30:
        t = fit_temperature(z_va.float(), torch.tensor(y_va))
    p = torch.softmax(z_te.float() / t, -1).numpy()
    m = C.metrics(p, y_te)
    m["temperature"] = t
    if z_va is None:              # k = 0: no labelled data, the scale (hence ECE/NLL) is arbitrary
        m["ece"], m["nll"] = None, None
    if z_oos is not None and len(z_oos):
        p_oos = torch.softmax(z_oos.float() / t, -1).numpy()
        m["oos"] = C.oos_metrics(p.max(-1), p_oos.max(-1))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77", choices=["banking77", "clinc150"])
    ap.add_argument("--base", default=None, help="trunk base (default answerdotai/ModernBERT-base)")
    ap.add_argument("--ks", nargs="*", default=["0", "1", "5", "10", "25", "all"])
    ap.add_argument("--seeds", type=int, default=3, help="support-set seeds for k <= 10 (k=25: min(2, seeds); all: 1)")
    ap.add_argument("--arms", nargs="*", default=["proto", "li", "probe", "blocks", "full"])
    ap.add_argument("--probe-depth", type=int, default=8)
    ap.add_argument("--li-depth", type=int, default=8)
    ap.add_argument("--li-dim", type=int, default=128)
    ap.add_argument("--li-cap", type=int, default=25, help="max support messages per class in a bag")
    ap.add_argument("--block-split", type=int, default=None, help="default 14 (banking77) / 11 (clinc150)")
    ap.add_argument("--block-depth", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.device = args.device or "cpu"
        args.base = args.base or C.SMOKE_BASE
        args.ks = ["0", "1", "5", "all"]
        args.seeds = 1
        args.probe_depth = args.li_depth = 3
        args.block_split = args.block_split or 4
        args.li_dim = 32
        args.out = args.out or "results/tarski/explore_kshot_smoke.json"
    block_split = args.block_split or (14 if args.dataset == "banking77" else 11)
    log = C.Log(args.out)
    trunk = C.load_trunk(args.base, args.device)
    log(f"== k-shot curve on {args.dataset} | base {trunk.base} on {trunk.device} | arms {args.arms} | ks {args.ks}")

    ds = C.load_dataset_by_name(args.dataset)
    labels_all = ds.tasks["intent"].labels
    if args.smoke:
        keep = [i for i, l in enumerate(labels_all) if l != "oos"][:6]
        if args.dataset == "clinc150":
            keep = keep + [labels_all.index("oos")]
        ds = C.subsample(ds, 300, 60, 80, classes={"intent": keep})
    inscope = [i for i, l in enumerate(labels_all) if l != "oos"]
    if args.smoke:
        present = sorted({e.y["intent"] for e in ds.train + ds.val + ds.test})
        inscope = [i for i in inscope if i in present]
    remap = {old: new for new, old in enumerate(inscope)}
    labels = [labels_all[i] for i in inscope]
    allx = ds.train + ds.val + ds.test
    y_all = np.array([remap.get(e.y["intent"], -1) for e in allx])
    n_tr, n_va = len(ds.train), len(ds.val)
    split_of = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * len(ds.test))
    train_pool = [i for i in range(len(allx)) if split_of[i] == "train" and y_all[i] >= 0]
    val_pool = [i for i in range(len(allx)) if split_of[i] == "val" and y_all[i] >= 0]
    test_in = [i for i in range(len(allx)) if split_of[i] == "test" and y_all[i] >= 0]
    test_oos = [i for i in range(len(allx)) if split_of[i] == "test" and y_all[i] < 0]
    names = C.label_texts(labels)
    name_idx = list(range(len(allx), len(allx) + len(names)))
    y_all = np.r_[y_all, np.arange(len(names))]
    texts = [e.text for e in allx] + names

    depths = {args.probe_depth, args.li_depth, block_split}
    if "full" in args.arms:
        depths.add(0)
    cache = FeatureCache(trunk, texts, sorted(depths), ds.max_len)
    log(f"   {len(labels)} classes, train pool {len(train_pool)}, val pool {len(val_pool)}, test {len(test_in)} "
        f"in-scope + {len(test_oos)} oos; cached depths {sorted(depths)} in {cache.seconds:.1f}s "
        f"({cache.bytes() / 1e6:.0f} MB)")

    res = C.load_json(args.out) if not args.smoke else {}
    res.setdefault("config", {k: v for k, v in vars(args).items()})
    res["config"].update({"base": trunk.base, "block_split": block_split, "n_classes": len(labels),
                          "protocol": "k<all: k-shot val (min(k,10)/class) for temperature only, last epoch, "
                                      ">=300 steps; k=all: full val, best epoch after half the schedule"})
    runs = res.setdefault("runs", {})
    y_te = y_all[test_in]

    def record(key, m):
        runs[key] = m
        C.dump(res, args.out)

    for kstr in args.ks:
        k = None if kstr == "all" else int(kstr)
        n_seeds = 1 if k is None else (min(2, args.seeds) if k >= 25 else args.seeds)
        if k == 0:
            n_seeds = 1
        for seed in range(n_seeds):
            tag = f"k={kstr}|seed={seed}"
            tr = C.kshot(y_all, train_pool, k, seed) if k != 0 else []
            va = val_pool if k is None else (C.kshot(y_all, val_pool, min(k, 10), seed + 1000) if k else [])
            log(f"-- {tag}: {len(tr)} train, {len(va)} val")
            y_tr, y_va = y_all[tr], y_all[va]

            # prototypes (no training)
            if "proto" in args.arms and f"{tag}|proto" not in runs:
                d = args.probe_depth
                te_f = C.pooled(cache, d, test_in)
                oos_f = C.pooled(cache, d, test_oos) if test_oos else None
                nm_f = C.pooled(cache, d, name_idx)
                if k == 0:
                    center = nm_f.mean(0)
                    cents = {"proto_names": nm_f}
                else:
                    sup = C.pooled(cache, d, tr)
                    center = sup.mean(0)
                    supn = F.normalize(sup - center, dim=-1)
                    cent = torch.zeros(len(labels), sup.shape[1])
                    cent.index_add_(0, torch.tensor(y_tr), supn)
                    cents = {"proto": cent, "proto+name": cent + F.normalize(nm_f - center, dim=-1)}
                va_f = C.pooled(cache, d, va) if len(va) else None
                for name, cent in cents.items():
                    z = lambda f: C.cosine_logits(f, cent, center)
                    m = evaluate_logits(z(te_f), y_te, z(va_f) if va_f is not None else None, y_va,
                                        z(oos_f) if oos_f is not None else None)
                    record(f"{tag}|{name}", m)
                    log(f"      {name}: acc {m['acc']:.4f}" + (f", oos auroc {m['oos']['auroc']:.3f}" if 'oos' in m else ""))
                runs[f"{tag}|proto"] = True

            # late interaction
            if "li" in args.arms and f"{tag}|li_done" not in runs:
                d = args.li_depth
                docs, doc_cls = list(name_idx), list(range(len(labels)))
                if k != 0:
                    cap = C.kshot(y_all, tr, args.li_cap, seed) if (k is None or k > args.li_cap) else tr
                    docs += list(cap)
                    doc_cls += [int(y_all[i]) for i in cap]
                bag = Bag(cache, d, docs, doc_cls, len(labels), trunk.device)
                untrained = LateInteraction(trunk.hidden, trainable=False)
                zf = lambda mdl, idx: li_logits(mdl, bag, cache, d, idx) if len(idx) else None
                m = evaluate_logits(zf(untrained, test_in), y_te, zf(untrained, va) if k != 0 else None, y_va,
                                    zf(untrained, test_oos))
                m["bag_tokens"] = int(len(bag.h))
                record(f"{tag}|{'li_names' if k == 0 else 'li_untrained'}", m)
                log(f"      li_untrained: acc {m['acc']:.4f} ({len(bag.h)} bag tokens)")
                if k != 0:
                    model = LateInteraction(trunk.hidden, args.li_dim, trainable=True)
                    info = li_train(model, bag, cache, d, tr, y_all, seed=seed)
                    m = evaluate_logits(zf(model, test_in), y_te, zf(model, va), y_va, zf(model, test_oos))
                    m.update(info)
                    record(f"{tag}|li", m)
                    log(f"      li: acc {m['acc']:.4f}, {info['opt_steps']} steps, {info['train_s']}s")
                del bag
                runs[f"{tag}|li_done"] = True
                C.gpu_gc()

            # tarski branches and the full fine-tune
            for arm in ("probe", "blocks", "full"):
                if arm not in args.arms or k == 0 or f"{tag}|{arm}" in runs:
                    continue
                split = {"probe": args.probe_depth, "blocks": block_split, "full": 0}[arm]
                br = C.make(arm, split, args.block_depth, labels, trunk)
                info = C.run_branch(br, cache, tr, torch.tensor(y_tr), va, torch.tensor(y_va), arm, seed=seed,
                                    select=(k is None), log=log)
                p = C.probs_of(br, cache, test_in)
                m = C.metrics(p, y_te)
                if test_oos:
                    m["oos"] = C.oos_metrics(p.max(-1), C.probs_of(br, cache, test_oos).max(-1))
                m.update(info)
                m["params"] = sum(q.numel() for q in br.parameters())
                record(f"{tag}|{arm}", m)
                log(f"      {arm}: acc {m['acc']:.4f} macro-F1 {m['macro_f1']:.4f} ECE {m['ece']:.3f}"
                    + (f", oos auroc {m['oos']['auroc']:.3f}" if 'oos' in m else ""))
                del br
                C.gpu_gc()

    # summary: mean and std over seeds per (k, arm)
    summ: Dict[str, Dict] = {}
    for key, m in runs.items():
        if not isinstance(m, dict):
            continue
        kpart, _, arm = key.split("|")
        s = summ.setdefault(kpart, {}).setdefault(arm, {"acc": [], "macro_f1": [], "ece": [], "oos_auroc": []})
        s["acc"].append(m["acc"])
        s["macro_f1"].append(m["macro_f1"])
        if m.get("ece") is not None:
            s["ece"].append(m["ece"])
        if m.get("oos"):
            s["oos_auroc"].append(m["oos"]["auroc"])
    res["summary"] = {kp: {arm: {f"{met}_mean": float(np.mean(v)) if v else None for met, v in d.items()}
                           | {"acc_std": float(np.std(d["acc"])), "n_seeds": len(d["acc"])}
                           for arm, d in arms.items()} for kp, arms in summ.items()}
    C.dump(res, args.out)
    log("== summary (test accuracy, mean over seeds)")
    arms_seen = sorted({a for d in res["summary"].values() for a in d})
    log("   k      " + " ".join(f"{a:>12s}" for a in arms_seen))
    for kp in [f"k={k}" for k in args.ks]:
        row = res["summary"].get(kp, {})
        log(f"   {kp:6s} " + " ".join(f"{row[a]['acc_mean']:12.4f}" if a in row else f"{'-':>12s}" for a in arms_seen))
    log("done")


if __name__ == "__main__":
    main()
