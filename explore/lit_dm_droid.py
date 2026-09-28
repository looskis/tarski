"""Out-of-scope detection without out-of-scope labels, DROID-style (lit_decision_models #4).

DROID (Rashwan et al., arXiv 2510.14110, 2025) trains a small head on frozen encoders with an extra "none of the
above" class built from two free sources, and picks the rejection threshold on in-scope validation data only:
  - synthetic feature-space outliers: convex combinations of embeddings of two different known classes;
  - open-domain negatives: SQuAD 2.0 questions.
Users defining their own Slack routes never label "none of the above", so this is the setting tarski needs.

Here, on tarski's frozen trunk (branch = probe or blocks, K known classes + 1 "none"):
  base        K-way, no negatives; out-of-scope score = 1 - max probability (MSP)
  mixup       + interpolated pooled features of two different-class messages in the batch, labelled "none"
              (lambda ~ U(0.3, 0.7); mixing happens after pooling, before the linear head)
  squad       + SQuAD 2.0 questions run through the trunk once (cached), labelled "none"
  droid       mixup + squad
  supervised  reference that uses out-of-scope labels: on CLINC the 250 real oos training messages as "none";
              on banking77-open the held-out intents' training messages (an oracle ceiling, not a deployable arm)
Scores for K+1 arms: 1 - p(none) and the max known-class probability; threshold keeps 95% of in-scope validation.

Data:
  clinc150        train on the 150 in-scope intents (oos training messages removed); test = 4,500 in-scope +
                  1,000 oos.
  banking77-open  75% of the 77 intents known (seeded), the rest held out as unknown (DROID's protocol).
Training: custom loop mirroring tarski.train.train_branch's schedule (AdamW, one-cycle, >= 300 optimiser steps,
last epoch kept). Negatives are added as small extra batches (--neg-bs per source per step).

Usage:
  .venv/bin/python explore/lit_dm_droid.py --smoke
  .venv/bin/python explore/lit_dm_droid.py --out results/tarski/explore_droid.json
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "0"          # SQuAD 2.0 is a first download on the GPU box
os.environ["HF_DATASETS_OFFLINE"] = "0"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Optional, Sequence  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import lit_dm_common as C  # noqa: E402
from tarski.branches import BlockBranch, mean_pool  # noqa: E402
from tarski.train import FeatureCache  # noqa: E402
from tarski.trunk import Trunk  # noqa: E402


def pooled_rep(branch, h, ctx):
    if isinstance(branch, BlockBranch):
        return mean_pool(branch.norm(ctx.run(branch.layers, h)), ctx.attention_mask)
    return mean_pool(branch.norm(h), ctx.attention_mask)


def head(branch, z):
    return branch.out(branch.drop(z))


def train_kplus1(branch, cache: FeatureCache, tr: Sequence[int], y: np.ndarray, K: int, use_mix: bool,
                 neg: Optional[tuple], epochs: int, lr_layers: float, lr_head: float, bs: int = 32, neg_bs: int = 8,
                 seed: int = 0, min_steps: int = 300) -> Dict:
    """y in [0, K] (K = none). neg = (neg_cache, neg_idx) or None."""
    dev = cache.trunk.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    branch.to(dev).train()
    per_epoch = (len(tr) + bs - 1) // bs
    epochs = max(epochs, -(-min_steps // per_epoch))
    steps = epochs * per_epoch
    layer_p = [p for n, p in branch.named_parameters() if n.startswith("layers.")]
    head_p = [p for n, p in branch.named_parameters() if not n.startswith("layers.")]
    groups = [{"params": head_p, "lr": lr_head}] + ([{"params": layer_p, "lr": lr_layers}] if layer_p else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    none = torch.tensor(K, device=dev)
    t0, tot = time.time(), {"real": 0.0, "mix": 0.0, "neg": 0.0}
    for ep in range(epochs):
        order = sorted(range(len(tr)), key=lambda j: cache.lengths[tr[j]] + rng.random() * 8)
        chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
        rng.shuffle(chunks)
        for sel in chunks:
            ids = [tr[j] for j in sel]
            yy = torch.tensor(y[ids], device=dev)
            h, ctx = cache.batch(branch.split, ids)
            z = pooled_rep(branch, h, ctx)
            loss = F.cross_entropy(head(branch, z).float(), yy)
            tot["real"] += loss.item()
            if use_mix and len(ids) > 1:
                perm = torch.randperm(len(ids), device=dev)
                ok = (yy != yy[perm]) & (yy < K) & (yy[perm] < K)
                if ok.any():
                    a = ok.nonzero().squeeze(1)[:neg_bs]
                    lam = torch.empty(len(a), 1, device=dev).uniform_(0.3, 0.7)
                    zm = lam * z[a] + (1 - lam) * z[perm[a]]
                    lm = F.cross_entropy(head(branch, zm).float(), none.expand(len(a)))
                    loss = loss + lm
                    tot["mix"] += lm.item()
            if neg is not None:
                ncache, nidx = neg
                pick = [nidx[i] for i in rng.choice(len(nidx), size=min(neg_bs, len(nidx)), replace=False)]
                hn, ctxn = ncache.batch(branch.split, pick)
                ln = F.cross_entropy(head(branch, pooled_rep(branch, hn, ctxn)).float(), none.expand(len(pick)))
                loss = loss + ln
                tot["neg"] += ln.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
            opt.step()
            sched.step()
    branch.eval()
    return {"opt_steps": steps, "epochs_run": epochs, "train_s": round(time.time() - t0, 1),
            "mean_loss": {k: v / steps for k, v in tot.items()}}


@torch.no_grad()
def probs(branch, cache: FeatureCache, idx: Sequence[int], bs: int = 128) -> np.ndarray:
    branch.eval()
    out = []
    for s in range(0, len(idx), bs):
        h, ctx = cache.batch(branch.split, list(idx[s:s + bs]))
        out.append(torch.softmax(head(branch, pooled_rep(branch, h, ctx)).float(), -1).cpu())
    return torch.cat(out).numpy()


def squad_questions(n: int, seed: int) -> List[str]:
    from datasets import load_dataset
    qs = sorted(set(load_dataset("rajpurkar/squad_v2", split="train")["question"]))
    rng = np.random.default_rng(seed)
    return [qs[i].strip() for i in rng.choice(len(qs), size=min(n, len(qs)), replace=False)]


def prepare(name: str, seed: int, smoke: bool):
    """Returns texts, y (known class index, -1 = out of scope), split array, known label names, extra oos
    training indices (CLINC's real oos messages, for the supervised reference)."""
    if name == "clinc150":
        ds = C.load_dataset_by_name("clinc150")
        labs = ds.tasks["intent"].labels
        if smoke:
            keep = [i for i, l in enumerate(labs) if l != "oos"][:6] + [labs.index("oos")]
            ds = C.subsample(ds, 400, 80, 120, classes={"intent": keep})
        known = [i for i, l in enumerate(labs) if l != "oos"]
    else:
        ds = C.load_dataset_by_name("banking77")
        labs = ds.tasks["intent"].labels
        if smoke:
            ds = C.subsample(ds, 400, 80, 120, classes={"intent": list(range(8))})
        present = sorted({e.y["intent"] for e in ds.train})
        rng = np.random.default_rng(seed)
        known = sorted(rng.permutation(present)[: int(round(0.75 * len(present)))].tolist())
    if smoke:
        present = {e.y["intent"] for e in ds.train}
        known = [k for k in known if k in present]
    remap = {o: j for j, o in enumerate(known)}
    allx = ds.train + ds.val + ds.test
    split = np.array(["train"] * len(ds.train) + ["val"] * len(ds.val) + ["test"] * len(ds.test))
    y = np.array([remap.get(e.y["intent"], -1) for e in allx])
    return [e.text for e in allx], y, split, [labs[k] for k in known], ds.max_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["clinc150", "banking77-open"])
    ap.add_argument("--arms", nargs="*", default=["base", "mixup", "squad", "droid", "supervised"])
    ap.add_argument("--probe-depths", type=int, nargs="*", default=[4, 22])
    ap.add_argument("--blocks", type=int, nargs=2, default=[11, 2], metavar=("SPLIT", "DEPTH"))
    ap.add_argument("--n-squad", type=int, default=2000)
    ap.add_argument("--neg-bs", type=int, default=8)
    ap.add_argument("--probe-epochs", type=int, default=10)
    ap.add_argument("--block-epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.device = args.device or "cpu"
        args.base = args.base or C.SMOKE_BASE
        args.probe_depths, args.blocks, args.n_squad = [3, 7], [4, 2], 64
        args.probe_epochs, args.block_epochs = 1, 1
        args.out = args.out or "results/tarski/explore_droid_smoke.json"
    log = C.Log(args.out)
    trunk = C.load_trunk(args.base, args.device)
    res = C.load_json(args.out) if not args.smoke else {}
    res.setdefault("config", vars(args) | {"base": trunk.base})
    res.setdefault("datasets", {})
    squad = squad_questions(args.n_squad, args.seed)
    log(f"== DROID-style oos on {args.datasets} | base {trunk.base} on {trunk.device} | {len(squad)} SQuAD negatives")

    for name in args.datasets:
        texts, y, split, known, max_len = prepare(name, args.seed, args.smoke)
        K = len(known)
        depths = sorted(set(args.probe_depths) | {args.blocks[0]})
        cache = FeatureCache(trunk, texts, depths, max_len)
        ncache = FeatureCache(trunk, squad, depths, max_len)
        tr = [i for i in range(len(texts)) if split[i] == "train" and y[i] >= 0]
        tr_oos = [i for i in range(len(texts)) if split[i] == "train" and y[i] < 0]
        va = [i for i in range(len(texts)) if split[i] == "val" and y[i] >= 0]
        te_in = [i for i in range(len(texts)) if split[i] == "test" and y[i] >= 0]
        te_oos = [i for i in range(len(texts)) if split[i] == "test" and y[i] < 0]
        log(f"== {name}: {K} known classes, {len(tr)} train, {len(va)} in-scope val, test {len(te_in)} in-scope + "
            f"{len(te_oos)} out-of-scope ({len(tr_oos)} labelled oos train messages, used only by 'supervised')")
        out = res["datasets"].setdefault(name, {"known": known, "n": {"train": len(tr), "val": len(va),
                                                                      "test_in": len(te_in), "test_oos": len(te_oos)}})
        yk = y.copy()
        yk[yk < 0] = K                                   # "none" index for the supervised reference
        configs = [("probe", d, 0) for d in args.probe_depths] + [("blocks", args.blocks[0], args.blocks[1])]
        for kind, sp, dp in configs:
            cname = f"probe@{sp}" if kind == "probe" else f"blocks@{sp}+{dp}"
            for arm in args.arms:
                if arm == "supervised" and not tr_oos:
                    continue
                key = f"{cname}|{arm}"
                if key in out:
                    continue
                n_out = K if arm == "base" else K + 1
                br = C.make(kind, sp, dp, [str(i) for i in range(n_out)], trunk)
                lr_l, lr_h = C.LR[kind]
                ep = args.probe_epochs if kind == "probe" else args.block_epochs
                rows = tr + (tr_oos if arm == "supervised" else [])
                info = train_kplus1(br, cache, rows, yk, K, use_mix=arm in ("mixup", "droid"),
                                    neg=(ncache, list(range(len(squad)))) if arm in ("squad", "droid") else None,
                                    epochs=ep, lr_layers=lr_l, lr_head=lr_h, neg_bs=args.neg_bs, seed=args.seed)
                P = {s: probs(br, cache, ids) for s, ids in (("val", va), ("in", te_in), ("oos", te_oos))}
                m = {"train": info, "inscope_acc": float((P["in"][:, :K].argmax(-1) == y[te_in]).mean())}
                if n_out == K:
                    scores = {"msp": lambda p: p.max(-1)}
                else:
                    scores = {"1-p_none": lambda p: 1 - p[:, K], "max_known": lambda p: p[:, :K].max(-1)}
                for sname, fn in scores.items():
                    thr = float(np.quantile(fn(P["val"]), 0.05))
                    o = C.oos_metrics(fn(P["in"]), fn(P["oos"]), thr)
                    # accuracy over all test messages when rejected messages count as "out of scope"
                    acc_in = ((P["in"][:, :K].argmax(-1) == y[te_in]) & (fn(P["in"]) >= thr)).sum()
                    o["acc_all_with_reject"] = float((acc_in + (fn(P["oos"]) < thr).sum()) / (len(te_in) + len(te_oos)))
                    m[sname] = o
                out[key] = m
                best = max((v for k2, v in m.items() if isinstance(v, dict) and "auroc" in v), key=lambda v: v["auroc"])
                log(f"   {key:28s} in-scope acc {m['inscope_acc']:.4f} | best oos auroc {best['auroc']:.3f} "
                    f"fpr95 {best['fpr95']:.3f} oos F1 {best.get('oos_f1', float('nan')):.3f} "
                    f"({info['opt_steps']} steps, {info['train_s']}s)")
                C.dump(res, args.out)
                del br
                C.gpu_gc()
        del cache, ncache
        C.gpu_gc()
    log("done")


if __name__ == "__main__":
    main()
