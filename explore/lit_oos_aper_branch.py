"""Entry 9 (lit_incremental_oos.md): adapt once, then add routes as statistics on the branch's own features.

  - Zhou et al., "Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and
    Adaptivity are All You Need" (SimpleCIL / APER, IJCV 2024, arXiv 2303.07338): adapt the model on the
    first session only, then build every later class's prototype from the concatenation of frozen and
    adapted features.
  - Zhou et al., "Expandable Subspace Ensemble" (EASE, CVPR 2024, arXiv 2403.12030): one light adapter per
    task, prototypes in every subspace.
  - Zheng, Qiu & Ma, "Learn or Recall?" (SEQ*, ACL 2024, arXiv 2312.07887): pretrained LMs forget little;
    freeze old classifier rows and train only the new ones.
A tarski blocks branch IS an adapted subspace. When a user adds routes after the branch was trained, this
tests whether the branch's features help the new routes without any gradient step on the branch.

Protocol (CLINC150 in-scope intents): session 1 = --first random intents; a blocks branch (layers
[split, split+depth) copied from the base, tarski.train.train_branch, >= 300 optimiser steps) is trained on
them. Later sessions add --step intents each, with no gradient steps on the branch. Heads for all seen
routes, hyperparameters chosen on session-1 validation data only:
  lda / klda on  trunk@split | trunk@22 | branch output (pooled) | [trunk@split ; branch] (APER) |
                 [trunk@22 ; branch]
  seq_newrows    the branch's own head for session-1 routes, frozen; new rows trained on each new session's
                 branch features only (no replay), 300 steps; new rows read features centred on the
                 session-1 mean and their biases start at the old rows' mean (without both, new rows fire on
                 every message and old routes collapse to 0 in the smoke run); seq_newrows_wa also rescales the new rows to
                 the old rows' mean norm (weight aligning, Zhao et al., CVPR 2020) against new-route bias
References: the branch itself on session-1 routes; a joint blocks branch trained on all 150 routes.
Metrics: accuracy on test messages of the routes seen so far, after each session; final accuracy;
average incremental accuracy; final accuracy split into session-1 routes and later routes.
Expected: the concatenation beats trunk-only by 1-2 points on later routes in domains the first session
covered, and roughly ties elsewhere.

Usage:
  .venv/bin/python explore/lit_oos_aper_branch.py --smoke
  .venv/bin/python explore/lit_oos_aper_branch.py --out results/tarski/explore_lit_aper_branch.json   # ~12 min on an A10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_oos_common import (RFF, Gauss, Logger, get_trunk, l2n, load_ds, median_sqdist, pick_device, pooled,
                            seed_all, split_index)

import numpy as np
import torch
import torch.nn.functional as F

from tarski.branches import mean_pool
from tarski.train import FeatureCache, make_branch, predict_logits, train_branch


@torch.no_grad()
def branch_features(branch, fc: FeatureCache, idx, bs: int = 128) -> torch.Tensor:
    """The branch's pooled output (after its layers and final norm, before dropout and the linear head)."""
    branch.eval()
    out = torch.zeros(len(idx), branch.hidden)
    order = sorted(range(len(idx)), key=lambda j: fc.lengths[idx[j]])
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        h, ctx = fc.batch(branch.split, [idx[j] for j in sel])
        hh = branch.norm(ctx.run(branch.layers, h))
        out[sel] = mean_pool(hh.float(), ctx.attention_mask).cpu()
    return out


def acc(z, y):
    return float((torch.as_tensor(z).argmax(-1).numpy() == y).mean())


def pick_head(X, y, tr1, va1, C, rff_dim, seed, dev):
    """LDA and KLDA hyperparameters from session-1 data only."""
    mu, sd = X[tr1].mean(0, keepdim=True), X[tr1].std(0, keepdim=True).clamp_min(1e-4)
    Z = (X - mu) / sd
    best_lda = None
    g0 = Gauss(Z[tr1], y[tr1], C, 0.1)
    for s in (1e-3, 1e-2, 0.1, 0.3):
        a = acc(g0.reshrink(s).lda_logits(Z[va1]), y[va1])
        if best_lda is None or a > best_lda[0]:
            best_lda = (a, s)
    Zl = l2n(Z)
    med = median_sqdist(Zl[tr1])
    best_k = None
    for c in (0.1, 0.25, 0.5, 1.0):
        rff = RFF(Zl.shape[1], rff_dim, c / max(med, 1e-9), seed)
        R = rff(Zl, dev)
        g = Gauss(R[tr1], y[tr1], C, 0.1)
        for s in (1e-2, 0.1, 0.3):
            a = acc(g.reshrink(s).lda_logits(R[va1]), y[va1])
            if best_k is None or a > best_k[0]:
                best_k = (a, c, s, R)
    return {"Z": Z, "lda_shrink": best_lda[1], "R": best_k[3], "klda_c": best_k[1], "klda_shrink": best_k[2]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--split", type=int, default=11)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--first", type=int, default=50)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--rff-dim", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--no-joint", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.split, args.depth, args.epochs, args.first, args.step, args.rff_dim = 4, 1, 1, 10, 5, 256
    out = args.out or ("results/tarski/explore_lit_aper_branch_smoke.json" if args.smoke
                       else "results/tarski/explore_lit_aper_branch.json")
    log = Logger(out)
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = pick_device(args.smoke)
    trunk = get_trunk(args.smoke)
    t0 = time.time()
    ds = load_ds("clinc150", args.smoke, args.seed)
    allx, idx = split_index(ds)
    intents = ds.tasks["intent"].labels
    oos_i = intents.index("oos")
    in_ids = [i for i in range(len(intents)) if i != oos_i]
    imap = {i: j for j, i in enumerate(in_ids)}
    y = np.array([imap.get(e.y["intent"], -1) for e in allx])
    C = 150
    order = rng.permutation(C)
    if args.smoke:
        order = order[: args.first + 2 * args.step]
    sessions = [order[: args.first]] + [order[s:s + args.step] for s in range(args.first, len(order), args.step)]
    used = np.concatenate(sessions)
    keep = np.isin(y, used)
    tr, va, te = [r[keep[r]] for r in (idx["train"], idx["val"], idx["test"])]
    s1 = sessions[0]
    tr1, va1 = tr[np.isin(y[tr], s1)], va[np.isin(y[va], s1)]
    log(f"== APER on forked branches | CLINC150, {len(sessions)} sessions ({args.first} + {args.step} x "
        f"{len(sessions) - 1}) | blocks@{args.split}+{args.depth} | trunk {trunk.device}")

    rows_all = np.concatenate([tr, va, te])
    fc = FeatureCache(trunk, [allx[i].text for i in rows_all], [args.split], ds.max_len)
    pos = {int(r): k for k, r in enumerate(rows_all)}                # dataset row -> cache index
    ci = lambda rows: [pos[int(r)] for r in rows]
    log(f"   cached trunk depth {args.split} for {len(rows_all)} messages in {fc.seconds:.1f}s")

    # session-1 branch (label space = session-1 routes)
    s1_map = {int(c): j for j, c in enumerate(s1)}
    y1 = lambda rows: torch.as_tensor([s1_map[int(v)] for v in y[rows]])
    br = make_branch("blocks", args.split, [intents[in_ids[c]] for c in s1], trunk, args.depth)
    t1 = time.time()
    info = train_branch(br, fc, ci(tr1), y1(tr1), None, ci(va1), y1(va1), epochs=args.epochs, seed=args.seed,
                        min_steps=args.steps)
    te1 = te[np.isin(y[te], s1)]
    br_acc_s1 = acc(predict_logits(br, fc, ci(te1)), y1(te1).numpy())
    log(f"   session-1 branch: {len(info['history'])} epochs, val acc {info['val_acc']:.4f}, test acc on session-1 "
        f"routes {br_acc_s1:.4f} ({time.time() - t1:.0f}s)")

    feats = {}
    P = pooled(trunk, [allx[i].text for i in rows_all], sorted({args.split, 22}), ds.max_len, log=log)
    B = branch_features(br, fc, list(range(len(rows_all))))
    feats[f"trunk@{args.split}"] = P[args.split]
    feats["trunk@22"] = P[22]
    feats["branch"] = B
    feats[f"concat_trunk@{args.split}_branch"] = torch.cat([P[args.split], B], 1)
    feats["concat_trunk@22_branch"] = torch.cat([P[22], B], 1)
    yc = y[rows_all]
    TR, VA, TE = np.array(ci(tr)), np.array(ci(va)), np.array(ci(te))
    TR1, VA1 = np.array(ci(tr1)), np.array(ci(va1))

    heads = {name: pick_head(X, yc, TR1, VA1, C, args.rff_dim, args.seed, dev) for name, X in feats.items()}
    for name, h in heads.items():
        log(f"   {name}: lda shrink {h['lda_shrink']}, klda c {h['klda_c']} shrink {h['klda_shrink']}")

    # incremental sessions
    curves = {f"{kind}:{name}": [] for name in feats for kind in ("lda", "klda")}
    curves["seq_newrows:branch"] = []
    curves["seq_newrows_wa:branch"] = []
    W = torch.zeros(B.shape[1], C)
    b = torch.zeros(C)
    W[:, torch.as_tensor(s1)] = br.out.weight.detach().cpu().T
    b[torch.as_tensor(s1)] = br.out.bias.detach().cpu()
    Wa, ba = W.clone(), b.clone()                       # weight-aligned copy (Zhao et al., CVPR 2020)
    seen = np.array([], dtype=int)
    for k, cls in enumerate(sessions):
        seen = np.concatenate([seen, cls])
        act = np.zeros(C, dtype=bool)
        act[seen] = True
        trs, tes = TR[np.isin(yc[TR], seen)], TE[np.isin(yc[TE], seen)]
        for name, h in heads.items():
            g = Gauss(h["Z"][trs], yc[trs], C, h["lda_shrink"])
            curves[f"lda:{name}"].append(acc(g.lda_logits(h["Z"][tes]), yc[tes]))
            gk = Gauss(h["R"][trs], yc[trs], C, h["klda_shrink"])
            curves[f"klda:{name}"].append(acc(gk.lda_logits(h["R"][tes]), yc[tes]))
        if k > 0:
            trn = TR[np.isin(yc[TR], cls)]
            posn = {int(c): i for i, c in enumerate(cls)}
            yn = torch.as_tensor([posn[int(v)] for v in yc[trn]])
            old = np.setdiff1d(seen, cls)
            Wn = torch.zeros(B.shape[1], len(cls), requires_grad=True)
            bn = torch.full((len(cls),), float(b[torch.as_tensor(old)].mean()))   # bias fixed at the old rows' mean
            opt = torch.optim.Adam([Wn], lr=1e-2)
            Xn = B[trn]
            mB = B[TR1].mean(0, keepdim=True)             # new rows read centred features (no common direction)
            for _ in range(max(300, args.steps)):
                logits = torch.cat([Xn @ W[:, old] + b[old], (Xn - mB) @ Wn + bn], 1)
                loss = F.cross_entropy(logits, yn + len(old)) + 1e-3 * (Wn ** 2).sum() / 2
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                # fold the centring into the bias: (x - m) W + b = x W + (b - m W)
                bn_eff = bn - (mB @ Wn.detach())[0]
                W[:, torch.as_tensor(cls)] = Wn.detach()
                b[torch.as_tensor(cls)] = bn_eff
                gam = float(Wa[:, torch.as_tensor(old)].norm(dim=0).mean() / Wn.detach().norm(dim=0).mean().clamp_min(1e-9))
                Wa[:, torch.as_tensor(cls)] = gam * Wn.detach()
                ba[torch.as_tensor(cls)] = bn - gam * (mB @ Wn.detach())[0]
        for key, (WW, bb) in (("seq_newrows:branch", (W, b)), ("seq_newrows_wa:branch", (Wa, ba))):
            z = B[tes] @ WW + bb
            z[:, ~act] = -1e9
            curves[key].append(acc(z, yc[tes]))
        log(f"-- session {k + 1}/{len(sessions)} ({len(seen)} routes): " + " | ".join(
            f"{n} {v[-1]:.4f}" for n, v in curves.items()))
    final_split = {}
    later = np.setdiff1d(used, s1)
    for n in curves:
        kind, name = n.split(":")
        if kind == "seq_newrows":
            z = B[TE] @ W + b
        elif kind == "seq_newrows_wa":
            z = B[TE] @ Wa + ba
        else:
            h = heads[name]
            X = h["Z"] if kind == "lda" else h["R"]
            z = Gauss(X[TR], yc[TR], C, h[f"{kind}_shrink"]).lda_logits(X[TE])
        z = torch.as_tensor(z).clone()
        z[:, ~np.isin(np.arange(C), used)] = -1e9
        pred = z.argmax(-1).numpy()
        final_split[n] = {"session1_routes": float((pred == yc[TE])[np.isin(yc[TE], s1)].mean()),
                          "later_routes": float((pred == yc[TE])[np.isin(yc[TE], later)].mean())}
    res = {"args": vars(args), "sessions": [s.tolist() for s in sessions], "branch_session1_test_acc": br_acc_s1,
           "curves": curves, "final": {n: v[-1] for n, v in curves.items()},
           "avg_incremental": {n: float(np.mean(v)) for n, v in curves.items()}, "final_split": final_split,
           "heads": {n: {k: v for k, v in h.items() if k not in ("Z", "R")} for n, h in heads.items()}}
    log("== final (all routes) | avg incremental | session-1 routes | later routes")
    for n in sorted(curves, key=lambda n: -curves[n][-1]):
        log(f"   {n:40s} {curves[n][-1]:.4f} | {res['avg_incremental'][n]:.4f} | "
            f"{final_split[n]['session1_routes']:.4f} | {final_split[n]['later_routes']:.4f}")
    log.dump(res)
    if not args.no_joint:
        t1 = time.time()
        brj = make_branch("blocks", args.split, [intents[in_ids[c]] for c in range(C)], trunk, args.depth)
        train_branch(brj, fc, ci(tr), torch.as_tensor(y[tr]), None, ci(va), torch.as_tensor(y[va]),
                     epochs=args.epochs, seed=args.seed, min_steps=args.steps)
        zj = predict_logits(brj, fc, ci(te)).clone()
        zj[:, ~np.isin(np.arange(C), used)] = -1e9
        res["joint_blocks_branch_test_acc"] = acc(zj, y[te])
        log(f"   reference: joint blocks branch on all {len(used)} routes: {res['joint_blocks_branch_test_acc']:.4f} "
            f"({time.time() - t1:.0f}s)")
    res["wall_s"] = round(time.time() - t0, 1)
    log.dump(res)
    log(f"done in {res['wall_s']}s -> {out}")


if __name__ == "__main__":
    main()
