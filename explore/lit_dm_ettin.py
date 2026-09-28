"""Paired encoder vs decoder trunks (Ettin) and a causal trunk with bidirectional branches (lit_decision_models #7).

Ettin (Weller et al., "Seq vs Seq", ICLR 2026) trained encoders and decoders with identical data, recipe and
shapes. jhu-clsp/ettin-encoder-150m has the ModernBERT architecture (22 layers, hidden 768) and loads into
tarski's Trunk unchanged; jhu-clsp/ettin-decoder-150m is a ModernBertDecoderModel of the same shape. That makes
the controlled question answerable: does a frozen-trunk decision harness need bidirectional attention?

A decoder trunk would give read-once for free (a message's key/value cache can be reused and questions appended
as a suffix), which is what the one-way laya work retrofits onto an encoder. The twist tested here: a *causal*
frozen trunk whose *branch layers* (copies of decoder layers [k, k+d)) run with a bidirectional mask.

Arms (all branches trained with tarski.train.train_branch, >= 300 optimiser steps, standard selection):
  enc            encoder trunk: probes (mean pool) at --depths, blocks@S+2 (mean pool)
  dec_causal     decoder trunk, causal everywhere: probes with mean and last-token pooling; blocks@S+2 with
                 causal branch layers, mean and last-token pooling
  dec_bibranch   decoder trunk (causal), branch layers bidirectional (their attention modules get is_causal=False
                 and bidirectional masks), mean pool
Datasets: Banking77, CLINC150 (151-way intent; oos AUROC from p(oos)), typed-decisions customer_service
(5 decisions over long JSON states: where a causal trunk should hurt most, since early fields never see later
ones).

The CausalTrunk subclass lives in this script (tarski/ is not modified): it builds causal masks for the trunk
pass and either causal or bidirectional masks for branches (FeatureCache.batch calls trunk.context).

Usage:
  .venv/bin/python explore/lit_dm_ettin.py --smoke
  .venv/bin/python explore/lit_dm_ettin.py --arms enc --out results/tarski/explore_ettin_enc.json
  .venv/bin/python explore/lit_dm_ettin.py --arms dec_causal dec_bibranch --out results/tarski/explore_ettin_dec.json
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "0"          # Ettin checkpoints are a first download on the GPU box
os.environ["HF_DATASETS_OFFLINE"] = "0"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from typing import Dict, Iterable, List, Optional, Tuple  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.masking_utils import (create_bidirectional_mask, create_bidirectional_sliding_window_mask,  # noqa: E402
                                        create_causal_mask, create_sliding_window_causal_mask)

import lit_dm_common as C  # noqa: E402
from tarski.branches import BlockBranch, ProbeBranch  # noqa: E402
from tarski.train import FeatureCache  # noqa: E402
from tarski.trunk import Context, Trunk  # noqa: E402


class CausalTrunk(Trunk):
    """A frozen decoder (ModernBertDecoderModel) as the trunk. The trunk pass is causal; `branch_mode` sets
    whether branches reading cached states see causal or bidirectional masks."""

    def __init__(self, base: str, device: Optional[str] = None, max_len: int = 256):
        super().__init__(base, device, max_len)
        for i, layer in enumerate(self.model.layers):
            layer.attention_type = self.cfg.layer_types[i]      # the encoder layers carry this; the decoder's do not
        self.branch_mode = "causal"

    def fingerprint(self) -> str:
        h = hashlib.sha256(self.base.encode())
        h.update(json.dumps(self.cfg.to_dict(), sort_keys=True, default=str).encode())
        return h.hexdigest()[:16]

    def _context(self, h: torch.Tensor, attention_mask: torch.Tensor, causal: bool) -> Context:
        pos = torch.arange(h.shape[1], device=h.device)[None]
        if causal:
            kw = {"config": self.cfg, "inputs_embeds": h, "attention_mask": attention_mask, "past_key_values": None,
                  "position_ids": pos.expand(h.shape[0], -1)}
            masks = {"full_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw)}
        else:
            kw = {"config": self.cfg, "inputs_embeds": h, "attention_mask": attention_mask}
            masks = {"full_attention": create_bidirectional_mask(**kw),
                     "sliding_attention": create_bidirectional_sliding_window_mask(**kw)}
        rope = {t: self.model.rotary_emb(h, pos, t) for t in set(self.cfg.layer_types)}
        return Context(masks, rope, attention_mask)

    def context(self, h: torch.Tensor, attention_mask: torch.Tensor) -> Context:
        return self._context(h, attention_mask, causal=(self.branch_mode == "causal"))

    @torch.no_grad()
    def taps(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
             depths: Iterable[int]) -> Tuple[Dict[int, torch.Tensor], Context]:
        want = sorted(set(int(d) for d in depths))
        h = self.model.embeddings(input_ids=input_ids)
        ctx = self._context(h, attention_mask, causal=True)
        out = {0: h} if 0 in want else {}
        for i in range(want[-1]):
            h = ctx.run([self.model.layers[i]], h)
            if i + 1 in want:
                out[i + 1] = h
        if self.n_layers in out:
            out[self.n_layers] = self.model.final_norm(out[self.n_layers])
        return out, ctx


@torch.no_grad()
def check_matches_stock(trunk: CausalTrunk, log) -> float:
    """The layer loop must reproduce the stock decoder forward (causal, right-padded batch)."""
    enc = trunk.tokenize(["a short one", "a somewhat longer message to check the causal trunk"])
    taps, _ = trunk.taps(enc["input_ids"], enc["attention_mask"], [trunk.n_layers])
    ref = trunk.model(**enc).last_hidden_state
    m = enc["attention_mask"].bool()
    err = float((taps[trunk.n_layers][m] - ref[m]).abs().max())
    log(f"   causal trunk vs stock forward: max abs diff {err:.2e}")
    if err > 1e-3:
        raise RuntimeError("CausalTrunk does not reproduce the stock decoder forward")
    return err


def last_token(h: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
    idx = att.sum(1).clamp_min(1) - 1
    return h[torch.arange(h.shape[0], device=h.device), idx]


class LastProbe(ProbeBranch):
    def logits(self, h, ctx):
        return self.out(self.drop(last_token(self.norm(h), ctx.attention_mask)))


class LastBlocks(BlockBranch):
    def logits(self, h, ctx):
        return self.out(self.drop(last_token(self.norm(ctx.run(self.layers, h)), ctx.attention_mask)))


def make_branch(kind: str, pool: str, split: int, depth: int, labels, trunk: Trunk, bidirectional: bool):
    if kind == "probe":
        return (LastProbe if pool == "last" else ProbeBranch)(split, list(labels), trunk.hidden)
    br = (LastBlocks if pool == "last" else BlockBranch)(split, list(labels), trunk.hidden, depth, trunk)
    for layer in br.layers:
        if hasattr(layer, "attn") and hasattr(layer.attn, "is_causal"):
            layer.attn.is_causal = not bidirectional
    return br


def run_dataset(name: str, trunk: Trunk, arm: str, args, log, res: Dict) -> None:
    ds = C.load_dataset_by_name(name)
    if name == "clinc150":
        ds = C.restrict_tasks(ds, ["intent"], "clinc150")
    if args.smoke:
        if name == "typed-cs":
            ds = C.restrict_tasks(ds, list(ds.tasks)[:2], "typed-cs")
            ds = C.subsample(ds, 60, 40, 40)
            ds.max_len = 128
        else:
            labs = ds.tasks["intent"].labels
            keep = [i for i, l in enumerate(labs) if l != "oos"][:6] + ([labs.index("oos")] if "oos" in labs else [])
            ds = C.subsample(ds, 240, 60, 80, classes={"intent": keep})
    n = trunk.n_layers
    depths = [d for d in args.depths if d <= n]
    split = min(args.block_split.get(name, 11), n - 2)
    allx = ds.train + ds.val + ds.test
    split_of = np.array(["train"] * len(ds.train) + ["val"] * len(ds.val) + ["test"] * len(ds.test))
    cache = FeatureCache(trunk, [e.text for e in allx], sorted(set(depths) | {split}), ds.max_len)
    log(f"== [{arm}] {ds.summary()} | cached {sorted(set(depths) | {split})} in {cache.seconds:.1f}s")
    out = res.setdefault(arm, {}).setdefault(name, {})

    if arm == "enc":
        configs = [("probe", "mean", d, 0, False) for d in depths] + [("blocks", "mean", split, 2, True)]
    elif arm == "dec_causal":
        configs = [("probe", p, d, 0, False) for d in depths for p in ("mean", "last")]
        configs += [("blocks", p, split, 2, False) for p in ("last", "mean")]
    else:
        configs = [("blocks", "mean", split, 2, True)]
    for kind, pool, sp, dp, bidir in configs:
        key = (f"probe@{sp}" if kind == "probe" else f"blocks@{sp}+{dp}") + f"|{pool}" + ("|bidir" if (kind == "blocks" and bidir) else "")
        if key in out:
            continue
        if isinstance(trunk, CausalTrunk):
            trunk.branch_mode = "bidirectional" if bidir else "causal"
        per = {}
        for t in ds.tasks:
            ids = {s: [i for i in range(len(allx)) if split_of[i] == s and t in allx[i].y] for s in ("train", "val", "test")}
            y = {s: np.array([allx[i].y[t] for i in ids[s]]) for s in ids}
            soft = None
            if all(t in allx[i].soft for i in ids["train"]):
                soft = torch.tensor(np.stack([allx[i].soft[t] for i in ids["train"]]))
            br = make_branch(kind, pool, sp, dp, ds.tasks[t].labels, trunk, bidir)
            info = C.run_branch(br, cache, ids["train"], torch.tensor(y["train"]), ids["val"], torch.tensor(y["val"]),
                                kind, soft_tr=soft)
            p = C.probs_of(br, cache, ids["test"])
            m = C.metrics(p, y["test"])
            m.update({k: info[k] for k in ("opt_steps", "epochs_run", "train_s")})
            labs = ds.tasks[t].labels
            if t == "intent" and "oos" in labs:
                o = labs.index("oos")
                ins = y["test"] != o
                m["oos_via_p_oos"] = C.oos_metrics(1 - p[ins, o], 1 - p[~ins, o])
            per[t] = m
            del br
            C.gpu_gc()
        out[key] = {"mean_acc": float(np.mean([v["acc"] for v in per.values()])),
                    "mean_macro_f1": float(np.mean([v["macro_f1"] for v in per.values()])), "tasks": per}
        log(f"   [{arm}] {name} {key}: mean acc {out[key]['mean_acc']:.4f}")
        C.dump(res, args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="jhu-clsp/ettin-encoder-150m")
    ap.add_argument("--decoder", default="jhu-clsp/ettin-decoder-150m")
    ap.add_argument("--arms", nargs="*", default=["enc", "dec_causal", "dec_bibranch"])
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150", "typed-cs"])
    ap.add_argument("--depths", type=int, nargs="*", default=[6, 11, 16, 22])
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.block_split = {"banking77": 14, "clinc150": 11, "typed-cs": 14}
    if args.smoke:
        args.device = args.device or "cpu"
        args.encoder, args.decoder = "jhu-clsp/ettin-encoder-17m", "jhu-clsp/ettin-decoder-17m"
        args.depths = [3, 7]
        args.block_split = {k: 4 for k in args.block_split}
        args.datasets = ["banking77", "typed-cs"]
        args.out = args.out or "results/tarski/explore_ettin_smoke.json"
    log = C.Log(args.out)
    res = C.load_json(args.out) if not args.smoke else {}
    res.setdefault("config", {k: v for k, v in vars(args).items()})
    trunks = {}
    for arm in args.arms:
        kind = "enc" if arm == "enc" else "dec"
        if kind not in trunks:
            for k in list(trunks):
                del trunks[k]
            C.gpu_gc()
            trunks[kind] = C.load_trunk(args.encoder, args.device) if kind == "enc" else C.load_trunk(args.decoder, args.device, CausalTrunk)
            t = trunks[kind]
            log(f"== {kind} trunk {t.base} on {t.device}: {t.n_layers} layers, hidden {t.hidden}")
            if kind == "dec":
                res["config"]["stock_forward_max_abs_diff"] = check_matches_stock(t, log)
        for name in args.datasets:
            run_dataset(name, trunks[kind], arm, args, log, res)
            C.gpu_gc()
    C.dump(res, args.out)
    log("done")


if __name__ == "__main__":
    main()
