"""Far-field idea 2: structure-following coarse-graining with per-decision field gates, on typed-decisions.

A typed-decisions message is a JSON state (~240 tokens), most of which is schema: keys, punctuation and
boilerplate identical across messages of a workflow. Mean pooling over tokens is a block-spin transform with
one block, and it washes out the few informative fields. Here the blocks follow the data's own structure:
the frozen trunk's token states are pooled per JSON field (key path truncated at depth 2), giving an F x 768
coarse state per message, and each decision reads it through a gain vector over fields (softmax of F learned
logits, the "dendritic gate": urgency listens to `alert.evidence`, duplicate to `invoice.number`).

Heads compared at each split depth, all trained on cached trunk states with soft-label cross-entropy,
full batch, fixed steps, no early stopping (the sweep's probes stop after 4-5 epochs because validation
accuracy on ~26 messages sits at the majority class during warm-up, so `mean` here is the fair probe):

  mean           standardise + linear on the mean-pooled message state (the fair probe baseline)
  fields_gate    per-task static softmax gate over field-pooled states, then linear
  fields_attn    input-dependent attention over fields (learned query), then linear
  fields_concat  linear on the concatenation of field-pooled states (strong weight decay)
  token_attn     input-dependent attention over tokens (learned query), then linear: the un-coarse-grained
                 control that separates "attention helps" from "field structure helps"

Baselines: majority class per task, the sweep's probe@k means when results/tarski/sweep_typed.json exists,
laya's full fine-tune 0.766 as the ceiling.

Usage:
  .venv/bin/python explore/farfield_coarsegrain.py --smoke
  .venv/bin/python explore/farfield_coarsegrain.py --out results/tarski/explore_coarsegrain.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, so `tarski` imports

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tarski import data
from tarski.train import FeatureCache, evaluate, fit_temperature
from tarski.trunk import Trunk

SYNTAX = "_syntax"
LAYA_FULL_FINETUNE = 0.766


def log(msg: str, f=None):
    print(msg, flush=True)
    if f is not None:
        f.write(msg + "\n")
        f.flush()


# ---------------------------------------------------------------------------------------------------
# JSON field spans: re-serialise the parsed object exactly as json.dumps did, tracking character positions
# ---------------------------------------------------------------------------------------------------

def field_spans(text: str, max_depth: int = 2) -> Optional[List[Tuple[str, int, int]]]:
    """(field path, char start, char end) for every node at `max_depth` or leaf above it. Keys are part of
    their field's span. Returns None if the text is not the canonical json.dumps of its parse."""
    try:
        obj = json.loads(text)
    except Exception:
        return None
    parts: List[str] = []
    pos = [0]
    spans: List[Tuple[str, int, int]] = []

    def write(s: str):
        parts.append(s)
        pos[0] += len(s)

    def walk(v, path: Tuple[str, ...], depth: int, start: int):
        if isinstance(v, dict):
            write("{")
            for i, (k, val) in enumerate(v.items()):
                if i:
                    write(", ")
                kstart = pos[0]
                write(json.dumps(k, ensure_ascii=False))
                write(": ")
                walk(val, path + (k,), depth + 1, kstart)
            write("}")
        elif isinstance(v, list):
            write("[")
            for i, val in enumerate(v):
                if i:
                    write(", ")
                walk(val, path + ("[]",), depth + 1, pos[0])
            write("]")
        else:
            write(json.dumps(v, ensure_ascii=False))
        container = isinstance(v, (dict, list))
        if depth >= 1 and (depth == max_depth or (not container and depth < max_depth)):
            spans.append((".".join(path), start, pos[0]))

    walk(obj, (), 0, 0)
    if "".join(parts) != text:
        return None
    return spans


def token_fields(text: str, offsets: Sequence[Tuple[int, int]], max_depth: int) -> List[str]:
    """Field name per token from character offsets; special tokens and structural text go to `_syntax`."""
    spans = field_spans(text, max_depth)
    if spans is None:
        return ["_all"] * len(offsets)
    starts = np.array([s for _, s, _ in spans])
    ends = np.array([e for _, _, e in spans])
    out = []
    for a, b in offsets:
        if b <= a:                                   # special token
            out.append(SYNTAX)
            continue
        hit = np.where((starts <= a) & (a < ends))[0]
        out.append(spans[hit[0]][0] if len(hit) else SYNTAX)
    return out


# ---------------------------------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------------------------------

class MeanHead(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.out = nn.Linear(d, c)

    def forward(self, batch):
        return self.out(batch["mean"])


class FieldsGate(nn.Module):
    """Static per-task softmax gate over fields (input-independent gain vector), masked by field presence."""

    def __init__(self, d, c, n_fields):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(n_fields))
        self.out = nn.Linear(d, c)

    def weights(self, present):
        g = self.gate[None].expand(present.shape[0], -1).masked_fill(~present, -1e4)
        return torch.softmax(g, -1)

    def forward(self, batch):
        a = self.weights(batch["present"])                     # (B, F)
        return self.out(torch.einsum("bf,bfd->bd", a, batch["fields"]))


class FieldsAttn(nn.Module):
    """Input-dependent attention over fields with one learned query."""

    def __init__(self, d, c, n_fields):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(d))
        self.out = nn.Linear(d, c)

    def forward(self, batch):
        s = (batch["fields"] @ self.q) / batch["fields"].shape[-1] ** 0.5
        a = torch.softmax(s.masked_fill(~batch["present"], -1e4), -1)
        return self.out(torch.einsum("bf,bfd->bd", a, batch["fields"]))


class FieldsConcat(nn.Module):
    def __init__(self, d, c, n_fields):
        super().__init__()
        self.out = nn.Linear(d * n_fields, c)

    def forward(self, batch):
        return self.out(batch["fields"].flatten(1))


class TokenAttn(nn.Module):
    """Input-dependent attention over tokens with one learned query (no coarse-graining)."""

    def __init__(self, d, c):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(d))
        self.out = nn.Linear(d, c)

    def forward(self, batch):
        h, m = batch["tokens"], batch["tok_mask"]
        s = (h @ self.q) / h.shape[-1] ** 0.5
        a = torch.softmax(s.masked_fill(~m, -1e4), -1)
        return self.out(torch.einsum("bl,bld->bd", a, h))


HEADS = ("mean", "fields_gate", "fields_attn", "fields_concat", "token_attn")


def make_head(name, d, c, n_fields):
    return {"mean": lambda: MeanHead(d, c), "fields_gate": lambda: FieldsGate(d, c, n_fields),
            "fields_attn": lambda: FieldsAttn(d, c, n_fields), "fields_concat": lambda: FieldsConcat(d, c, n_fields),
            "token_attn": lambda: TokenAttn(d, c)}[name]()


def train_head(head: nn.Module, batch: Dict, soft: torch.Tensor, steps: int, lr: float, wd: float, seed: int) -> List[float]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    hist = []
    for _ in range(steps):
        z = head(batch).float()
        loss = -(soft * F.log_softmax(z, -1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    return hist


# ---------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="CPU, ~30 messages per workflow, one depth")
    ap.add_argument("--device", default=None)
    ap.add_argument("--depths", type=int, nargs="*", default=[6, 11, 16, 22])
    ap.add_argument("--field-depth", type=int, default=2, help="JSON key-path depth for the coarse-graining blocks")
    ap.add_argument("--heads", nargs="*", default=list(HEADS))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--concat-wd", type=float, default=3e-2, help="weight decay for fields_concat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.device = args.device or "cpu"
        args.depths = args.depths if args.depths != [6, 11, 16, 22] else [6]
        args.steps = min(args.steps, 150)
        args.out = args.out or "results/tarski/explore_coarsegrain_smoke.json"
    else:
        args.out = args.out or "results/tarski/explore_coarsegrain.json"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logf = open(args.out.replace(".json", ".log"), "a")
    L = lambda m: log(m, logf)

    t_start = time.time()
    trunk = Trunk(device=args.device)
    dev = trunk.device
    ds = data.load_typed_decisions()
    workflows = sorted({t.split(".")[0] for t in ds.tasks})
    wf_of = lambda e: next(t.split(".")[0] for t in e.y)

    train, val, test = ds.train, ds.val, ds.test
    if args.smoke:
        rng = random.Random(args.seed)

        def take(rows, n):
            by = {}
            for e in rows:
                by.setdefault(wf_of(e), []).append(e)
            out = []
            for g in by.values():
                rng.shuffle(g)
                out += g[:n]
            return out

        train, val, test = take(train, 40), take(val, 8), take(test, 20)
    allx = train + val + test
    n_tr, n_va = len(train), len(val)
    rows = {"train": list(range(0, n_tr)), "val": list(range(n_tr, n_tr + n_va)), "test": list(range(n_tr + n_va, len(allx)))}
    L(f"== coarse-graining on {ds.name}: {n_tr}/{n_va}/{len(test)} train/val/test, depths {args.depths}, "
      f"field depth {args.field_depth}, heads {args.heads}, device {dev}, smoke={args.smoke}")

    # token -> field assignment (same tokenizer call as the cache: truncation at ds.max_len)
    texts = [e.text for e in allx]
    enc = trunk.tok(texts, truncation=True, max_length=ds.max_len, return_offsets_mapping=True)
    tok_fields = [token_fields(t, enc["offset_mapping"][i], args.field_depth) for i, t in enumerate(texts)]
    n_fallback = sum(f[0] == "_all" for f in tok_fields)
    # field vocabulary per workflow, from training messages; rare/unknown fields fold into _syntax
    vocab: Dict[str, List[str]] = {}
    for wf in workflows:
        names = {}
        for i in rows["train"]:
            if wf_of(allx[i]) == wf:
                for f in set(tok_fields[i]):
                    names[f] = names.get(f, 0) + 1
        vocab[wf] = sorted(f for f in names if f != SYNTAX) + [SYNTAX]
    L(f"  field vocabularies: " + ", ".join(f"{wf} {len(v)}" for wf, v in vocab.items()) +
      (f"; {n_fallback} messages fell back to a single field" if n_fallback else ""))
    schema_frac = float(np.mean([np.mean([f == SYNTAX for f in tf]) for tf in tok_fields]))
    L(f"  fraction of tokens that are schema/syntax: {schema_frac:.2f}")

    results: Dict = {"config": vars(args), "device": str(dev), "n": {k: len(v) for k, v in rows.items()},
                     "field_vocab": vocab, "schema_token_fraction": schema_frac, "laya_full_finetune": LAYA_FULL_FINETUNE,
                     "depths": {}}
    sweep_path = "results/tarski/sweep_typed.json"
    if os.path.exists(sweep_path):
        sw = json.load(open(sweep_path))
        results["sweep_reference_mean_acc"] = {k: round(v["mean"]["acc"], 4) for k, v in sw.items()}
        results["sweep_reference_mean_macro_f1"] = {k: round(v["mean"]["macro_f1"], 4) for k, v in sw.items()}

    # majority-class baseline per task
    majority = {}
    for task in ds.tasks:
        ytr = [allx[i].y[task] for i in rows["train"] if task in allx[i].y]
        yte = np.array([allx[i].y[task] for i in rows["test"] if task in allx[i].y])
        maj = int(np.bincount(ytr, minlength=len(ds.tasks[task].labels)).argmax())
        probs = np.zeros((len(yte), len(ds.tasks[task].labels)))
        probs[:, maj] = 1.0
        majority[task] = evaluate(probs, yte)
    results["majority"] = {"tasks": majority, "mean_acc": float(np.mean([m["acc"] for m in majority.values()])),
                           "mean_macro_f1": float(np.mean([m["macro_f1"] for m in majority.values()]))}
    L(f"  majority-class baseline: mean acc {results['majority']['mean_acc']:.4f}, macro-F1 {results['majority']['mean_macro_f1']:.4f}")

    for d in args.depths:
        t0 = time.time()
        cache = FeatureCache(trunk, texts, [d], ds.max_len)            # one trunk pass per depth, fp16 on CPU
        L(f"-- depth {d}: trunk cached in {cache.seconds:.1f}s ({cache.bytes() / 1e6:.0f} MB)")
        for i in range(len(allx)):
            assert cache.lengths[i] == len(tok_fields[i]), "tokenisation mismatch between cache and field spans"

        # pooled features per message: mean (768) and per-field means (F_wf x 768) + presence
        mean_feat = torch.stack([cache.h[d][i].float().mean(0) for i in range(len(allx))])
        field_feat: Dict[str, torch.Tensor] = {}
        present: Dict[str, torch.Tensor] = {}
        for wf in workflows:
            ids = [i for i in range(len(allx)) if wf_of(allx[i]) == wf]
            fidx = {f: j for j, f in enumerate(vocab[wf])}
            Fw = len(vocab[wf])
            feat = torch.zeros(len(allx), Fw, trunk.hidden)
            pres = torch.zeros(len(allx), Fw, dtype=torch.bool)
            for i in ids:
                h = cache.h[d][i].float()
                fid = torch.tensor([fidx.get(f, fidx[SYNTAX]) for f in tok_fields[i]])
                cnt = torch.bincount(fid, minlength=Fw).float()
                feat[i] = torch.zeros(Fw, trunk.hidden).index_add_(0, fid, h) / cnt.clamp_min(1)[:, None]
                pres[i] = cnt > 0
            field_feat[wf], present[wf] = feat, pres

        depth_res = {"heads": {}, "field_gates": {}}
        per_head_metrics: Dict[str, Dict[str, Dict]] = {h: {} for h in args.heads}
        for task in ds.tasks:
            wf = task.split(".")[0]
            labels = ds.tasks[task].labels
            sel = {s: [i for i in rows[s] if task in allx[i].y] for s in rows}
            y = {s: torch.tensor([allx[i].y[task] for i in sel[s]]) for s in sel}
            soft = {s: torch.tensor(np.stack([allx[i].soft[task] for i in sel[s]])).float() for s in sel}

            # standardisation with training statistics (per hidden dim; fields share the statistics)
            mu, sd = mean_feat[sel["train"]].mean(0), mean_feat[sel["train"]].std(0).clamp_min(1e-4)
            ff = field_feat[wf]
            pm = present[wf][sel["train"]].float()
            fmu = (ff[sel["train"]] * pm[..., None]).sum((0, 1)) / pm.sum()
            fsd = (((ff[sel["train"]] - fmu) ** 2 * pm[..., None]).sum((0, 1)) / pm.sum()).sqrt().clamp_min(1e-4)

            def batch_for(s):
                b = {"mean": ((mean_feat[sel[s]] - mu) / sd).to(dev),
                     "fields": (((ff[sel[s]] - fmu) / fsd) * present[wf][sel[s]][..., None]).to(dev),
                     "present": present[wf][sel[s]].to(dev)}
                if "token_attn" in args.heads:
                    Lmax = max(cache.lengths[i] for i in sel[s])
                    toks = torch.zeros(len(sel[s]), Lmax, trunk.hidden)
                    mask = torch.zeros(len(sel[s]), Lmax, dtype=torch.bool)
                    for j, i in enumerate(sel[s]):
                        toks[j, : cache.lengths[i]] = (cache.h[d][i].float() - mu) / sd
                        mask[j, : cache.lengths[i]] = True
                    b["tokens"], b["tok_mask"] = toks.to(dev), mask.to(dev)
                return b

            B = {s: batch_for(s) for s in sel}
            for hname in args.heads:
                head = make_head(hname, trunk.hidden, len(labels), len(vocab[wf])).to(dev)
                wd = args.concat_wd if hname == "fields_concat" else args.wd
                hist = train_head(head, B["train"], soft["train"].to(dev), args.steps, args.lr, wd, args.seed)
                head.eval()
                with torch.no_grad():
                    zv, zt = head(B["val"]).float().cpu(), head(B["test"]).float().cpu()
                T = fit_temperature(zv, y["val"], soft["val"]) if len(sel["val"]) >= 30 else 1.0   # the harness rule
                probs = torch.softmax(zt / T, -1).numpy()
                m = evaluate(probs, y["test"].numpy(), soft["test"].numpy())
                m.update({"val_acc": float((zv.argmax(-1) == y["val"]).float().mean()), "temperature": T,
                          "train_loss_final": hist[-1], "params": sum(p.numel() for p in head.parameters())})
                per_head_metrics[hname][task] = m
                if hname == "fields_gate":
                    with torch.no_grad():
                        w = torch.softmax(head.gate, -1).cpu().numpy()
                    top = np.argsort(-w)[:3]
                    depth_res["field_gates"][task] = [(vocab[wf][j], round(float(w[j]), 3)) for j in top]

        for hname in args.heads:
            ms = per_head_metrics[hname]
            mean = {k: float(np.mean([m[k] for m in ms.values()])) for k in ("acc", "macro_f1", "nll", "ece", "brier_soft")}
            by_wf = {wf: float(np.mean([m["acc"] for t, m in ms.items() if t.startswith(wf)])) for wf in workflows}
            depth_res["heads"][hname] = {"mean": mean, "by_workflow_acc": by_wf, "tasks": ms}
            L(f"   {hname:>14}: mean acc {mean['acc']:.4f}  macro-F1 {mean['macro_f1']:.4f}  NLL {mean['nll']:.4f}  "
              f"Brier {mean['brier_soft']:.4f} | by workflow " + " ".join(f"{wf.split('_')[0]} {a:.3f}" for wf, a in by_wf.items()))
        if "fields_gate" in args.heads:
            L("   top fields per task (fields_gate): " + "; ".join(
                f"{t.split('.')[1]}<-{g[0][0]}({g[0][1]})" for t, g in depth_res["field_gates"].items()))
        depth_res["seconds"] = round(time.time() - t0, 1)
        results["depths"][str(d)] = depth_res
        del cache, field_feat, mean_feat
        with open(args.out, "w") as f:                           # write after every depth (resumable reading)
            json.dump(results, f, indent=1, default=float)

    # summary table: head x depth mean accuracy
    summary = {h: {str(d): round(results["depths"][str(d)]["heads"][h]["mean"]["acc"], 4) for d in args.depths} for h in args.heads}
    results["summary_mean_acc"] = summary
    results["wall_seconds"] = round(time.time() - t_start, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    L(f"== done in {results['wall_seconds']}s; wrote {args.out}")
    L("   mean acc by head x depth: " + json.dumps(summary) + f" | majority {results['majority']['mean_acc']:.4f}"
      + (f" | sweep probes {results['sweep_reference_mean_acc']}" if "sweep_reference_mean_acc" in results else "")
      + f" | laya {LAYA_FULL_FINETUNE}")


if __name__ == "__main__":
    main()
