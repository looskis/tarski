"""Lit-scan idea 2: split off ModernBERT's massive channels, rotate, then quantise. Applied to the cached
trunk states that branches train on, to the trunk's keys/values, and to the trunk's own weights and
activations (an int8 trunk, end to end), on Banking77, CLINC150 and typed-decisions.

ModernBERT-base's residual stream carries a few fixed, input-independent massive channels (251, 254, 67,
... up to ~35,000 at depth 18; Sun et al., COLM 2024). A per-token int8 scale is then set by one channel
and every other channel is crushed. The fix: keep those channels in fp16, rotate the rest with a random
orthogonal matrix (QuaRot / TurboQuant style) and quantise per token.

Part A  outlier profile per depth (max |h|, 99.9th percentile, top channels).
Part B  cached states. Schemes (per-token symmetric absmax, dequantised back to fp16):
          fp16 | int8_plain | int8_split | int8_split_rot | int4_plain | int4_split | int4_split_rot |
          int3_split_rot
        (i)  serve-mismatch: branches trained on fp16 states, scored on quantised test states;
        (ii) train-on-quantised: branches retrained on quantised states (--retrain schemes).
        Branch kinds: probe and blocks:2. Also bytes per token.
Part C  quantised trunk, end to end (fp16-trained branches scored on test states produced by it):
          w8           weight-only int8, per output channel, every Linear in trunk layers [0, k)
          w4g64        weight-only int4, groups of 64 input channels
          w8a8         w8 + dynamic per-token int8 activations at every Linear input
          w8a8_split   w8a8, but each Linear's outlier input channels (calibrated on 64 training messages,
                       |x| > 6 as in LLM.int8, at most 2% of channels) stay in floating point
          kv4_kivi / kv2_kivi   keys per channel (over the message's tokens), values per token (KIVI)
          kv4_rot / kv3_rot     per-head random rotation, then per-token quantisation (TurboQuant-style)
        Reported: accuracy and agreement with the fp16 trunk's decisions.
Part D  (--cpu-latency, CPU only; run on the Mac): batch-1 latency of the fp32 trunk to depth k vs
        torch dynamic int8 (qnnpack/fbgemm) at 16, 64 and 240 tokens.

Prediction (lit_efficiency.md entry 2): int8_plain loses points at deep splits; split+rot int8 is
indistinguishable from fp16; split+rot 4-bit within ~0.5 (probe) / ~1 (blocks) points, mostly recovered
by training on quantised states; w8 and w8a8_split near-lossless, w8a8 (no split) visibly worse.

Usage:
  .venv/bin/python explore/lit_eff_quant.py --smoke
  .venv/bin/python explore/lit_eff_quant.py --out results/tarski/explore_lit_quant.json
  .venv/bin/python explore/lit_eff_quant.py --cpu-latency --out results/tarski/explore_lit_quant_cpu_mac.json
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_eff_common import (Log, cache_from_states, index, load_ds, make, save, task_rows, texts, threads, timeit,
                            train_eval, ys)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tarski.train import FeatureCache, predict_logits
from tarski.trunk import Trunk

DEFAULT_SPLITS = {"banking77": [8, 14], "clinc150": [11], "typed-decisions": [14]}
CACHE_SCHEMES = ["fp16", "int8_plain", "int8_split", "int8_split_rot", "int4_plain", "int4_split",
                 "int4_split_rot", "int3_split_rot"]
TRUNK_VARIANTS = ["w8", "w4g64", "w8a8", "w8a8_split", "kv4_kivi", "kv2_kivi", "kv4_rot", "kv3_rot"]


# ---------------------------------------------------------------------------------------------------
# Quantisers
# ---------------------------------------------------------------------------------------------------

def q_sym(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Symmetric absmax fake quantisation along `dim` (one scale per slice)."""
    L = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim, keepdim=True).clamp_min(1e-8) / L
    return torch.round(x / s).clamp(-L, L) * s


def rotation(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(n, n, generator=g))
    return Q * torch.sign(torch.diagonal(R))[None]


def outlier_channels(states: List[torch.Tensor], ratio: float = 8.0, cap: int = 8) -> List[int]:
    m = torch.stack([s.float().abs().amax(0) for s in states]).amax(0)
    med = m.median()
    order = torch.argsort(m, descending=True)
    return [int(c) for c in order[:cap] if m[c] > ratio * med]


class StateQuant:
    """Quantise-dequantise one message's (L, D) states under a scheme."""

    def __init__(self, scheme: str, D: int, outliers: List[int], seed: int = 0):
        self.scheme = scheme
        self.bits = None if scheme == "fp16" else int(scheme[3])
        self.split = "split" in scheme
        self.rot = "rot" in scheme
        keep = torch.ones(D, dtype=torch.bool)
        if self.split:
            keep[outliers] = False
        self.keep = keep
        self.m = int((~keep).sum())
        self.R = rotation(int(keep.sum()), seed) if self.rot else None

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        if self.bits is None:
            return h
        x = h.float()
        body = x[:, self.keep]
        if self.R is not None:
            body = q_sym(body @ self.R, self.bits) @ self.R.T
        else:
            body = q_sym(body, self.bits)
        out = x.clone()
        out[:, self.keep] = body
        return out.to(h.dtype)

    def bytes_per_token(self, D: int) -> float:
        if self.bits is None:
            return 2.0 * D
        return (D - self.m) * self.bits / 8 + 2 + 2 * self.m


def quantised_cache(trunk, fc: FeatureCache, k: int, sq: StateQuant) -> FeatureCache:
    return cache_from_states(trunk, {k: [sq(h) for h in fc.h[k]]})


def rel_err(fc_a: FeatureCache, fc_b: FeatureCache, k: int, rows, keep: torch.Tensor) -> float:
    num = den = 0.0
    for i in rows:
        a, b = fc_a.h[k][i].float()[:, keep], fc_b.h[k][i].float()[:, keep]
        num += float(((a - b) ** 2).sum())
        den += float((b ** 2).sum())
    return (num / max(den, 1e-12)) ** 0.5


# ---------------------------------------------------------------------------------------------------
# Quantised trunk
# ---------------------------------------------------------------------------------------------------

def quant_weight(W: torch.Tensor, bits: int, group: Optional[int]) -> torch.Tensor:
    if group is None:
        return q_sym(W, bits, dim=1)                                    # per output channel
    o, i = W.shape
    return q_sym(W.view(o, i // group, group), bits, dim=2).view(o, i)


class QLinear(nn.Module):
    def __init__(self, lin: nn.Linear, wbits: int, group: Optional[int], abits: Optional[int],
                 keep_fp: Optional[torch.Tensor]):
        super().__init__()
        self.register_buffer("weight", quant_weight(lin.weight.data.float(), wbits, group).to(lin.weight.dtype))
        self.bias = lin.bias
        self.abits = abits
        self.register_buffer("keep_fp", keep_fp if keep_fp is not None else torch.zeros(lin.in_features, dtype=torch.bool))

    def forward(self, x):
        if self.abits:
            body = x.masked_fill(self.keep_fp, 0)
            x = q_sym(body.float(), self.abits).to(x.dtype) + x * self.keep_fp.to(x.dtype)
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


LINEARS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")


@torch.no_grad()
def calibrate_inputs(trunk: Trunk, calib_texts: List[str], k: int, max_len: int) -> Dict[str, torch.Tensor]:
    """max |x| per input channel of every Linear in layers [0, k) over calibration messages."""
    mx: Dict[str, torch.Tensor] = {}
    hooks = []
    for i in range(k):
        for name in LINEARS:
            mod = trunk.model.layers[i].get_submodule(name)
            key = f"{i}.{name}"

            def hook(m, inp, out, key=key):
                v = inp[0].detach().float().abs().flatten(0, -2).amax(0).cpu()
                mx[key] = torch.maximum(mx[key], v) if key in mx else v
            hooks.append(mod.register_forward_hook(hook))
    b = trunk.tokenize(calib_texts, max_len)
    trunk.taps(b["input_ids"], b["attention_mask"], [k])
    for h in hooks:
        h.remove()
    return mx


def quantise_trunk(trunk: Trunk, variant: str, k: int, calib: Optional[Dict[str, torch.Tensor]] = None):
    """In-place: replace Linears in layers [0, k) (w*), or hook K/V (kv*)."""
    if variant.startswith("w"):
        wbits = int(variant[1])
        group = 64 if "g64" in variant else None
        abits = 8 if "a8" in variant else None
        for i in range(k):
            layer = trunk.model.layers[i]
            for name in LINEARS:
                parent, attr = name.split(".")
                lin = getattr(getattr(layer, parent), attr)
                keep = None
                if variant.endswith("split") and calib is not None:
                    mx = calib[f"{i}.{name}"]
                    cand = torch.nonzero(mx > 6.0).squeeze(1)
                    cap = max(1, int(0.02 * lin.in_features))
                    cand = cand[torch.argsort(mx[cand], descending=True)[:cap]]
                    keep = torch.zeros(lin.in_features, dtype=torch.bool)
                    keep[cand] = True
                    keep = keep.to(lin.weight.device)
                setattr(getattr(layer, parent), attr, QLinear(lin, wbits, group, abits, keep).to(lin.weight.device))
        return
    bits = int(variant[2])
    H = trunk.cfg.num_attention_heads
    D = trunk.hidden
    hd = D // H
    R = rotation(hd, 1).to(trunk.device)

    def hook(mod, inp, out):
        B, L, _ = out.shape
        qkv = out.float().view(B, L, 3, H, hd)
        kk, vv = qkv[:, :, 1], qkv[:, :, 2]
        if variant.endswith("kivi"):
            kk = q_sym(kk, bits, dim=1)                                  # per channel, over tokens
            vv = q_sym(vv, bits, dim=-1)                                 # per token
        else:
            kk = q_sym(kk @ R, bits, dim=-1) @ R.T
            vv = q_sym(vv @ R, bits, dim=-1) @ R.T
        qkv = torch.stack([qkv[:, :, 0], kk, vv], 2)
        return qkv.view(B, L, 3 * D).to(out.dtype)
    for i in range(k):
        trunk.model.layers[i].attn.Wqkv.register_forward_hook(hook)


# ---------------------------------------------------------------------------------------------------
# CPU latency (Part D)
# ---------------------------------------------------------------------------------------------------

@torch.no_grad()
def cpu_latency(k_list: List[int], lengths=(16, 64, 240), reps: int = 10) -> Dict:
    torch.backends.quantized.engine = torch.backends.quantized.supported_engines[-1]
    from torch.ao.quantization import quantize_dynamic
    trunk = Trunk(device="cpu", max_len=512)
    out = {"threads": torch.get_num_threads(), "engine": torch.backends.quantized.engine, "rows": []}
    qmodel = quantize_dynamic(copy.deepcopy(trunk.model), {nn.Linear}, dtype=torch.qint8)
    fp_model = trunk.model
    for L in lengths:
        x = torch.randint(5, 50000, (1, L))
        att = torch.ones_like(x)
        for k in k_list:
            row = {"tokens": L, "depth": k}
            for tag, model in (("fp32", fp_model), ("dyn_int8", qmodel)):
                trunk.model = model
                row[f"{tag}_ms"] = timeit(lambda: trunk.taps(x, att, [k]), torch.device("cpu"), reps=reps, warm=2)
            trunk.model = fp_model
            row["speedup"] = row["fp32_ms"] / row["dyn_int8_ms"]
            out["rows"].append(row)
    return out


# ---------------------------------------------------------------------------------------------------

def run(a, log):
    trunk = Trunk(device=a.device)
    res = {"args": vars(a), "datasets": {}}
    for name in a.datasets:
        ds = load_ds(name, a.smoke, workflows=a.typed_workflows if name == "typed-decisions" else None)
        allx, rng = index(ds)
        tasks = list(ds.tasks)[: (2 if a.smoke else None)]
        splits = a.splits or DEFAULT_SPLITS[name]
        if a.smoke:
            splits = splits[:1]
        fc_all = FeatureCache(trunk, texts(ds), splits, ds.max_len)
        log(f"== {name}: {len(allx)} messages, {len(tasks)} tasks, splits {splits}, cached in {fc_all.seconds:.0f}s")
        R = {}
        for k in splits:
            t0 = time.time()
            fc = cache_from_states(trunk, {k: fc_all.h[k]})
            out: Dict = {}
            # --- Part A: outlier profile ---
            sample = [fc.h[k][i].float() for i in rng["train"][:512]]
            flat = torch.cat(sample)
            chmax = flat.abs().amax(0)
            top = torch.argsort(chmax, descending=True)[:6]
            outl = outlier_channels(sample)
            out["profile"] = {"max_abs": float(chmax.max()), "p999_abs": float(flat.abs().flatten().kthvalue(int(0.999 * flat.numel())).values),
                              "top_channels": [int(c) for c in top], "top_values": [float(chmax[c]) for c in top],
                              "outlier_channels": outl}
            log(f"  split {k}: max |h| {out['profile']['max_abs']:.0f}, p99.9 {out['profile']['p999_abs']:.1f}, "
                f"top {out['profile']['top_channels'][:4]}, split-off channels {outl}")
            # --- Part B: cached states ---
            D = trunk.hidden
            sqs = {s: StateQuant(s, D, outl) for s in a.schemes}
            caches = {s: (fc if s == "fp16" else quantised_cache(trunk, fc, k, sq)) for s, sq in sqs.items()}
            keep_normal = torch.ones(D, dtype=torch.bool)
            keep_normal[outl] = False
            out["cache"] = {s: {"bytes_per_token": sqs[s].bytes_per_token(D),
                                "rel_err_normal_channels": rel_err(caches[s], fc, k, rng["test"][:256], keep_normal)}
                            for s in a.schemes}
            log("   rel. error on normal channels: " + " | ".join(f"{s} {v['rel_err_normal_channels']:.3f}"
                                                                 for s, v in out["cache"].items()))
            out["branches"] = {}
            base_branches = {}
            for kind in a.kinds:
                kname = kind.split(":")[0]
                per = {}
                for task in tasks:
                    sel = task_rows(allx, rng, task)
                    br = make(kind, k, ds.tasks[task].labels, trunk)
                    r = train_eval(br, fc, allx, sel, task, kname, min_steps=a.min_steps,
                                   epochs=a.probe_epochs if kname == "probe" else None,
                                   eval_caches={s: c for s, c in caches.items() if s != "fp16"})
                    base_branches[(kind, task)] = (br, sel)
                    base_pred = r["test_probs"].argmax(-1)
                    per[task] = {"fp16_train": {s: {"acc": r["test" if s == "fp16" else s]["acc"],
                                                    "agree_fp16": float((r[("test" if s == "fp16" else s) + "_probs"].argmax(-1) == base_pred).mean())}
                                                for s in a.schemes}}
                    for s in a.retrain:
                        if s == "fp16":
                            continue
                        br2 = make(kind, k, ds.tasks[task].labels, trunk)
                        r2 = train_eval(br2, caches[s], allx, sel, task, kname, min_steps=a.min_steps,
                                        epochs=a.probe_epochs if kname == "probe" else None)
                        per[task].setdefault("retrained", {})[s] = {"acc": r2["test"]["acc"]}
                    log(f"   [{kind} {task}] fp16 {per[task]['fp16_train']['fp16']['acc']:.4f} | serve on: " +
                        " ".join(f"{s} {v['acc']:.4f}" for s, v in per[task]["fp16_train"].items() if s != "fp16") +
                        (" | retrained: " + " ".join(f"{s} {v['acc']:.4f}" for s, v in per[task].get("retrained", {}).items())
                         if per[task].get("retrained") else ""))
                out["branches"][kind] = per
            # --- Part C: quantised trunk, end to end on the test split ---
            test_texts = [allx[i].text for i in rng["test"]]
            calib = None
            out["trunk"] = {}
            for variant in a.trunk_variants:
                qt = Trunk(device=a.device)
                if variant == "w8a8_split" and calib is None:
                    calib = calibrate_inputs(trunk, [allx[i].text for i in rng["train"][:64]], k, ds.max_len)
                quantise_trunk(qt, variant, k, calib)
                fq = FeatureCache(qt, test_texts, [k], ds.max_len)
                # a cache over allx whose test rows hold the quantised-trunk states
                states = list(fc.h[k])
                for j, i in enumerate(rng["test"]):
                    states[i] = fq.h[k][j]
                cq = cache_from_states(trunk, {k: states})
                row = {"rel_err_normal_channels": rel_err(cq, fc, k, rng["test"][:256], keep_normal)}
                for kind in a.kinds:
                    accs, agrees = [], []
                    for task in tasks:
                        br, sel = base_branches[(kind, task)]
                        z_ref = predict_logits(br, fc, sel["test"]).argmax(-1)
                        z_q = predict_logits(br, cq, sel["test"]).argmax(-1)
                        y_te = ys(allx, sel["test"], task)[0]
                        accs.append(float((z_q == y_te).float().mean()))
                        agrees.append(float((z_q == z_ref).float().mean()))
                    row[kind] = {"mean_acc": float(np.mean(accs)), "mean_agree_fp16": float(np.mean(agrees)),
                                 "per_task_acc": dict(zip(tasks, accs))}
                out["trunk"][variant] = row
                ref = {kind: float(np.mean([out["branches"][kind][t]["fp16_train"]["fp16"]["acc"] for t in tasks]))
                       for kind in a.kinds}
                log(f"   trunk {variant:10s}: rel err {row['rel_err_normal_channels']:.3f} | " + " | ".join(
                    f"{kind} acc {row[kind]['mean_acc']:.4f} (fp16 {ref[kind]:.4f}, agree {row[kind]['mean_agree_fp16']:.4f})"
                    for kind in a.kinds))
                del qt
            out["wall_s"] = round(time.time() - t0, 1)
            R[k] = out
            res["datasets"][name] = R
            save(res, a.out)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=["banking77", "clinc150", "typed-decisions"])
    ap.add_argument("--splits", type=int, nargs="*", default=None, help="override the per-dataset splits")
    ap.add_argument("--typed-workflows", nargs="*", default=["customer_service"])
    ap.add_argument("--kinds", nargs="*", default=["probe", "blocks:2"])
    ap.add_argument("--schemes", nargs="*", default=CACHE_SCHEMES)
    ap.add_argument("--retrain", nargs="*", default=["int8_plain", "int4_split_rot", "int3_split_rot"])
    ap.add_argument("--trunk-variants", nargs="*", default=TRUNK_VARIANTS)
    ap.add_argument("--probe-epochs", type=int, default=8)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--cpu-latency", action="store_true", help="Part D only (CPU)")
    ap.add_argument("--latency-depths", type=int, nargs="*", default=[8, 14, 22])
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.device, a.min_steps, a.probe_epochs = "cpu", 20, 1
        a.retrain = ["int4_split_rot"]
        a.trunk_variants = ["w8a8", "w8a8_split", "kv4_kivi", "kv3_rot"]
        a.latency_depths = [4]
        a.out = a.out or "results/tarski/explore_lit_quant_smoke.json"
    if not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    if a.cpu_latency or a.smoke:
        lat = cpu_latency(a.latency_depths, lengths=(16, 64) if a.smoke else (16, 64, 240), reps=3 if a.smoke else 10)
        for r in lat["rows"]:
            log(f"  CPU ({lat['engine']}, {lat['threads']} threads) {r['tokens']:3d} tokens, depth {r['depth']:2d}: "
                f"fp32 {r['fp32_ms']:.2f} ms, dynamic int8 {r['dyn_int8_ms']:.2f} ms ({r['speedup']:.2f}x)")
        if a.cpu_latency:
            save({"args": vars(a), "cpu_latency": lat}, a.out)
            return
    res = run(a, log)
    if a.smoke:
        res["cpu_latency"] = lat
    save(res, a.out)
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
