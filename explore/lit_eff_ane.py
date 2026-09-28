"""Lit-scan idea 6 (Mac only): the trunk on the Apple Neural Engine via Core ML, branches on the GPU/CPU.

The trunk (embeddings + layers [0, k)) is re-implemented with static shapes (the math of tarski/fused.py:
pre-norm attention with RoPE, GeGLU MLP; masks and rotary tables baked in per sequence-length bucket),
checked against Trunk.taps, and converted with coremltools to an ML Program (fp16 activations; weights
fp16 or int8). HF's ModernBERT attention module does not convert (an `int` cast on a tensor), hence the
re-implementation.

Measured on this Mac:
  latency   batch-1 Core ML predict per compute unit (CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, ALL) per bucket,
            against the PyTorch trunk on MPS and on CPU
  parity    Banking77 test set: relative error of the Core ML states against the PyTorch trunk (all
            channels and without the massive channels), NaN/inf, and the accuracy / decision agreement of
            branches trained on PyTorch states (probe@k, blocks:2@k) when fed Core ML states
  fused     does moving the trunk off the GPU change "fused branches do not help on the Mac GPU"? Per
            message, N = 1/5/20 branches: loop vs tarski FusedBlocks vs SharedSVF (lit_eff_svf), with the
            trunk on MPS vs on the ANE; plus a two-thread pipeline (ANE trunk for message i+1 while the GPU
            runs message i's branches)
  energy    only if `sudo -n powermetrics` is allowed (usually not)

Usage (Mac):
  .venv/bin/python explore/lit_eff_ane.py --smoke
  .venv/bin/python explore/lit_eff_ane.py --out results/tarski/explore_lit_ane_mac.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
import queue
import warnings
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_eff_common import Log, index, load_ds, make, save, task_rows, texts, threads, timeit, train_eval

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tarski.branches import BlockBranch
from tarski.fused import FusedBlocks
from tarski.train import FeatureCache, autocast, predict_logits
from tarski.trunk import Trunk

warnings.simplefilter("ignore")


class ANETrunk(nn.Module):
    """Embeddings + base layers [0, k) at a fixed length L. Inputs: ids (1, L) int32, mask (1, L) int32."""

    def __init__(self, trunk: Trunk, k: int, L: int):
        super().__init__()
        m, cfg = trunk.model, trunk.cfg
        self.D, self.H = trunk.hidden, cfg.num_attention_heads
        self.hd, self.I, self.eps, self.L = self.D // self.H, cfg.intermediate_size, cfg.norm_eps, L
        cp = lambda t: t.detach().float().cpu().clone()
        self.register_buffer("emb", cp(m.embeddings.tok_embeddings.weight))
        self.register_buffer("emb_norm", cp(m.embeddings.norm.weight))
        self.types = [l.attention_type for l in m.layers[:k]]
        self.has_attn_norm = [not isinstance(l.attn_norm, nn.Identity) for l in m.layers[:k]]
        for i, l in enumerate(m.layers[:k]):
            if self.has_attn_norm[i]:
                self.register_buffer(f"an{i}", cp(l.attn_norm.weight))
            self.register_buffer(f"qkv{i}", cp(l.attn.Wqkv.weight))
            self.register_buffer(f"o{i}", cp(l.attn.Wo.weight))
            self.register_buffer(f"mn{i}", cp(l.mlp_norm.weight))
            self.register_buffer(f"wi{i}", cp(l.mlp.Wi.weight))
            self.register_buffer(f"wo{i}", cp(l.mlp.Wo.weight))
        x = torch.zeros(1, L, self.D)
        pos = torch.arange(L)[None]
        for t in set(self.types):
            c, s = m.rotary_emb(x, pos, t)
            self.register_buffer(f"cos_{t}", c[0][None, None].float().cpu().clone())       # (1, 1, L, hd)
            self.register_buffer(f"sin_{t}", s[0][None, None].float().cpu().clone())
        i = torch.arange(L)
        self.register_buffer("band", ((i[:, None] - i[None, :]).abs() <= cfg.sliding_window).float()[None, None])

    def _rot(self, x):
        h = self.hd // 2                      # a Python constant: traced shape arithmetic does not convert
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def _ln(self, x, w):
        return F.layer_norm(x, (self.D,), w, None, self.eps)

    def forward(self, ids, mask):
        L, D, H, hd = self.L, self.D, self.H, self.hd
        h = self._ln(F.embedding(ids, self.emb), self.emb_norm)
        key = mask.to(h.dtype)[:, None, None, :]
        madd = {"full_attention": (1.0 - key) * -1e4,
                "sliding_attention": (1.0 - key * self.band) * -1e4}
        for i, t in enumerate(self.types):
            a = self._ln(h, getattr(self, f"an{i}")) if self.has_attn_norm[i] else h
            qkv = (a @ getattr(self, f"qkv{i}").T).reshape(1, L, 3, H, hd)
            q = qkv[:, :, 0].permute(0, 2, 1, 3)
            k = qkv[:, :, 1].permute(0, 2, 1, 3)
            v = qkv[:, :, 2].permute(0, 2, 1, 3)
            cos, sin = getattr(self, f"cos_{t}"), getattr(self, f"sin_{t}")
            q = q * cos + self._rot(q) * sin
            k = k * cos + self._rot(k) * sin
            s = (q @ k.transpose(-1, -2)) * (hd ** -0.5) + madd[t]
            o = (torch.softmax(s, -1) @ v).permute(0, 2, 1, 3).reshape(1, L, D)
            h = h + o @ getattr(self, f"o{i}").T
            ig = self._ln(h, getattr(self, f"mn{i}")) @ getattr(self, f"wi{i}").T
            h = h + (F.gelu(ig[..., : self.I]) * ig[..., self.I:]) @ getattr(self, f"wo{i}").T
        return h


def convert(trunk: Trunk, k: int, L: int, int8: bool, work: str):
    import coremltools as ct
    m = ANETrunk(trunk, k, L).eval()
    ids = torch.full((1, L), 5, dtype=torch.int32)
    mask = torch.ones(1, L, dtype=torch.int32)
    tr = torch.jit.trace(m, (ids, mask))
    ml = ct.convert(tr, inputs=[ct.TensorType(name="ids", shape=(1, L), dtype=np.int32),
                                ct.TensorType(name="mask", shape=(1, L), dtype=np.int32)],
                    outputs=[ct.TensorType(name="h")], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
    if int8:
        import coremltools.optimize.coreml as cto
        cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
        ml = cto.linear_quantize_weights(ml, config=cfg)
    path = os.path.join(work, f"trunk_k{k}_L{L}_{'int8' if int8 else 'fp16'}.mlpackage")
    ml.save(path)
    return path


class CoreMLTrunk:
    def __init__(self, paths: Dict[int, str], unit: str):
        import coremltools as ct
        self.models = {L: ct.models.MLModel(p, compute_units=getattr(ct.ComputeUnit, unit)) for L, p in paths.items()}
        self.buckets = sorted(paths)

    def states(self, ids: List[int]) -> np.ndarray:
        n = len(ids)
        L = next(b for b in self.buckets if b >= n)
        x = np.zeros((1, L), np.int32)
        x[0, :n] = ids
        msk = np.zeros((1, L), np.int32)
        msk[0, :n] = 1
        return self.models[L].predict({"ids": x, "mask": msk})["h"][0, :n]


def powermetrics_ok() -> bool:
    try:
        return subprocess.run(["sudo", "-n", "powermetrics", "-n", "1", "-i", "100", "--samplers", "ane_power"],
                              capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=int, nargs="*", default=[8, 14])
    ap.add_argument("--buckets", type=int, nargs="*", default=[32, 64])
    ap.add_argument("--units", nargs="*", default=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"])
    ap.add_argument("--parity-unit", default="CPU_AND_NE")
    ap.add_argument("--n-test", type=int, default=0, help="test messages for parity (0 = all)")
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--work", default=None, help="directory for .mlpackage files (default: a temp dir)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.splits, a.buckets, a.units, a.n_test, a.min_steps = [4], [32, 64], ["CPU_ONLY", "CPU_AND_NE"], 40, 20
        a.out = a.out or "results/tarski/explore_lit_ane_smoke.json"
    if not a.out:
        ap.error("--out is required outside --smoke")
    log = Log(a.out)
    threads()
    work = a.work or tempfile.mkdtemp(prefix="tarski_coreml_")
    os.makedirs(work, exist_ok=True)
    dev = torch.device("mps") if torch.backends.mps.is_available() and not a.smoke else torch.device("cpu")
    trunk_cpu = Trunk(device="cpu")
    trunk = Trunk(device=str(dev)) if dev.type != "cpu" else trunk_cpu
    res = {"args": vars(a), "device": str(dev), "powermetrics": powermetrics_ok(), "splits": {}}
    log(f"== ANE trunk | branches on {dev} | powermetrics via sudo: {res['powermetrics']} | work dir {work}")
    ds = load_ds("banking77", a.smoke, n_smoke=(300, 60, 60))
    allx, rng = index(ds)
    test_rows = rng["test"][: a.n_test] if a.n_test else rng["test"]
    for k in a.splits:
        out: Dict = {}
        # --- re-implementation check ---
        b = trunk_cpu.tokenize(["my card was charged twice for the same purchase, please refund"], 32)
        n = b["input_ids"].shape[1]
        ids = torch.zeros(1, 32, dtype=torch.int32)
        msk = torch.zeros(1, 32, dtype=torch.int32)
        ids[0, :n], msk[0, :n] = b["input_ids"][0].int(), 1
        with torch.no_grad():
            ref = trunk_cpu.taps(b["input_ids"], b["attention_mask"], [k])[0][k][0]
            mine = ANETrunk(trunk_cpu, k, 32)(ids, msk)[0, :n]
        out["reimpl_rel_err"] = float((mine - ref).norm() / ref.norm())
        log(f"  split {k}: static re-implementation vs Trunk.taps rel err {out['reimpl_rel_err']:.1e}")
        assert out["reimpl_rel_err"] < 1e-4
        # --- conversion ---
        paths = {}
        for int8 in (False, True):
            t0 = time.time()
            paths[int8] = {L: convert(trunk_cpu, k, L, int8, work) for L in a.buckets}
            log(f"  converted {'int8' if int8 else 'fp16'}-weight models for buckets {a.buckets} in {time.time() - t0:.0f}s")
        # --- latency ---
        tok_ids = trunk_cpu.token_ids([allx[i].text for i in test_rows], ds.max_len)
        lat = {}
        for L in a.buckets:
            sample = next((t for t in tok_ids if L // 2 < len(t) <= L), tok_ids[0][:L])
            x = torch.tensor([sample])
            att = torch.ones_like(x)
            row = {"tokens": len(sample)}
            row["torch_cpu_ms"] = timeit(lambda: trunk_cpu.taps(x, att, [k]), torch.device("cpu"), reps=20)
            if dev.type == "mps":
                xm, am = x.to(dev), att.to(dev)
                with autocast(dev):
                    row["torch_mps_ms"] = timeit(lambda: trunk.taps(xm, am, [k]), dev, reps=20)
            for int8 in (False, True):
                for unit in a.units:
                    cm = CoreMLTrunk({L: paths[int8][L]}, unit)
                    cm.states(sample)
                    row[f"coreml_{'int8' if int8 else 'fp16'}_{unit}_ms"] = timeit(lambda: cm.states(sample), torch.device("cpu"), reps=30)
            lat[L] = row
            log(f"   L={L} ({len(sample)} tokens): " + " | ".join(f"{kk} {v:.2f}" for kk, v in row.items() if kk.endswith("_ms")))
        out["latency_ms"] = lat
        out["load_avg_after_latency"] = os.getloadavg()
        # --- parity on the Banking77 test set ---
        fc = FeatureCache(trunk, texts(ds), [k], ds.max_len)
        sel = task_rows(allx, rng, "intent")
        sel_eval = {**sel, "test": [i for i in sel["test"] if i in set(test_rows)]}
        branches = {}
        for kind in (["probe", "blocks:2"] if k == a.splits[0] else ["probe"]):
            br = make(kind, k, ds.tasks["intent"].labels, trunk)
            r = train_eval(br, fc, allx, sel_eval, "intent", kind.split(":")[0], min_steps=a.min_steps,
                           epochs=8 if kind == "probe" else 6)
            branches[kind] = (br, r["test"]["acc"])
            log(f"   trained {kind}@{k} on PyTorch ({dev}) states: test acc {r['test']['acc']:.4f} ({r['train_s']:.0f}s)")
        keep = torch.ones(trunk.hidden, dtype=torch.bool)
        keep[[251, 254, 67, 142, 689, 606]] = False
        out["parity"] = {}
        for int8 in (False, True):
            cm = CoreMLTrunk(paths[int8], a.parity_unit)
            t0 = time.time()
            states = list(fc.h[k])
            num = numk = den = denk = 0.0
            bad = 0
            for i, tid in zip(test_rows, tok_ids):
                s = torch.tensor(cm.states(tid)).float()
                bad += int(~torch.isfinite(s).all())
                ref = fc.h[k][i].float()
                num += float(((s - ref) ** 2).sum()); den += float((ref ** 2).sum())
                numk += float(((s - ref)[:, keep] ** 2).sum()); denk += float((ref[:, keep] ** 2).sum())
                states[i] = torch.nan_to_num(s).half()
            from lit_eff_common import cache_from_states
            cq = cache_from_states(trunk, {k: states})
            row = {"rel_err_all": (num / den) ** 0.5, "rel_err_normal_channels": (numk / denk) ** 0.5,
                   "messages_with_nonfinite": bad, "seconds": round(time.time() - t0, 1)}
            for kind, (br, acc_ref) in branches.items():
                zr = predict_logits(br, fc, sel_eval["test"]).argmax(-1)
                zq = predict_logits(br, cq, sel_eval["test"]).argmax(-1)
                y = torch.tensor([allx[i].y["intent"] for i in sel_eval["test"]])
                row[kind] = {"acc_torch": float((zr == y).float().mean()), "acc_coreml": float((zq == y).float().mean()),
                             "agreement": float((zr == zq).float().mean())}
            out["parity"]["int8" if int8 else "fp16"] = row
            log(f"   parity {'int8' if int8 else 'fp16'} weights on {a.parity_unit}: rel err {row['rel_err_all']:.4f} "
                f"(normal channels {row['rel_err_normal_channels']:.4f}), non-finite msgs {bad} | " +
                " | ".join(f"{kind} acc {v['acc_coreml']:.4f} vs {v['acc_torch']:.4f} (agree {v['agreement']:.4f})"
                           for kind, v in row.items() if isinstance(v, dict)))
        res["splits"][k] = out
        save(res, a.out)
    # --- fused-branches question: trunk on MPS vs on ANE ---
    if dev.type == "mps" or a.smoke:
        from lit_eff_svf import SharedSVF, build as svf_build, SVFLinear
        k, L = a.splits[0], a.buckets[0]
        cm = CoreMLTrunk({L: convert(trunk_cpu, k, L, False, work)}, "CPU_AND_NE")
        sample = trunk_cpu.token_ids(["hey can someone on billing look at the double charge for acme, they're pretty upset"], 64)[0]
        x = torch.tensor([sample], device=dev)
        att = torch.ones_like(x)
        labels = [f"l{i}" for i in range(20)]
        fused_rows = []
        with torch.no_grad(), autocast(dev):
            for N in ((1, 5) if a.smoke else (1, 5, 20)):
                full = [BlockBranch(k, labels, trunk.hidden, 2, trunk).to(dev).eval() for _ in range(N)]
                fb = FusedBlocks(full).to(dev)
                svf = [svf_build("svf_norm", k, labels, trunk).to(dev).eval() for _ in range(N)]
                sh = SharedSVF(svf).to(dev)
                heads = {"loop": lambda h, c: [br.logits(h, c) for br in full], "fused": lambda h, c: fb(h, c),
                         "shared_svf": lambda h, c: sh(h, c)}
                row = {"n": N}
                for name, fn in heads.items():
                    def mps_all():
                        taps, ctx = trunk.taps(x, att, [k])
                        return fn(taps[k], ctx)

                    def ane_then_gpu():
                        h = torch.from_numpy(cm.states(sample)).to(dev)[None].float()
                        return fn(h, trunk.context(h, att))
                    row[f"mps_trunk+{name}_ms"] = timeit(mps_all, dev, reps=20)
                    row[f"ane_trunk+{name}_ms"] = timeit(ane_then_gpu, dev, reps=20)
                # two-thread pipeline over a stream of messages (ANE trunk | GPU branches, best head)
                M = 40 if a.smoke else 200
                qq: "queue.Queue" = queue.Queue(maxsize=4)

                def producer():
                    for _ in range(M):
                        qq.put(cm.states(sample))
                    qq.put(None)
                t0 = time.perf_counter()
                th = threading.Thread(target=producer)
                th.start()
                while True:
                    s = qq.get()
                    if s is None:
                        break
                    h = torch.from_numpy(s).to(dev)[None].float()
                    fb(h, trunk.context(h, att))
                torch.mps.synchronize() if dev.type == "mps" else None
                th.join()
                row["pipelined_ane+fused_msgs_per_s"] = M / (time.perf_counter() - t0)
                t0 = time.perf_counter()
                for _ in range(M):
                    taps, ctx = trunk.taps(x, att, [k])
                    fb(taps[k], ctx)
                torch.mps.synchronize() if dev.type == "mps" else None
                row["sequential_mps+fused_msgs_per_s"] = M / (time.perf_counter() - t0)
                fused_rows.append(row)
                log(f"   N={N:2d}: " + " | ".join(f"{kk} {v:.2f}" for kk, v in row.items() if kk != "n"))
                del full, fb, svf, sh
        res["fused"] = {"split": k, "bucket": L, "rows": fused_rows, "load_avg": os.getloadavg()}
    save(res, a.out)
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
