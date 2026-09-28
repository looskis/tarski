"""What an extra decision costs, and what switching tasks costs.

Per-message latency (batch 1) for N decisions about one message:
  separate models   N full passes of the base (what N fine-tuned copies cost)
  probe@k           one trunk pass to depth k + N probes
  blocks@k+d        one trunk pass to depth k + N branches of d layers
  mixed             branches at different splits share one trunk pass to the deepest split

Task switching: time to load a branch file from disk vs a full fine-tuned copy of the base.
Weights are random-initialised branches; latency does not depend on training.

Usage: python -m experiments.latency --device mps --out results/tarski/latency_mps.json
"""

import argparse
import json
import os
import statistics
import tempfile
import time

import torch

from tarski.branches import Branch, BlockBranch, ProbeBranch
from tarski.fused import FusedBlocks
from tarski.train import autocast
from tarski.trunk import Trunk

SHORT = "hey can someone on billing look at the double charge for acme, they're pretty upset"
LONG = ("Subject: Production outage in eu-west after the 14:05 deploy. " * 40)[:2400]


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, dev, reps=20, warm=3):
    for _ in range(warm):
        fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--ns", type=int, nargs="*", default=[1, 2, 3, 5, 10, 20])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-laya", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    trunk = Trunk(device=a.device, max_len=512)
    dev, L = trunk.device, trunk.n_layers
    res = {"device": str(dev), "base": trunk.base, "layers": L, "per_message_ms": {}, "switch_ms": {}, "bytes": {}}
    labels = [f"l{i}" for i in range(20)]
    laya_agent = None
    if not a.no_laya:
        try:   # question-in-input decision model: re-reads the message once per question
            import laya
            laya_agent = laya.load("convaiinnovations/laya", device=str(dev))
            res["laya"] = "convaiinnovations/laya (ModernBERT-large cross-encoder, 421M), Agent.predict"
        except Exception as e:  # keep the benchmark running without it
            print("laya unavailable:", e, flush=True)
    laya_qs = {f"q{i}": {"type": "choice", "instructions": f"Which of these fits best (question {i})?",
                         "criteria": {"billing": "charges and refunds", "outage": "service down",
                                      "feature": "requests", "other": "anything else"}} for i in range(20)}
    for name, text in (("short", SHORT), ("long", LONG)):
        b = trunk.tokenize([text], 512)
        ids, att = b["input_ids"], b["attention_mask"]
        res.setdefault("tokens", {})[name] = int(att.sum())
        maxn = max(a.ns)
        probes = [ProbeBranch(11, labels, trunk.hidden).to(dev).eval() for _ in range(maxn)]
        blocks = {(k, d): [BlockBranch(k, labels, trunk.hidden, d, trunk).to(dev).eval() for _ in range(maxn)]
                  for k, d in ((11, 1), (11, 2), (16, 2), (20, 2))}
        mixed = [BlockBranch(k, labels, trunk.hidden, 2, trunk).to(dev).eval() for k in (8, 11, 14, 16, 18) * 4]

        def separate(n):
            def f():
                for _ in range(n):
                    taps, _ = trunk.taps(ids, att, [L])
                    taps[L].mean(1)
            return f

        def branched(brs, n):
            def f():
                taps, ctx = trunk.taps(ids, att, {br.split for br in brs[:n]})
                for br in brs[:n]:
                    br.probs(taps[br.split], ctx)
            return f

        rows = {}
        for n in a.ns:
            with autocast(dev):
                row = {"separate_models": timeit(separate(n), dev, a.reps),
                       "probe@11": timeit(branched(probes, n), dev, a.reps)}
                for (k, d), brs in blocks.items():
                    row[f"blocks@{k}+{d}"] = timeit(branched(brs, n), dev, a.reps)
                row["mixed@8-18+2"] = timeit(branched(mixed, n), dev, a.reps)
                for k, d in ((11, 2), (16, 2)):
                    if n >= 2:
                        fb = FusedBlocks(blocks[(k, d)][:n])

                        def fused(fb=fb, k=k):
                            taps, ctx = trunk.taps(ids, att, [k])
                            fb.probs(taps[k], ctx)
                        row[f"fused blocks@{k}+{d}"] = timeit(fused, dev, a.reps)
            if laya_agent is not None:
                qs = dict(list(laya_qs.items())[:n])
                row["laya (question-in-input)"] = timeit(lambda: laya_agent.predict(text, qs), dev, a.reps)
            rows[n] = {k: round(v, 2) for k, v in row.items()}
            print(name, f"N={n:2d}", json.dumps(rows[n]), flush=True)
        res["per_message_ms"][name] = rows

    # task switching: branch file vs a full fine-tuned copy of the base
    with tempfile.TemporaryDirectory() as d:
        fp = trunk.fingerprint()
        for kind, br in (("probe", ProbeBranch(11, labels, trunk.hidden)),
                         ("blocks+1", BlockBranch(11, labels, trunk.hidden, 1, trunk)),
                         ("blocks+2", BlockBranch(11, labels, trunk.hidden, 2, trunk))):
            p = os.path.join(d, kind)
            br.save(p, fp, trunk.base)
            res["bytes"][kind] = os.path.getsize(os.path.join(p, "branch.safetensors"))
            ts = []
            for _ in range(5):
                t = time.perf_counter()
                Branch.load(p, trunk)
                sync(dev)
                ts.append((time.perf_counter() - t) * 1000)
            res["switch_ms"][kind] = round(statistics.median(ts), 2)
        from transformers import AutoModel
        full = os.path.join(d, "full")
        AutoModel.from_pretrained(trunk.base).save_pretrained(full)
        res["bytes"]["full_copy"] = sum(os.path.getsize(os.path.join(full, f)) for f in os.listdir(full))
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            AutoModel.from_pretrained(full).to(dev)
            sync(dev)
            ts.append((time.perf_counter() - t) * 1000)
        res["switch_ms"]["full_copy"] = round(statistics.median(ts), 2)
    print("switch ms", res["switch_ms"], "bytes", res["bytes"], flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
