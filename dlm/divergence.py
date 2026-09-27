"""Where do two question orders diverge? Per layer, the cosine similarity between orders of (a) the
encoded state (encoder layer outputs over the last N prompt tokens, which are the same tokens under
every order) and (b) each slot's residual stream in the decoder (matched by question). The seed
comparison (same order, seed 0 vs seed 1) is the noise floor: the encoder is identical there, so any
encoder divergence between orders is the prompt's question list changing how the state is read.

  .venv/bin/python dlm/divergence.py --limit 200 --out results/dlm/divergence_typed_test.jsonl
  .venv/bin/python dlm/divergence.py --report results/dlm/divergence_typed_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm import data  # noqa: E402

PAIRS = (("rot0", "rev"), ("rot0", "rot3"), ("rot0", "rot0_seed1"))


def encoder_layers(r):
    enc = r.enc
    for path in ("layers", "language_model.layers", "text_model.layers", "model.layers"):
        m = enc
        try:
            for part in path.split("."):
                m = getattr(m, part)
            return list(m)
        except AttributeError:
            continue
    raise RuntimeError("encoder layers not found")


class Outputs:
    def __init__(self, layers):
        self.layers, self.out, self._h = layers, {}, []

    def __enter__(self):
        self.out = {}
        for i, layer in enumerate(self.layers):
            def hook(module, args, output, i=i):
                self.out[i] = (output[0] if isinstance(output, (tuple, list)) else output).detach()
            self._h.append(layer.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()


def run(a):
    import torch
    from dlm.reads_torch import Reader
    recs = data.typed_decisions(a.split)[: a.limit]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} states, {len(todo)} to run", flush=True)
    r = Reader(attn_implementation=a.attn)
    enc_layers = encoder_layers(r)
    dec_layers = list(r.dec.layers)
    cos = torch.nn.CosineSimilarity(dim=-1)
    t0 = time.time()
    with open(a.out, "a") as f:
        for n, rec in enumerate(todo, 1):
            qids = list(rec["questions"])
            variants = {"rot0": (qids, 0), "rot3": (qids[3:] + qids[:3], 0), "rev": (list(reversed(qids)), 0),
                        "rot0_seed1": (qids, 1)}
            enc_h, slot_h, probs, plen = {}, {}, {}, {}
            for name, (order, seed) in variants.items():
                prep = r.prepare(rec["state"], {q: rec["questions"][q] for q in order})
                r._prefills.clear()
                with Outputs(enc_layers) as eo:
                    cache = r.prefill(prep["prompt"])
                plen[name] = len(prep["prompt"])
                enc_h[name] = torch.stack([eo.out[i][0, -a.tail:] for i in range(len(enc_layers))])   # [L, N, D]
                positions = [s["pos"] for s in prep["slots"]]
                keys = [q["key"] for q in prep["qs"]]
                ids = torch.tensor([r.canvas(prep, seed)], device=r.device)
                with Outputs(dec_layers) as do:
                    p = r.read(prep, "random", seed=seed)
                probs[name] = dict(zip(keys, p))
                slot_h[name] = {k: torch.stack([do.out[i][0, pos] for i in range(len(dec_layers))])  # [L, D]
                                for k, pos in zip(keys, positions)}
            row = {"id": rec["id"], "family": rec["family"], "prompt_len": plen, "pairs": {}}
            for x, y in PAIRS:
                enc_cos = cos(enc_h[x].float(), enc_h[y].float()).mean(dim=-1).tolist()          # per layer
                per_q = {}
                for k in qids:
                    px, py = probs[x][k], probs[y][k]
                    per_q[k] = {"slot_cos": cos(slot_h[x][k].float(), slot_h[y][k].float()).tolist(),
                                "flip": max(range(len(px)), key=px.__getitem__) != max(range(len(py)), key=py.__getitem__),
                                "tv": 0.5 * sum(abs(u - v) for u, v in zip(px, py))}
                row["pairs"][f"{x}|{y}"] = {"enc_cos": enc_cos, "questions": per_q}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({(time.time() - t0) / n:.1f} s/state)", flush=True)


def report(path):
    import numpy as np
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"{len(rows)} states; prompt lengths equal across orders in "
          f"{sum(len(set(r['prompt_len'].values())) == 1 for r in rows)}/{len(rows)} states")
    for pair in rows[0]["pairs"]:
        enc = np.array([r["pairs"][pair]["enc_cos"] for r in rows])                    # [S, L]
        slot = np.array([q["slot_cos"] for r in rows for q in r["pairs"][pair]["questions"].values()])  # [Q, L]
        flips = np.array([q["flip"] for r in rows for q in r["pairs"][pair]["questions"].values()])
        tvs = np.array([q["tv"] for r in rows for q in r["pairs"][pair]["questions"].values()])
        print(f"\n== {pair}: flips {flips.mean():.2f}, TV {tvs.mean():.3f}")
        print("  encoder, state tokens, cos by layer: " + " ".join(f"{i}:{v:.3f}" for i, v in enumerate(enc.mean(0))))
        print("  decoder, slot rows,   cos by layer: " + " ".join(f"{i}:{v:.3f}" for i, v in enumerate(slot.mean(0))))
        if flips.any() and (~flips).any():
            print("  slot cos at layers 5/15/22/29, flipped vs not: " +
                  "  ".join(f"L{i}: {slot[flips, i].mean():.3f} vs {slot[~flips, i].mean():.3f}" for i in (5, 15, 22, 29)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--tail", type=int, default=64, help="last N prompt tokens compared (the state's end)")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default="results/dlm/divergence_typed_test.jsonl")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        run(a)


if __name__ == "__main__":
    main()
