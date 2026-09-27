"""Multi-step reads (torch backend). OpenJev reads a slot in one denoising pass. Here the pass is
repeated: step t feeds step t-1's logits back as the self-conditioning signal and (optionally) writes
step t-1's argmax token into each slot, as DiffusionGemma's own sampler would, then reads the label
distribution again. Conditions per slot (start x update x step):

  start   random  slot starts as OpenJev's random token (seed 0);  mean  slot starts as the vocabulary mean
  update  token   slots take the previous step's argmax token;      sc    slots keep their start, only the
                  self-conditioning signal is passed
  step    1..K    step 1 is the stock single read

  .venv/bin/python dlm/steps.py --limit 200 --steps 4 --out results/dlm/steps_typed_test.jsonl
  .venv/bin/python dlm/steps.py --report results/dlm/steps_typed_test.jsonl [--by type]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm import data  # noqa: E402
from dlm.metrics import brier, ece, tv  # noqa: E402


def run(a):
    import torch
    from dlm.reads_torch import Reader
    if a.dataset == "jevbench":
        recs = data.jevbench_public(tuple(a.tiers))
    else:
        recs = data.typed_decisions(a.dataset.split("-")[1])
    recs = recs[: a.limit]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} states, {len(todo)} to run, {a.steps} steps", flush=True)
    r = Reader(attn_implementation=a.attn)
    t0 = time.time()
    with open(a.out, "a") as f:
        for n, rec in enumerate(todo, 1):
            prep = r.prepare(rec["state"], rec["questions"])
            cache = r.prefill(prep["prompt"])
            positions = [s["pos"] for s in prep["slots"]]
            label_ids = [s["label_ids"] for s in prep["slots"]]
            masks = r.masks(cache, prep["width"])
            keys = [q["key"] for q in prep["qs"]]
            out = {k: {} for k in keys}
            for start in ("random", "mean"):
                ids0 = torch.tensor([r.canvas(prep, 0 if start == "random" else None)], device=r.device)
                h0 = r.embed(ids0)
                if start == "mean":
                    h0 = r.with_slots(h0, positions, [r.mean_embedding()] * len(positions))
                for update in ("token", "sc"):
                    h, sc = h0, None
                    for t in range(1, a.steps + 1):
                        logits = r.decoder_logits(cache, h, masks, sc)          # [1, L, V]
                        lps = r.label_logprobs(logits[0, positions], label_ids)
                        for k, lp in zip(keys, lps):
                            out[k][f"{start}_{update}_step{t}"] = [float(v) for v in torch.exp(lp).tolist()]
                        sc = logits
                        if update == "token":
                            ids = ids0.clone()
                            ids[0, positions] = logits[0, positions].argmax(dim=-1)
                            h = r.embed(ids)
            row = {"id": rec["id"], "family": rec.get("family"), "source": rec.get("source"), "questions": []}
            for q in prep["qs"]:
                g = rec["gold"].get(q["key"], {})
                row["questions"].append({"qid": q["key"], "type": q["type"], "labels": [c[0] for c in q["choices"]],
                                         "gold": g.get("label"), "reads": out[q["key"]]})
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({(time.time() - t0) / n:.1f} s/state)", flush=True)


def report(path, by=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    acc = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for q in row["questions"]:
            if q["gold"] not in q["labels"]:
                continue
            y = q["labels"].index(q["gold"])
            key = {"type": q["type"], "family": row["family"], "source": row["source"]}.get(by, "all") if by else "all"
            for c, p in q["reads"].items():
                start = c.split("_")[0]
                ref = q["reads"][f"{start}_token_step1"]
                top = max(range(len(p)), key=p.__getitem__)
                acc[key][c].append({"correct": top == y, "conf": p[top], "brier": brier(p, y), "tv": tv(p, ref)})
    for key in sorted(acc):
        d = acc[key]
        print(f"\n== {key} (n={len(next(iter(d.values())))} slots)")
        print(f"{'condition':22s} {'acc':>6s} {'brier':>6s} {'ece':>6s} {'tv->step1':>9s}")
        for c in sorted(d, key=lambda c: (c.split("_")[0], c.split("_")[1], int(c.split("step")[1]))):
            v = d[c]; n = len(v)
            print(f"{c:22s} {sum(x['correct'] for x in v) / n:6.3f} {sum(x['brier'] for x in v) / n:6.3f} "
                  f"{ece([(x['conf'], x['correct']) for x in v]):6.3f} {sum(x['tv'] for x in v) / n:9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["jevbench", "typed-test", "typed-train"], default="typed-test")
    ap.add_argument("--tiers", nargs="+", default=["hard"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default="results/dlm/steps_typed_test.jsonl")
    ap.add_argument("--report", default=None)
    ap.add_argument("--by", choices=["type", "family", "source"], default=None)
    a = ap.parse_args()
    if a.report:
        report(a.report, a.by)
    else:
        run(a)


if __name__ == "__main__":
    main()
