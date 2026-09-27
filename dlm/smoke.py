"""Feasibility gate for learned denoising queries, on one JevBench item.

1. Parity: the embeddings path reproduces the stock `diffusion_decoder_logits` slot rows exactly.
2. Seed variance: a few random-token reads of the same canvas.
3. The mean-embedding read.
4. A gradient of a KL loss with respect to a slot embedding flows through the quantised MoE decoder.

  .venv/bin/python dlm/smoke.py [--item hard-opus-a-long_policy-01]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm.reads import Reader  # noqa: E402


def load_public(root):
    from jevbench.tasks import load_jsonl

    tasks = []
    for name in ("original", "easy", "hard"):
        tasks += load_jsonl(os.path.join(root, "third_party", "jevbench", "datasets", "public", f"{name}.jsonl"))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", default=None)
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tasks = load_public(root)
    task = next(t for t in tasks if t.id == a.item) if a.item else min(tasks, key=lambda t: len(json.dumps(t.state)))
    print(f"item {task.id} ({task.family}, {task.question['type']}, expected {task.expected!r})")

    t0 = time.time()
    r = Reader()
    print(f"model loaded in {time.time() - t0:.0f}s")
    mx = r.mx

    state = task.state if isinstance(task.state, str) else json.dumps(task.state, ensure_ascii=False)
    prep = r.prepare(state, {task.id: task.question})
    print(f"prompt {len(prep['prompt'])} tokens, canvas width {prep['width']}, slots {prep['slots']}")

    # 1. parity
    t0 = time.time()
    ref = r.reference_slot_logits(prep, seed=0)
    mx.eval(ref)
    t_ref = time.time() - t0
    cache = r.prefill(prep["prompt"])
    ids = mx.array([r.canvas(prep, 0)])
    mine = r.slot_logits(cache, r.embed(ids), [s["pos"] for s in prep["slots"]], r.masks(cache, prep["width"]))
    mx.eval(mine)
    diff = float(mx.abs(ref - mine).max())
    lab = [s["label_ids"] for s in prep["slots"]]
    p_ref = mx.exp(r.label_logprobs(ref, lab)[0])
    p_mine = mx.exp(r.label_logprobs(mine, lab)[0])
    tv = 0.5 * float(mx.abs(p_ref - p_mine).sum())
    print(f"parity: max |stock - embeds path| = {diff:.3e} over {ref.shape} bf16 vocab logits; "
          f"label-distribution TV = {tv:.2e} (stock read {t_ref:.2f}s)")

    # 2. seed variance
    labels = [c[0] for c in prep["qs"][0]["choices"]]
    for seed in range(a.seeds):
        t0 = time.time()
        p = r.read(prep, "random", seed=seed)[0]
        print(f"seed {seed}: " + " ".join(f"{l}={v:.3f}" for l, v in zip(labels, p)) + f"  ({time.time() - t0:.2f}s)")

    # 3. mean embedding
    t0 = time.time()
    p = r.read(prep, "mean")[0]
    print("mean-embedding: " + " ".join(f"{l}={v:.3f}" for l, v in zip(labels, p)) + f"  ({time.time() - t0:.2f}s)")

    # 4. gradient through the decoder w.r.t. a slot embedding
    positions = [s["pos"] for s in prep["slots"]]
    masks = r.masks(cache, prep["width"])
    base = r.embed(mx.array([r.canvas(prep, None)]))
    target = mx.array(p)                                                 # pretend target: the mean read

    def loss_fn(q):
        h = r.with_slots(base, positions, [q])
        logits = r.slot_logits(cache, h, positions, masks)
        lp = r.label_logprobs(logits, [prep["slots"][0]["label_ids"]])[0]
        return -(target * lp).sum()

    q0 = r.mean_embedding().astype(base.dtype)
    t0 = time.time()
    loss, grad = mx.value_and_grad(loss_fn)(q0)
    mx.eval(loss, grad)
    gn = float(mx.sqrt((grad.astype(mx.float32) ** 2).sum()))
    print(f"grad: loss {float(loss):.4f}, |grad| {gn:.4e}, finite {bool(mx.all(mx.isfinite(grad)))}, "
          f"{time.time() - t0:.2f}s, peak memory {mx.get_peak_memory() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
