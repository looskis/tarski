"""Can a slot embedding steer a read? Optimise one query on one item toward its expected label.

Also prints the embedding scale (per-element RMS) that the query trainer's learning rate is relative to.
This is a sanity check of the mechanism, not an experiment: fitting one item is trivially possible if
the gradient is right, and meaningless otherwise.

  .venv/bin/python dlm/steer_smoke.py --item hard-opus-a-long_policy-01 --steps 30
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm.reads import Reader  # noqa: E402
from dlm.smoke import load_public  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", default="hard-opus-a-long_policy-01")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr-frac", type=float, default=0.05, help="Adam step as a fraction of the embedding RMS")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task = next(t for t in load_public(root) if t.id == a.item)
    r = Reader()
    mx = r.mx
    import mlx.optimizers as optim

    state = task.state if isinstance(task.state, str) else json.dumps(task.state, ensure_ascii=False)
    prep = r.prepare(state, {task.id: task.question})
    cache = r.prefill(prep["prompt"])
    positions = [s["pos"] for s in prep["slots"]]
    masks = r.masks(cache, prep["width"])
    base = r.embed(mx.array([r.canvas(prep, None)]))
    labels = [c[0] for c in prep["qs"][0]["choices"]]
    y = labels.index(task.expected)
    label_ids = [prep["slots"][0]["label_ids"]]

    q0 = r.mean_embedding()
    rms = float(mx.sqrt((q0.astype(mx.float32) ** 2).mean()))
    tok_rms = float(mx.sqrt((r.embed(mx.array(label_ids[0])).astype(mx.float32) ** 2).mean()))
    print(f"embedding per-element RMS: vocabulary mean {rms:.4f}, label tokens {tok_rms:.4f}; "
          f"lr = {a.lr_frac} x {tok_rms:.4f} = {a.lr_frac * tok_rms:.5f}")

    def loss_fn(q):
        h = r.with_slots(base, positions, [q])
        logits = r.slot_logits(cache, h, positions, masks)
        lp = r.label_logprobs(logits, label_ids)[0]
        return -lp[y]

    params = {"q": q0.astype(mx.float32)}
    opt = optim.Adam(learning_rate=a.lr_frac * tok_rms)
    vg = mx.value_and_grad(lambda p: loss_fn(p["q"]))
    t0 = time.time()
    for step in range(a.steps + 1):
        loss, grads = vg(params)
        mx.eval(loss, grads)
        if step % 5 == 0 or step == a.steps:
            p = float(mx.exp(-loss))
            drift = float(mx.sqrt(((params["q"] - q0.astype(mx.float32)) ** 2).mean())) / tok_rms
            print(f"step {step:3d}  p({task.expected}) = {p:.3f}  loss {float(loss):.3f}  |q - q0| / tok_rms = {drift:.3f}  "
                  f"({time.time() - t0:.0f}s)")
        if step == a.steps:
            break
        params = opt.apply_gradients(grads, params)
        mx.eval(params)


if __name__ == "__main__":
    main()
