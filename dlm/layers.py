"""Where in the network the slot read is assembled (torch backend).

(a) Visibility sweep: the "invisible" mask (no canvas row sees any slot but itself) applied only in
    layers below a cut, or only from the cut upward, with the stock mask elsewhere; accuracy per cut
    locates the layers where the neighbour loop (template rows attending to the slot, the slot reading
    them back) matters.
(b) Logit lens: at every layer's output, final norm + lm_head at the slot rows, restricted to the
    label tokens; per slot, the label distribution per layer, so the report can show the layer at
    which the answer forms and whether it differs by question order (rot0 vs the reversed order).

  .venv/bin/python dlm/layers.py --limit 200 --out results/dlm/layers_typed_test.jsonl
  .venv/bin/python dlm/layers.py --report results/dlm/layers_typed_test.jsonl
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
from dlm.metrics import brier, ece  # noqa: E402

CUTS = (3, 6, 10, 15, 20, 25)


class PerLayerMasks:
    """Give each decoder layer its own attention mask: pre-hooks replace the `attention_mask` kwarg."""

    def __init__(self, reader, masks_by_layer):
        self.r, self.masks, self._h = reader, masks_by_layer, []

    def __enter__(self):
        for i, layer in enumerate(self.r.dec.layers):
            def pre(module, args, kwargs, i=i):
                if i in self.masks:
                    kwargs["attention_mask"] = self.masks[i]
                return args, kwargs
            self._h.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()


class LayerOutputs:
    def __init__(self, reader):
        self.r, self.out, self._h = reader, {}, []

    def __enter__(self):
        self.out = {}
        for i, layer in enumerate(self.r.dec.layers):
            def hook(module, args, output, i=i):
                self.out[i] = output.detach()
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
    n_layers = len(r.dec.layers)
    layer_types = r.text_config.layer_types
    cap = r.model.final_logit_softcapping
    t0 = time.time()
    with open(a.out, "a") as f:
        for n, rec in enumerate(todo, 1):
            qids = list(rec["questions"])
            out = {q: {} for q in qids}
            for oname, order in (("rot0", qids), ("rev", list(reversed(qids)))):
                prep = r.prepare(rec["state"], {q: rec["questions"][q] for q in order})
                cache = r.prefill(prep["prompt"])
                positions = [s["pos"] for s in prep["slots"]]
                label_ids = [s["label_ids"] for s in prep["slots"]]
                keys = [q["key"] for q in prep["qs"]]
                width = prep["width"]
                base = r.masks(cache, width)
                inv = r.slot_masks(cache, width, positions, "invisible")
                ids = torch.tensor([r.canvas(prep, 0)], device=r.device)
                h_rand = r.embed(ids)
                h_mean = r.with_slots(r.embed(torch.tensor([r.canvas(prep, None)], device=r.device)), positions,
                                      [r.mean_embedding()] * len(positions))
                # (a) visibility sweep, random slot
                conds = {"default": {}, "invisible": {i: inv[layer_types[i]] for i in range(n_layers)}}
                for c in CUTS:
                    conds[f"inv_below_{c}"] = {i: inv[layer_types[i]] for i in range(c)}
                    conds[f"inv_from_{c}"] = {i: inv[layer_types[i]] for i in range(c, n_layers)}
                for cname, per_layer in conds.items():
                    with PerLayerMasks(r, per_layer):
                        logits = r.slot_logits(cache, h_rand, positions, base)
                    for k, lp in zip(keys, r.label_logprobs(logits, label_ids)):
                        out[k][f"{oname}:{cname}"] = [float(v) for v in torch.exp(lp).tolist()]
                # (b) logit lens, random and mean slot, stock masks
                for mode, h in (("random", h_rand), ("mean", h_mean)):
                    with LayerOutputs(r) as lo:
                        r.slot_logits(cache, h, positions, base)
                    lens = {k: [] for k in keys}
                    for i in range(n_layers):
                        rows = r.dec.norm(lo.out[i][0, positions])
                        lg = r.model.lm_head(rows).float()
                        lg = torch.tanh(lg / cap) * cap
                        for k, lp in zip(keys, r.label_logprobs(lg, label_ids)):
                            lens[k].append([float(v) for v in torch.exp(lp).tolist()])
                    for k in keys:
                        out[k][f"{oname}:lens_{mode}"] = lens[k]
            base_prep = r.prepare(rec["state"], rec["questions"])
            row = {"id": rec["id"], "family": rec["family"], "questions": []}
            for q in base_prep["qs"]:
                g = rec["gold"].get(q["key"], {})
                row["questions"].append({"qid": q["key"], "type": q["type"], "labels": [c[0] for c in q["choices"]],
                                         "gold": g.get("label"), "reads": out[q["key"]]})
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({(time.time() - t0) / n:.1f} s/state)", flush=True)


def report(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"{len(rows)} states")
    for oname in ("rot0", "rev"):
        print(f"\n== visibility sweep, order {oname}, random slot")
        print(f"{'condition':16s} {'acc':>6s} {'brier':>6s} {'ece':>6s}")
        names = ["default", "invisible"] + [f"inv_below_{c}" for c in CUTS] + [f"inv_from_{c}" for c in CUTS]
        for cname in names:
            items = []
            for row in rows:
                for q in row["questions"]:
                    if q["gold"] not in q["labels"]:
                        continue
                    p = q["reads"].get(f"{oname}:{cname}")
                    if p is None:
                        continue
                    y = q["labels"].index(q["gold"]); top = max(range(len(p)), key=p.__getitem__)
                    items.append((top == y, p[top], brier(p, y)))
            if items:
                n = len(items)
                print(f"{cname:16s} {sum(i[0] for i in items) / n:6.3f} {sum(i[2] for i in items) / n:6.3f} {ece([(i[1], i[0]) for i in items]):6.3f}")
        for mode in ("random", "mean"):
            acc_by_layer = defaultdict(list); settled = []
            for row in rows:
                for q in row["questions"]:
                    if q["gold"] not in q["labels"]:
                        continue
                    lens = q["reads"].get(f"{oname}:lens_{mode}")
                    if not lens:
                        continue
                    y = q["labels"].index(q["gold"])
                    tops = [max(range(len(p)), key=p.__getitem__) for p in lens]
                    for i, t in enumerate(tops):
                        acc_by_layer[i].append(t == y)
                    final = tops[-1]
                    first = next((i for i in range(len(tops)) if all(t == final for t in tops[i:])), len(tops) - 1)
                    settled.append(first)
            if acc_by_layer:
                print(f"  logit lens ({mode} slot), accuracy by layer: " +
                      " ".join(f"{i}:{sum(v) / len(v):.2f}" for i, v in sorted(acc_by_layer.items())))
                print(f"  layer at which the final answer settles: mean {sum(settled) / len(settled):.1f}, "
                      f"median {sorted(settled)[len(settled) // 2]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default="results/dlm/layers_typed_test.jsonl")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        run(a)


if __name__ == "__main__":
    main()
