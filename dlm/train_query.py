"""Learned denoising queries: train the slot embeddings that replace OpenJev's random slot tokens.

A query is one continuous input embedding per question type (noul / choice / score), or one shared
vector. It is placed in the canvas where OpenJev puts a random token, and trained so that a single
read reproduces a target distribution per slot:

  --target meanK   the model's own noise-averaged read (mean over the seeds in the anatomy file):
                   label-free self-distillation, the main setting
  --target gold    the dataset's gold distribution (typed-decisions carries soft teacher labels),
                   the supervised control

Nothing else moves: weights, prefill cache and canvas are untouched. Training runs on the laptop at
about 0.3 s per state (five slots).

  .venv/bin/python dlm/train_query.py --train results/dlm/anatomy_typed_train.jsonl \
      --test results/dlm/anatomy_typed_test.jsonl --target meanK --per type --epochs 3 --name meanK_type

Writes results/dlm/queries_<name>.safetensors (+ .json with the run's settings and per-epoch test
metrics) and results/dlm/anatomy_typed_test.<name>.jsonl, the test anatomy file with the query read
added as reads["query:<name>"], for dlm/metrics.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlm import data  # noqa: E402
from dlm.metrics import brier, ece, mean_of, tv  # noqa: E402
from dlm.reads import Reader  # noqa: E402

TYPES = ("noul", "choice", "score")


def load_anatomy(path):
    with open(path) as f:
        return {row["id"]: row for row in (json.loads(l) for l in f if l.strip())}


def targets_for(row, target):
    """Per question: the target distribution in label order, or None."""
    out = []
    for q in row["questions"]:
        if target == "meanK":
            seeds = [v for k, v in q["reads"].items() if k.startswith("seed:")]
            out.append(mean_of(seeds))
        elif target == "gold":
            gp = q.get("gold_probs")
            out.append([float(gp.get(l, 0.0)) for l in q["labels"]] if gp else None)
        else:
            raise ValueError(target)
    return out


class Queries:
    def __init__(self, r: Reader, per: str, init):
        self.mx, self.per = r.mx, per
        keys = list(TYPES) if per == "type" else ["shared"]
        self.params = {k: init.astype(r.mx.float32) for k in keys}

    def key(self, qtype):
        return qtype if self.per == "type" else "shared"

    def rows(self, params, qtypes):
        return [params[self.key(t)] for t in qtypes]

    def save(self, path, meta):
        self.mx.save_safetensors(path, self.params)
        with open(path.replace(".safetensors", ".json"), "w") as f:
            json.dump(meta, f, indent=1)

    @staticmethod
    def load(path, r: Reader):
        with open(path.replace(".safetensors", ".json")) as f:
            meta = json.load(f)
        q = Queries(r, meta["per"], r.mean_embedding())
        q.params = dict(r.mx.load(path))
        return q, meta


def evaluate(r: Reader, records, anatomy, queries: Queries, name, out_path=None):
    """Query reads on every record; JevBench-style metrics against gold, and TV / flips against meanK."""
    mx = r.mx
    stats = {"n": 0, "correct": 0, "brier": 0.0, "pairs": [], "tv": 0.0, "flip": 0, "per_type": {}}
    rows_out = []
    for rec in records:
        row = anatomy.get(rec["id"])
        if row is None:
            continue
        prep = r.prepare(rec["state"], rec["questions"])
        probs = r.read(prep, "query", queries=queries.rows(queries.params, [q["type"] for q in prep["qs"]]))
        for q, p in zip(row["questions"], probs):
            q["reads"][f"query:{name}"] = p
            ref = mean_of([v for k, v in q["reads"].items() if k.startswith("seed:")])
            top = max(range(len(p)), key=p.__getitem__)
            d = stats["per_type"].setdefault(q["type"], {"n": 0, "correct": 0, "brier": 0.0, "pairs": [], "tv": 0.0})
            for s in (stats, d):
                s["n"] += 1
                s["tv"] += tv(p, ref)
            stats["flip"] += top != max(range(len(ref)), key=ref.__getitem__)
            if q["gold"] in q["labels"]:
                y = q["labels"].index(q["gold"])
                for s in (stats, d):
                    s["correct"] += top == y
                    s["brier"] += brier(p, y)
                    s["pairs"].append((p[top], top == y))
        rows_out.append(row)
    if out_path:
        with open(out_path, "w") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def summarise(s):
        n_g = len(s["pairs"])
        return {"n": s["n"], "acc": s["correct"] / max(n_g, 1), "brier": s["brier"] / max(n_g, 1),
                "ece": ece(s["pairs"]) if n_g else None, "tv_to_meanK": s["tv"] / max(s["n"], 1)}

    out = summarise(stats)
    out["flip_rate"] = stats["flip"] / max(stats["n"], 1)
    out["per_type"] = {t: summarise(d) for t, d in stats["per_type"].items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="anatomy JSONL of the training states")
    ap.add_argument("--test", required=True, help="anatomy JSONL of the test states")
    ap.add_argument("--train-dataset", default="typed-train", choices=["typed-train", "typed-test", "jevbench"])
    ap.add_argument("--test-dataset", default="typed-test", choices=["typed-train", "typed-test", "jevbench"])
    ap.add_argument("--target", choices=["meanK", "gold"], default="meanK")
    ap.add_argument("--per", choices=["type", "shared"], default="type")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-frac", type=float, default=0.01, help="Adam step as a fraction of the label-token embedding RMS")
    ap.add_argument("--anchor", type=float, default=0.0, help="L2 pull toward the mean embedding, per element")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="results/dlm")
    a = ap.parse_args()

    loaders = {"typed-train": lambda: data.typed_decisions("train"), "typed-test": lambda: data.typed_decisions("test"),
               "jevbench": lambda: data.jevbench_public()}
    train_recs, test_recs = loaders[a.train_dataset](), loaders[a.test_dataset]()
    train_an, test_an = load_anatomy(a.train), load_anatomy(a.test)
    train_recs = [x for x in train_recs if x["id"] in train_an][: a.limit]
    print(f"train {len(train_recs)} states, test {len(test_recs)} states, target {a.target}, per {a.per}", flush=True)

    r = Reader()
    mx = r.mx
    import mlx.optimizers as optim

    q0 = r.mean_embedding()
    tok_rms = float(mx.sqrt((r.embed(mx.array([562, 603, 565, 622])).astype(mx.float32) ** 2).mean()))
    lr = a.lr_frac * tok_rms
    queries = Queries(r, a.per, q0)
    opt = optim.Adam(learning_rate=lr)
    meta = {"name": a.name, "target": a.target, "per": a.per, "lr": lr, "lr_frac": a.lr_frac, "anchor": a.anchor,
            "epochs": a.epochs, "train": a.train, "n_train": len(train_recs), "history": []}
    os.makedirs(a.out_dir, exist_ok=True)
    qpath = os.path.join(a.out_dir, f"queries_{a.name}.safetensors")

    # epoch 0: the untrained query (= the mean-embedding read) as the reference point
    m = evaluate(r, test_recs, test_an, queries, a.name)
    print(f"epoch 0  test acc {m['acc']:.3f} brier {m['brier']:.3f} ece {m['ece']:.3f} tv->meanK {m['tv_to_meanK']:.3f}", flush=True)
    meta["history"].append({"epoch": 0, **m})

    def loss_fn(params, prep, tgt_rows, qtypes, base, cache, positions, masks, label_ids):
        h = r.with_slots(base, positions, queries.rows(params, qtypes))
        logits = r.slot_logits(cache, h, positions, masks)
        lps = r.label_logprobs(logits, label_ids)
        loss = mx.zeros(())
        n = 0
        for lp, t in zip(lps, tgt_rows):
            if t is None:
                continue
            loss = loss - (mx.array(t) * lp).sum()
            n += 1
        loss = loss / max(n, 1)
        if a.anchor:
            for k, v in params.items():
                loss = loss + a.anchor * ((v - q0) ** 2).mean()
        return loss

    vg = mx.value_and_grad(loss_fn)
    rng = random.Random(a.seed)
    preps = {}
    for epoch in range(1, a.epochs + 1):
        order = list(train_recs)
        rng.shuffle(order)
        t0, tot, n_steps = time.time(), 0.0, 0
        for i, rec in enumerate(order, 1):
            row = train_an[rec["id"]]
            tgts = targets_for(row, a.target)
            if all(t is None for t in tgts):
                continue
            if rec["id"] not in preps:
                preps[rec["id"]] = r.prepare(rec["state"], rec["questions"])
            prep = preps[rec["id"]]
            cache = r.prefill(prep["prompt"])
            positions = [s["pos"] for s in prep["slots"]]
            masks = r.masks(cache, prep["width"])
            base = r.embed(mx.array([r.canvas(prep, None)]))
            qtypes = [q["type"] for q in prep["qs"]]
            label_ids = [s["label_ids"] for s in prep["slots"]]
            loss, grads = vg(queries.params, prep, tgts, qtypes, base, cache, positions, masks, label_ids)
            queries.params = opt.apply_gradients(grads, queries.params)
            mx.eval(queries.params, loss)
            tot += float(loss)
            n_steps += 1
            if i % 100 == 0:
                print(f"  epoch {epoch} {i}/{len(order)} loss {tot / n_steps:.4f} ({(time.time() - t0) / i:.2f} s/state)", flush=True)
        m = evaluate(r, test_recs, test_an, queries, a.name,
                     out_path=os.path.join(a.out_dir, os.path.basename(a.test).replace(".jsonl", f".{a.name}.jsonl")))
        print(f"epoch {epoch}  train loss {tot / max(n_steps, 1):.4f}  test acc {m['acc']:.3f} brier {m['brier']:.3f} "
              f"ece {m['ece']:.3f} tv->meanK {m['tv_to_meanK']:.3f} flips {m['flip_rate']:.3f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        meta["history"].append({"epoch": epoch, "train_loss": tot / max(n_steps, 1), **m})
        queries.save(qpath, meta)
    print(f"saved {qpath}")


if __name__ == "__main__":
    main()
