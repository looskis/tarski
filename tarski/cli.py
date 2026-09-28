"""Command line: train branches on your own labelled data, inspect them, run them, serve them.

  tarski train tickets.csv --tasks team urgency      # one branch per label column
  tarski list
  tarski eval tickets.csv                            # accuracy of the trained branches on the file's test rows
  tarski predict "Everything is down before our demo" --tasks team urgency
  tarski serve --port 8080 [--fallback http://127.0.0.1:8081]
  tarski label unlabelled.csv --out labels.jsonl        # a local page to label messages, least-confident first
  tarski info

Everything runs locally. The base model is downloaded from Hugging Face once, then read from its cache.
`python -m tarski ...` is the same program.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_STORE = os.environ.get("TARSKI_STORE", "branches")

# short names for the trunks we have measured; any Hugging Face ModernBERT checkpoint works
BASES = {"modernbert": "answerdotai/ModernBERT-base", "gte": "Alibaba-NLP/gte-modernbert-base"}

# the branch-proxy depth selector needs this many validation rows per task to rank depths reliably
# (docs/research/THESIS.md, section 5.4); with fewer, a fixed mid-depth split does better
AUTO_SPLIT_MIN_VAL = 100
DEFAULT_SPLIT = 14


def resolve_base(name):
    return BASES.get(name, name) if name else None


def _load(a):
    from tarski import data

    if a.data in data.BENCHMARKS:
        return data.load(a.data)
    return data.load_table(a.data, text_col=a.text_col, task_cols=a.tasks or None, max_len=a.max_len)


def cmd_train(a):
    from tarski import train
    from tarski.trunk import DEFAULT_BASE, Trunk

    ds = _load(a)
    print(ds.summary())
    trunk = Trunk(resolve_base(a.base) or DEFAULT_BASE, a.device, max_len=ds.max_len)
    tasks = a.tasks or list(ds.tasks)
    if a.split == "auto":
        splits = choose_splits(trunk, ds, tasks, a)
    else:
        splits = {t: int(a.split) for t in tasks}
    res = {}
    for k in sorted(set(splits.values())):
        group = [t for t in tasks if splits[t] == k]
        res.update(train.fit(trunk, ds, group, kind=a.kind, split=k, depth=a.depth, epochs=a.epochs,
                             lr_layers=a.lr_layers, lr_head=a.lr_head, seed=a.seed, store=a.store,
                             oos=not a.no_oos, oos_quantile=a.oos_quantile))
    print(json.dumps({t: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
                      for t, m in res.items()}, indent=1))
    print(f"saved {len(res)} branch(es) to {os.path.abspath(a.store)}")


def choose_splits(trunk, ds, tasks, a):
    """Per-task split depth from a short 1-layer branch at each candidate depth, when there is enough
    validation data to rank depths; otherwise the fixed default. Linear-probe curves are flat across
    depth on ModernBERT and pick splits that are too shallow, so they are not used."""
    from tarski import autosplit

    n_val = {t: len(ds.for_task("val", t)) for t in tasks}
    few = [t for t in tasks if n_val[t] < AUTO_SPLIT_MIN_VAL]
    if few:
        print(f"  auto split: {len(few)} task(s) have fewer than {AUTO_SPLIT_MIN_VAL} validation rows "
              f"({', '.join(f'{t}={n_val[t]}' for t in few[:5])}{', ...' if len(few) > 5 else ''}); "
              f"using split {DEFAULT_SPLIT} for them")
    splits = {t: DEFAULT_SPLIT for t in few}
    rich = [t for t in tasks if t not in few]
    if rich:
        depth = a.depth if a.kind == "blocks" else 0
        cands = [d for d in range(2, trunk.n_layers, 2) if d + max(depth, 1) <= trunk.n_layers]
        curves = autosplit.branch_curves(trunk, ds, rich, depths=cands, seed=a.seed)
        for t, c in curves.items():
            splits[t] = autosplit.choose_split_branch(c, a.tol)
            print(f"  {t}: proxy val acc by depth " + " ".join(f"{d}:{v['acc']:.3f}" for d, v in c.items())
                  + f" -> split {splits[t]}")
    return splits


def cmd_list(a):
    from tarski.engine import scan_store

    metas = scan_store(a.store)
    if not metas:
        print(f"no branches in {os.path.abspath(a.store)}")
        return
    print(f"{'task':40s} {'kind':7s} {'split':>5s} {'depth':>5s} {'labels':>6s} {'MB':>6s} {'test acc':>8s} {'oos':>4s}")
    for name, m in metas.items():
        acc = m.get("metrics", {}).get("acc")
        print(f"{name:40s} {m['kind']:7s} {m['split']:5d} {m.get('depth', 0):5d} {len(m['labels']):6d} "
              f"{m['bytes'] / 1e6:6.1f} {acc if acc is None else f'{acc:.4f}':>8} {'yes' if m.get('oos') else 'no':>4s}")


def _engine(a, **kw):
    from tarski.engine import Engine

    eng = Engine(a.store, resolve_base(a.base), a.device, **kw)
    if not eng.tasks():
        sys.exit(f"no trained branches in {os.path.abspath(a.store)}; run `tarski train` first")
    return eng


def cmd_predict(a):
    from tarski.engine import confidence

    eng = _engine(a)
    tasks = a.tasks or list(eng.tasks())
    texts = a.text if a.text else [line.rstrip("\n") for line in sys.stdin if line.strip()]
    r = eng.decide(texts, tasks)
    for text, probs, oos in zip(texts, r["results"], r["oos"]):
        row = {}
        for t, p in probs.items():
            labels = eng.labels(t)
            row[t] = {"label": labels[int(p.argmax())], "confidence": round(confidence(p), 3)}
            if t in oos:
                row[t]["out_of_scope"] = oos[t]["flag"]
        print(json.dumps({"text": text[:80], "decisions": row}))
    print(json.dumps({"timing_ms": {k: (round(v, 2) if isinstance(v, float) else
                                        {kk: round(vv, 2) for kk, vv in v.items()})
                                    for k, v in r["timing_ms"].items()}}), file=sys.stderr)


def cmd_eval(a):
    """Score the trained branches on a labelled file: its test rows by default, or all rows."""
    import numpy as np

    from tarski.train import evaluate

    ds = _load(a)
    eng = _engine(a)
    known = eng.tasks()
    tasks = [t for t in (a.tasks or list(ds.tasks)) if t in known]
    skipped = [t for t in (a.tasks or list(ds.tasks)) if t not in known]
    if skipped:
        print(f"no trained branch for {skipped}; trained: {sorted(known)}", file=sys.stderr)
    if not tasks:
        sys.exit("nothing to evaluate")
    rows = ds.train + ds.val + ds.test if a.all_rows else ds.split(a.split_name)
    out = {}
    for task in tasks:
        labels = eng.labels(task)
        ex = [e for e in rows if task in e.y and ds.tasks[task].labels[e.y[task]] in labels]
        dropped = sum(1 for e in rows if task in e.y) - len(ex)
        if not ex:
            print(f"[{task}] no labelled rows in the {a.split_name} split", file=sys.stderr)
            continue
        y = np.array([labels.index(ds.tasks[task].labels[e.y[task]]) for e in ex])
        probs, flags = [], []
        for s in range(0, len(ex), a.batch):
            r = eng.decide([e.text for e in ex[s:s + a.batch]], [task])
            probs.extend(p[task] for p in r["results"])
            flags.extend(o[task]["flag"] for o in r["oos"] if task in o)
        m = evaluate(np.stack(probs), y)
        if flags:
            m["oos_flag_rate"] = float(np.mean(flags))
        if dropped:
            m["rows_with_unknown_label"] = dropped
        out[task] = m
        print(f"[{task}] n={m['n']} acc {m['acc']:.4f} macro-F1 {m['macro_f1']:.4f} NLL {m['nll']:.3f} "
              f"ECE {m['ece']:.3f}" + (f" oos-flagged {m['oos_flag_rate']:.3f}" if flags else "")
              + (f" (skipped {dropped} rows whose label the branch does not know)" if dropped else ""))
    if a.json:
        print(json.dumps(out, indent=1))


def cmd_info(a):
    import torch

    from tarski.engine import scan_store
    from tarski.trunk import DEFAULT_BASE, pick_device

    dev = pick_device(a.device)
    print(f"device: {dev} (cuda {torch.cuda.is_available()}, mps {torch.backends.mps.is_available()})")
    print(f"torch {torch.__version__}")
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub = os.path.join(cache, "hub") if not cache.endswith("hub") else cache
    for short, name in BASES.items():
        present = os.path.isdir(os.path.join(hub, "models--" + name.replace("/", "--")))
        print(f"base {short:10s} {name:35s} {'cached' if present else 'not downloaded'}"
              + ("  (default)" if name == DEFAULT_BASE else ""))
    metas = scan_store(a.store)
    print(f"store: {os.path.abspath(a.store)} ({len(metas)} branch(es))")
    for name, m in metas.items():
        print(f"  {name}: {m['kind']}@{m['split']}" + (f"+{m['depth']}" if m.get("depth") else "")
              + f", {len(m['labels'])} labels, {m['bytes'] / 1e6:.1f} MB, base {m['base']}")


def cmd_label(a):
    import uvicorn

    from tarski.label import build_queue, create_label_app

    q = build_queue(a.data, a.out, a.tasks, a.store, resolve_base(a.base), a.device, a.text_col, not a.no_suggest)
    s = q.stats()
    print(f"{s['remaining']} message(s) to label" + (f", {s['already_labelled']} already in {a.out}" if s["already_labelled"] else "")
          + (f"; suggestions from branches {s['suggested_tasks']}" if s["suggested_tasks"] else "; no trained branches, no suggestions")
          + f"\nopen http://{a.host}:{a.port}  (labels append to {os.path.abspath(a.out)}; Ctrl-C when done, "
          f"then `tarski train {a.out}`)", file=sys.stderr)
    uvicorn.run(create_label_app(q), host=a.host, port=a.port, log_level="warning")


def cmd_serve(a):
    import uvicorn

    from tarski.server import create_app

    eng = _engine(a, max_loaded=a.max_loaded)
    if a.preload:
        for t in eng.tasks():
            eng.branch(t)
    app = create_app(eng, a.fallback, a.fallback_model, os.environ.get("TARSKI_FALLBACK_KEY"),
                     a.escalate_below, os.environ.get("TARSKI_API_KEY"), escalate_oos=a.escalate_oos)
    print(f"tarski serving {len(eng.tasks())} task(s) on http://{a.host}:{a.port} (base {eng.trunk.base}, "
          f"device {eng.trunk.device})", file=sys.stderr)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def _data_args(p):
    p.add_argument("data", help="CSV or JSONL with a text column and one column per decision, or a benchmark name")
    p.add_argument("--text-col", default="text")
    p.add_argument("--tasks", nargs="*", help="label columns to use (default: every column but text/split)")
    p.add_argument("--max-len", type=int, default=256, help="tokens per message; longer messages are truncated")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tarski", description="Local decision models: small task branches on a frozen encoder.")
    ap.add_argument("--store", default=DEFAULT_STORE, help="branch directory (default: ./branches or $TARSKI_STORE)")
    ap.add_argument("--base", default=None,
                    help="trunk: 'modernbert' (default), 'gte' (Alibaba-NLP/gte-modernbert-base), or any HF ModernBERT id; "
                         "an existing store's base is used when not given")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train one branch per task from a labelled CSV/JSONL (or a benchmark name)")
    _data_args(t)
    t.add_argument("--kind", choices=["blocks", "probe"], default="blocks",
                   help="blocks: 1-2 transformer layers per task (default); probe: a linear head, KB-sized")
    t.add_argument("--split", default=str(DEFAULT_SPLIT),
                   help=f"trunk depth the branch reads from (default {DEFAULT_SPLIT}), or 'auto' to pick per task")
    t.add_argument("--tol", type=float, default=0.01, help="auto split: shallowest depth within this of the best")
    t.add_argument("--depth", type=int, default=2, help="layers in a blocks branch")
    t.add_argument("--epochs", type=int, default=8, help="minimum; small files run more so that every branch gets 300 steps")
    t.add_argument("--lr-layers", type=float, default=1e-4)
    t.add_argument("--lr-head", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--no-oos", action="store_true", help="skip the out-of-scope detector")
    t.add_argument("--oos-quantile", type=float, default=0.95,
                   help="out-of-scope threshold: share of in-scope validation messages kept unflagged")
    t.set_defaults(fn=cmd_train)

    sub.add_parser("list", help="list trained branches").set_defaults(fn=cmd_list)

    e = sub.add_parser("eval", help="score trained branches on a labelled file")
    _data_args(e)
    e.add_argument("--split-name", choices=["train", "val", "test"], default="test",
                   help="which rows to score (default: the file's test split, or the seeded 10%% carve-out)")
    e.add_argument("--all-rows", action="store_true", help="score every row (use for a file the branches never saw)")
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=cmd_eval)

    p = sub.add_parser("predict", help="run branches on text arguments or stdin lines")
    p.add_argument("text", nargs="*")
    p.add_argument("--tasks", nargs="*")
    p.set_defaults(fn=cmd_predict)

    sub.add_parser("info", help="device, cached base models and the branch store").set_defaults(fn=cmd_info)

    l = sub.add_parser("label", help="a local page to label your own messages, least-confident first")
    l.add_argument("data", help="messages to label: CSV/JSONL with a text column, or a text file with one message per line")
    l.add_argument("--out", default="labels.jsonl", help="where labels go, in the shape `tarski train` reads (appended, resumable)")
    l.add_argument("--tasks", nargs="*", default=[],
                   help="decisions and their labels, e.g. team=billing,outage urgent=yes,no (trained branches add theirs)")
    l.add_argument("--text-col", default="text")
    l.add_argument("--no-suggest", action="store_true", help="do not score messages with the trained branches")
    l.add_argument("--host", default="127.0.0.1")
    l.add_argument("--port", type=int, default=8090)
    l.set_defaults(fn=cmd_label)

    s = sub.add_parser("serve", help="serve /v1/decide and Jev-compatible /v1/systemone")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--max-loaded", type=int, default=64, help="branches kept resident (LRU)")
    s.add_argument("--preload", action="store_true")
    s.add_argument("--fallback", default=None, help="Jev-compatible server for untrained questions")
    s.add_argument("--fallback-model", default="jev-latest")
    s.add_argument("--escalate-below", type=float, default=None,
                   help="send local answers below this confidence to the fallback")
    s.add_argument("--escalate-oos", action="store_true",
                   help="send questions whose message the branch flags as out of scope to the fallback")
    s.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
