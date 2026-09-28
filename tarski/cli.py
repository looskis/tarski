"""Command line: train branches on your own labelled data, inspect them, run them, serve them.

  python -m tarski train tickets.csv --tasks team urgency          # one branch per label column
  python -m tarski list
  python -m tarski predict "Everything is down before our demo" --tasks team urgency
  python -m tarski serve --port 8080 [--fallback http://127.0.0.1:8081]

Everything runs locally. The base model is downloaded from Hugging Face once, then read from its cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_STORE = os.environ.get("TARSKI_STORE", "branches")


def cmd_train(a):
    from tarski import data, train
    from tarski.trunk import DEFAULT_BASE, Trunk

    ds = data.load(a.data, text_col=a.text_col, task_cols=a.tasks or None, max_len=a.max_len) \
        if a.data not in data.BENCHMARKS else data.load(a.data)
    print(ds.summary())
    trunk = Trunk(a.base or DEFAULT_BASE, a.device, max_len=ds.max_len)
    tasks = a.tasks or list(ds.tasks)
    if a.split == "auto":
        from tarski import autosplit

        hi = trunk.n_layers - (a.depth if a.kind == "blocks" else 0)
        curves = autosplit.layer_curves(trunk, ds, tasks, depths=range(1, hi + 1))
        splits = {t: autosplit.choose_split(c, a.tol) for t, c in curves.items()}
        for t, c in curves.items():
            print(f"  {t}: probe val acc by depth " + " ".join(f"{d}:{v:.3f}" for d, v in c.items())
                  + f" -> split {splits[t]}")
    else:
        splits = {t: int(a.split) for t in tasks}
    res = {}
    for k in sorted(set(splits.values())):
        group = [t for t in tasks if splits[t] == k]
        res.update(train.fit(trunk, ds, group, kind=a.kind, split=k, depth=a.depth, epochs=a.epochs,
                             lr_layers=a.lr_layers, lr_head=a.lr_head, seed=a.seed, store=a.store))
    print(json.dumps({t: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
                      for t, m in res.items()}, indent=1))
    print(f"saved {len(res)} branch(es) to {os.path.abspath(a.store)}")


def cmd_list(a):
    from tarski.engine import scan_store

    metas = scan_store(a.store)
    if not metas:
        print(f"no branches in {os.path.abspath(a.store)}")
        return
    print(f"{'task':40s} {'kind':7s} {'split':>5s} {'depth':>5s} {'labels':>6s} {'MB':>6s} {'test acc':>8s}")
    for name, m in metas.items():
        acc = m.get("metrics", {}).get("acc")
        print(f"{name:40s} {m['kind']:7s} {m['split']:5d} {m.get('depth', 0):5d} {len(m['labels']):6d} "
              f"{m['bytes'] / 1e6:6.1f} {acc if acc is None else f'{acc:.4f}':>8}")


def cmd_predict(a):
    from tarski.engine import Engine, confidence

    eng = Engine(a.store, a.base, a.device)
    tasks = a.tasks or list(eng.tasks())
    if not tasks:
        sys.exit(f"no trained branches in {os.path.abspath(a.store)}; run `python -m tarski train` first")
    texts = a.text if a.text else [line.rstrip("\n") for line in sys.stdin if line.strip()]
    r = eng.decide(texts, tasks)
    for text, probs in zip(texts, r["results"]):
        row = {}
        for t, p in probs.items():
            labels = eng.labels(t)
            row[t] = {"label": labels[int(p.argmax())], "confidence": round(confidence(p), 3)}
        print(json.dumps({"text": text[:80], "decisions": row}))
    print(json.dumps({"timing_ms": {k: (round(v, 2) if isinstance(v, float) else
                                        {kk: round(vv, 2) for kk, vv in v.items()})
                                    for k, v in r["timing_ms"].items()}}), file=sys.stderr)


def cmd_serve(a):
    import uvicorn

    from tarski.engine import Engine
    from tarski.server import create_app

    eng = Engine(a.store, a.base, a.device, a.max_loaded)
    if a.preload:
        for t in eng.tasks():
            eng.branch(t)
    app = create_app(eng, a.fallback, a.fallback_model, os.environ.get("TARSKI_FALLBACK_KEY"),
                     a.escalate_below, os.environ.get("TARSKI_API_KEY"))
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tarski")
    ap.add_argument("--store", default=DEFAULT_STORE, help="branch directory (default: ./branches or $TARSKI_STORE)")
    ap.add_argument("--base", default=None, help="base model (default: ModernBERT-base, or what the store was trained on)")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train one branch per task from a labelled CSV/JSONL (or a benchmark name)")
    t.add_argument("data")
    t.add_argument("--text-col", default="text")
    t.add_argument("--tasks", nargs="*", help="label columns to train (default: every column but text/split)")
    t.add_argument("--kind", choices=["blocks", "probe"], default="blocks")
    t.add_argument("--split", default="11", help="trunk depth the branch reads from, or 'auto' to pick per task")
    t.add_argument("--tol", type=float, default=0.01, help="auto split: shallowest depth within this of the best")
    t.add_argument("--depth", type=int, default=2, help="layers in a blocks branch")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--lr-layers", type=float, default=1e-4)
    t.add_argument("--lr-head", type=float, default=1e-3)
    t.add_argument("--max-len", type=int, default=256)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)

    sub.add_parser("list", help="list trained branches").set_defaults(fn=cmd_list)

    p = sub.add_parser("predict", help="run branches on text arguments or stdin lines")
    p.add_argument("text", nargs="*")
    p.add_argument("--tasks", nargs="*")
    p.set_defaults(fn=cmd_predict)

    s = sub.add_parser("serve", help="serve /v1/decide and Jev-compatible /v1/systemone")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--max-loaded", type=int, default=64, help="branches kept resident (LRU)")
    s.add_argument("--preload", action="store_true")
    s.add_argument("--fallback", default=None, help="Jev-compatible server for untrained questions")
    s.add_argument("--fallback-model", default="jev-latest")
    s.add_argument("--escalate-below", type=float, default=None,
                   help="send local answers below this confidence to the fallback")
    s.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
