"""A local labelling page: label your own messages, with the current branches' suggestions first.

  tarski label messages.csv --out labels.jsonl
  tarski label messages.csv --out labels.jsonl --tasks team=billing,outage,feature urgent=yes,no

Messages come from a CSV or JSONL file (a text column) or a plain text file (one message per line).
Labels are appended to a JSONL file in the shape `tarski train` reads: one object per message with a
`text` key and one key per decision. Messages already in the output file are skipped, so a session can
be stopped and resumed. When a branch store exists, every unlabelled message is scored once and the
queue is ordered so that messages a branch flags as out of scope come first, then the least confident:
labelling effort goes where the model is weakest, and confirming a correct suggestion is one click.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel

from tarski.engine import confidence

STATIC = os.path.join(os.path.dirname(__file__), "static", "label.html")


class SaveRequest(BaseModel):
    id: str
    decisions: Dict[str, str]


class SkipRequest(BaseModel):
    id: str


def read_messages(path: str, text_col: str = "text") -> List[str]:
    if path.endswith((".jsonl", ".json")):
        with open(path) as f:
            return [str(json.loads(line)[text_col]) for line in f if line.strip()]
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and text_col not in rows[0]:
            raise ValueError(f"{path} has no {text_col!r} column; columns are {list(rows[0])}")
        return [str(r[text_col]) for r in rows]
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


def parse_task_specs(specs: Sequence[str]) -> Dict[str, List[str]]:
    """'team=billing,outage,feature' -> {'team': [...]}; a bare 'team' means labels come from the store
    or from what the labeller types."""
    out = {}
    for s in specs:
        name, _, labels = s.partition("=")
        out[name.strip()] = [l.strip() for l in labels.split(",") if l.strip()] if labels else []
    return out


def _key(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()[:16]


class LabelQueue:
    def __init__(self, messages: Sequence[str], out_path: str, tasks: Dict[str, List[str]], engine=None,
                 batch: int = 64):
        self.out_path = out_path
        self.tasks: Dict[str, List[str]] = {t: list(ls) for t, ls in tasks.items()}
        self.done = self._read_done()
        self.skipped: set = set()
        seen, self.items = set(), []
        for text in messages:
            text = text.strip()
            k = _key(text)
            if not text or k in seen or k in self.done:
                continue
            seen.add(k)
            self.items.append({"id": k, "text": text, "suggest": {}, "priority": [1, 1.0]})
        self.by_id = {it["id"]: it for it in self.items}
        self.suggested_tasks: List[str] = []
        if engine is not None:
            self._suggest(engine, batch)

    # -- setup ----------------------------------------------------------------------------------------

    def _read_done(self) -> set:
        done = set()
        if os.path.exists(self.out_path):
            with open(self.out_path) as f:
                for line in f:
                    if line.strip():
                        done.add(_key(str(json.loads(line).get("text", ""))))
        return done

    def _suggest(self, engine, batch: int) -> None:
        known = engine.tasks()
        for t, m in known.items():
            self.tasks.setdefault(t, [])
            for l in m["labels"]:
                if l not in self.tasks[t]:
                    self.tasks[t].append(l)
        tasks = [t for t in self.tasks if t in known]
        self.suggested_tasks = tasks
        if not tasks or not self.items:
            return
        for s in range(0, len(self.items), batch):
            chunk = self.items[s:s + batch]
            r = engine.decide([it["text"] for it in chunk], tasks)
            for it, probs, oos in zip(chunk, r["results"], r["oos"]):
                for t in tasks:
                    p = probs[t]
                    labels = engine.labels(t)
                    it["suggest"][t] = {"label": labels[int(p.argmax())], "confidence": round(confidence(p), 3),
                                        "out_of_scope": bool(oos.get(t, {}).get("flag", False))}
                flagged = any(v["out_of_scope"] for v in it["suggest"].values())
                it["priority"] = [0 if flagged else 1, min(v["confidence"] for v in it["suggest"].values())]
        self.items.sort(key=lambda it: it["priority"])

    # -- queue ----------------------------------------------------------------------------------------

    def next(self, n: int = 1) -> List[Dict]:
        out = []
        for it in self.items:
            if it["id"] not in self.done and it["id"] not in self.skipped:
                out.append({k: it[k] for k in ("id", "text", "suggest")})
                if len(out) >= n:
                    break
        return out

    def save(self, item_id: str, decisions: Dict[str, str]) -> Dict:
        it = self.by_id.get(item_id)
        if it is None:
            raise KeyError(item_id)
        decisions = {t: str(l).strip() for t, l in decisions.items() if str(l).strip()}
        if not decisions:
            raise ValueError("no decision given")
        for t, l in decisions.items():
            if t not in self.tasks:
                raise ValueError(f"unknown decision {t!r}; decisions are {sorted(self.tasks)}")
            if l not in self.tasks[t]:
                self.tasks[t].append(l)                     # a new label typed by the labeller
        row = {"text": it["text"], **decisions}
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        with open(self.out_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.done.add(item_id)
        self.skipped.discard(item_id)
        return row

    def skip(self, item_id: str) -> None:
        if item_id not in self.by_id:
            raise KeyError(item_id)
        self.skipped.add(item_id)

    def stats(self) -> Dict:
        ids = {it["id"] for it in self.items}
        done_here = len(ids & self.done)
        return {"total": len(self.items), "done": done_here, "skipped": len(self.skipped),
                "remaining": len(self.items) - done_here - len(self.skipped),
                "already_labelled": len(self.done) - done_here, "out": os.path.abspath(self.out_path),
                "suggested_tasks": self.suggested_tasks}


def create_label_app(queue: LabelQueue):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="tarski label")

    @app.get("/", response_class=HTMLResponse)
    def page():
        with open(STATIC) as f:
            return f.read()

    @app.get("/api/tasks")
    def tasks():
        return {"tasks": queue.tasks, "suggested": queue.suggested_tasks}

    @app.get("/api/next")
    def next_items(n: int = 1):
        return {"items": queue.next(n), "stats": queue.stats()}

    @app.post("/api/save")
    def save(req: SaveRequest):
        try:
            row = queue.save(req.id, req.decisions)
        except KeyError:
            raise HTTPException(404, "unknown message id")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"saved": row, "stats": queue.stats(), "tasks": queue.tasks}

    @app.post("/api/skip")
    def skip(req: SkipRequest):
        try:
            queue.skip(req.id)
        except KeyError:
            raise HTTPException(404, "unknown message id")
        return {"stats": queue.stats()}

    @app.get("/api/stats")
    def stats():
        return queue.stats()

    return app


def build_queue(data: str, out: str, task_specs: Sequence[str], store: Optional[str], base: Optional[str],
                device: Optional[str], text_col: str = "text", suggest: bool = True) -> LabelQueue:
    from tarski.engine import Engine, scan_store

    messages = read_messages(data, text_col)
    tasks = parse_task_specs(task_specs or [])
    engine = None
    if suggest and store and scan_store(store):
        engine = Engine(store, base, device)
    if not tasks and engine is None:
        raise ValueError("no decisions to label: pass --tasks name=label1,label2 ... or point --store at trained branches")
    return LabelQueue(messages, out, tasks, engine)
