"""Datasets as messages that carry one or more decisions.

A `Dataset` has tasks (a label set each) and messages; a message is labelled for any subset of the
tasks. That is the shape of routing: one Slack message or ticket, several decisions about it.

Loaders: a user CSV/JSONL file, Banking77, CLINC150 (intent + domain + out-of-scope, three decisions per
message) and typed-decisions (20 decisions over four workflows, five per message, with soft labels).
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Task:
    name: str
    labels: List[str]


@dataclass
class Example:
    text: str
    y: Dict[str, int]                                          # task -> label index
    soft: Dict[str, np.ndarray] = field(default_factory=dict)  # task -> target distribution, optional


@dataclass
class Dataset:
    name: str
    tasks: Dict[str, Task]
    train: List[Example]
    val: List[Example]
    test: List[Example]
    max_len: int = 256

    def split(self, name: str) -> List[Example]:
        return {"train": self.train, "val": self.val, "test": self.test}[name]

    def for_task(self, split: str, task: str) -> List[Example]:
        return [e for e in self.split(split) if task in e.y]

    def summary(self) -> str:
        per = ", ".join(f"{t.name}({len(t.labels)})" for t in self.tasks.values())
        return (f"{self.name}: {len(self.train)}/{len(self.val)}/{len(self.test)} messages "
                f"(train/val/test), tasks: {per}")


def _carve(rows: List, frac: float, seed: int):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = int(round(len(rows) * frac))
    return rows[n:], rows[:n]


# ---------------------------------------------------------------------------------------------------
# User data
# ---------------------------------------------------------------------------------------------------

def load_table(path: str, text_col: str = "text", task_cols: Optional[Sequence[str]] = None,
               split_col: str = "split", val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 0,
               max_len: int = 256) -> Dataset:
    """A CSV or JSONL file: one row per message, a text column, and one column per decision.

    Every column other than the text and split columns is a task unless `task_cols` names them.
    An empty cell means the message is not labelled for that task. If a `split` column holds
    train/val/test, it is used; otherwise rows are split at random (seeded).
    """
    if path.endswith((".jsonl", ".json")):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    if text_col not in rows[0]:
        raise ValueError(f"{path} has no {text_col!r} column; columns are {list(rows[0])}")
    cols = list(task_cols) if task_cols else [c for c in rows[0] if c not in (text_col, split_col)]
    if not cols:
        raise ValueError(f"{path} has no label columns besides {text_col!r}")

    def val(row, c):
        v = row.get(c)
        return None if v is None or str(v).strip() == "" else str(v).strip()

    tasks = {c: Task(c, sorted({val(r, c) for r in rows if val(r, c) is not None})) for c in cols}
    for t in tasks.values():
        if len(t.labels) < 2:
            raise ValueError(f"task {t.name!r} needs at least two distinct labels, found {t.labels}")
    index = {c: {l: i for i, l in enumerate(t.labels)} for c, t in tasks.items()}

    def example(r):
        return Example(str(r[text_col]), {c: index[c][val(r, c)] for c in cols if val(r, c) is not None})

    if rows[0].get(split_col) is not None and {str(r.get(split_col)) for r in rows} <= {"train", "val", "test"}:
        parts = {s: [example(r) for r in rows if r[split_col] == s] for s in ("train", "val", "test")}
        train, val_, test = parts["train"], parts["val"], parts["test"]
        if not val_:
            train, val_ = _carve(train, val_frac, seed)
    else:
        allx = [example(r) for r in rows]
        rest, test = _carve(allx, test_frac, seed)
        train, val_ = _carve(rest, val_frac / (1 - test_frac), seed + 1)
    return Dataset(os.path.basename(path), tasks, train, val_, test, max_len)


# ---------------------------------------------------------------------------------------------------
# Public benchmarks
# ---------------------------------------------------------------------------------------------------

def load_banking77(seed: int = 0) -> Dataset:
    from datasets import load_dataset

    ds = load_dataset("mteb/banking77")
    names = sorted({r["label_text"] for r in ds["train"]}, key=lambda s: s)
    by_id = {r["label"]: r["label_text"] for r in ds["train"]}
    labels = [by_id[i] for i in range(len(by_id))]
    assert sorted(labels) == names
    mk = lambda r: Example(r["text"], {"intent": int(r["label"])})
    train, val = _carve([mk(r) for r in ds["train"]], 0.1, seed)
    return Dataset("banking77", {"intent": Task("intent", labels)}, train, val, [mk(r) for r in ds["test"]], 64)


def load_clinc(seed: int = 0) -> Dataset:
    """CLINC150 "plus": 150 intents + out-of-scope. Three decisions per message: the intent (151 labels),
    its domain (10 + out-of-scope) and whether it is out of scope at all."""
    from datasets import load_dataset

    ds = load_dataset("clinc/clinc_oos", "plus")
    intents = ds["train"].features["intent"].names
    with open(os.path.join(os.path.dirname(__file__), "resources", "clinc_domains.json")) as f:
        domains = json.load(f)
    dom_of = {i: d for d, its in domains.items() for i in its}
    dom_labels = sorted(domains) + ["oos"]
    tasks = {"intent": Task("intent", intents), "domain": Task("domain", dom_labels),
             "oos": Task("oos", ["in_scope", "out_of_scope"])}

    def mk(r):
        name = intents[r["intent"]]
        oos = name == "oos"
        return Example(r["text"], {"intent": int(r["intent"]),
                                   "domain": dom_labels.index("oos" if oos else dom_of[name]),
                                   "oos": int(oos)})

    return Dataset("clinc150", tasks, [mk(r) for r in ds["train"]], [mk(r) for r in ds["validation"]],
                   [mk(r) for r in ds["test"]], 64)


def _option_keys(q: Dict) -> List[str]:
    if q["type"] == "choice":
        return list(q["criteria"].keys())
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def load_typed_decisions(seed: int = 0) -> Dataset:
    """LocalLLaMA/typed-decisions: 4 workflows x 5 fixed questions. Each message (a JSON state) carries the
    five decisions of its workflow, with the teacher's soft distribution as well as its argmax label."""
    from datasets import load_dataset

    tasks: Dict[str, Task] = {}

    def convert(split):
        out = []
        for r in load_dataset("LocalLLaMA/typed-decisions", "all", split=split):
            qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
            ex = Example(json.dumps(json.loads(r["state"]), ensure_ascii=False), {})
            for qid, q in qs.items():
                name = f"{r['workflow']}.{qid}"
                keys = _option_keys(q)
                tasks.setdefault(name, Task(name, keys))
                assert tasks[name].labels == keys, name
                probs = gold[qid]["probabilities"]
                p = np.array([float(probs.get(k, 0.0)) for k in keys], dtype=np.float32)
                ex.y[name] = keys.index(str(gold[qid]["label"]))
                ex.soft[name] = p / p.sum()
            out.append(ex)
        return out

    train, val = _carve(convert("train"), 0.1, seed)
    return Dataset("typed-decisions", tasks, train, val, convert("test"), 512)


BENCHMARKS = {"banking77": load_banking77, "clinc150": load_clinc, "typed-decisions": load_typed_decisions}


def load(name_or_path: str, **kw) -> Dataset:
    if name_or_path in BENCHMARKS:
        return BENCHMARKS[name_or_path]()
    return load_table(name_or_path, **kw)
