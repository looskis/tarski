"""Decision datasets as (state, Jev-format questions, gold) records.

  jevbench_public()      231 public JevBench v1 items (72 original, 48 easy, 111 hard), one question each
  typed_decisions(split) LocalLLaMA/typed-decisions: 1,200 train / 400 test states, five questions each,
                         with the teacher's soft label distributions

A record: {"id", "source", "family", "state": str, "questions": {qid: {type, instructions, criteria}},
           "gold": {qid: {"label": str, "probs": {label: p} | None}}}
Labels are named as OpenJev names them: yes/no for noul, the criteria keys for choice, "0".. for score.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JEVBENCH = os.path.join(ROOT, "third_party", "jevbench")
if JEVBENCH not in sys.path:
    sys.path.insert(0, JEVBENCH)


def _state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def jevbench_public(tiers=("original", "easy", "hard")) -> List[Dict]:
    from jevbench.tasks import load_jsonl

    out = []
    for tier in tiers:
        for t in load_jsonl(os.path.join(JEVBENCH, "datasets", "public", f"{tier}.jsonl")):
            expected = t.expected
            if t.question["type"] == "score" and expected is not None:
                expected = str(expected)
            out.append({"id": t.id, "source": f"jevbench/{tier}", "family": t.family, "group": t.group,
                        "state": _state_text(t.state), "questions": {t.id: t.question},
                        "labels": list(t.labels),
                        "gold": {t.id: {"label": expected, "probs": None}}})
    return out


def typed_decisions(split: str = "test") -> List[Dict]:
    from datasets import load_dataset

    rows = load_dataset("LocalLLaMA/typed-decisions", "all", split=split)
    out = []
    for r in rows:
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        g = {}
        for qid, a in gold.items():
            probs = a.get("probabilities")
            if qs[qid]["type"] == "noul":
                p = float(a.get("noul", probs.get("true", probs.get("yes")) if probs else 0.5))
                probs = {"yes": p, "no": 1.0 - p}
                label = "yes" if str(a["label"]).lower() in ("true", "yes", "1") else "no"
            else:
                label = str(a["label"])
                probs = {str(k): float(v) for k, v in probs.items()} if probs else None
            g[qid] = {"label": label, "probs": probs}
        out.append({"id": f"{r['workflow']}/{r['id']}", "source": f"typed-decisions/{split}",
                    "family": r["workflow"], "group": None, "state": _state_text(json.loads(r["state"])),
                    "questions": qs, "labels": None, "gold": g})
    return out


def label_names(question: Dict) -> List[str]:
    """Label names in the order OpenJev's engine lays them out."""
    t = question["type"]
    if t == "noul":
        return ["yes", "no"]
    if t == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(question["criteria"])
