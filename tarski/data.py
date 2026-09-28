"""Strict JSON/JSONL validation and atomic output for agent workflows."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from tarski.config import TYPES, require


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def decode(text):
    def invalid(value):
        raise ValueError(f"non-finite JSON number: {value}")
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            invalid(value)
        return parsed
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid, parse_float=number)


def load_json(path):
    return decode(Path(path).read_text(encoding="utf-8"))


def load_rows(path, kind):
    text = sys.stdin.read() if str(path) == "-" else Path(path).read_text(encoding="utf-8")
    rows, ids = [], set()
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = decode(line)
            validate_row(row, kind)
            require(row["id"] not in ids, f"duplicate id: {row['id']}")
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        ids.add(row["id"])
        rows.append(row)
    require(rows, f"{path}: no records")
    return rows


def probabilities(values, labels, where):
    require(isinstance(values, list) and len(values) == len(labels), f"{where}: probability count must match labels")
    require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in values),
            f"{where}: probabilities must be finite numbers in [0, 1]")
    require(abs(sum(values) - 1) <= 1e-4, f"{where}: probabilities must sum to 1")


def labels_for(question):
    kind = question.get("type")
    require(kind in TYPES, f"question.type: choose from {TYPES}")
    if kind == "noul":
        require(isinstance(question.get("criteria", {}), dict), "noul.criteria: expected an object")
        return ["yes", "no"]
    criteria = question.get("criteria")
    if kind == "choice":
        require(isinstance(criteria, dict) and 2 <= len(criteria) <= 255, "choice.criteria: expected 2..255 named choices")
        require(all(isinstance(k, str) and k for k in criteria), "choice.criteria: labels must be nonempty strings")
        return list(criteria)
    require(isinstance(criteria, list) and 2 <= len(criteria) <= 10, "score.criteria: expected an array of 2..10 levels")
    return [str(i) for i in range(len(criteria))]


def gold_label(value):
    return value.get("label") if isinstance(value, dict) else value


def validate_row(row, kind):
    require(isinstance(row, dict), "record: expected an object")
    require(isinstance(row.get("id"), str) and bool(row["id"]), "record.id: expected a nonempty string")
    questions = row.get("questions")
    if kind == "records":
        require(isinstance(row.get("state"), (str, dict, list)), "record.state: expected text, an object or an array")
        require(isinstance(questions, dict) and bool(questions), "record.questions: expected a nonempty object")
        gold = row.get("gold", {})
        require(isinstance(gold, dict) and set(gold) <= set(questions), "record.gold: expected an object keyed by question id")
        for qid, q in questions.items():
            require(bool(qid) and isinstance(q, dict), "questions: each id must name a question object")
            labels = labels_for(q)
            value = gold_label(gold.get(qid))
            require(value is None or value in labels, f"gold.{qid}: label must be one of {labels}")
    else:
        require(isinstance(questions, list) and bool(questions), "record.questions: expected a nonempty array of stored reads")
        qids = set()
        for q in questions:
            require(isinstance(q, dict), "question: expected an object")
            qid = q.get("qid")
            require(isinstance(qid, str) and qid and qid not in qids, "question.qid: expected a unique nonempty string")
            qids.add(qid)
            require(q.get("type") in TYPES, "question.type: expected choice, noul or score")
            labels = q.get("labels")
            require(isinstance(labels, list) and len(labels) >= 2 and all(isinstance(x, str) and x for x in labels),
                    f"{qid}.labels: expected at least two nonempty strings")
            require(len(labels) == len(set(labels)), f"{qid}.labels: duplicate labels")
            require(q.get("gold") is None or q["gold"] in labels, f"{qid}.gold: unknown label")
            require(isinstance(q.get("reads"), dict) and bool(q["reads"]), f"{qid}.reads: expected named probability vectors")
            for name, values in q["reads"].items():
                probabilities(values, labels, f"{qid}.reads.{name}")
        for name in ("state_sha256", "input_sha256"):
            if name in row:
                require(isinstance(row[name], str) and len(row[name]) == 64 and all(c in "0123456789abcdef" for c in row[name]),
                        f"{name}: expected a SHA-256 digest")
    return row


def write_output(path, value, *, lines=False, force=False):
    content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in value) if lines else \
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if str(path) == "-":
        sys.stdout.write(content)
        return
    path = Path(path)
    require(force or not path.exists(), f"output exists: {path}; choose another path or pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        if force:
            os.replace(temp, path)
        else:
            os.link(temp, path)  # exclusive creation, even if another process races us
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
