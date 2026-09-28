"""Select read settings on calibration data and score a separate evaluation set."""

from __future__ import annotations

from copy import deepcopy
import math

from tarski.config import condition, parse_condition, require, validate_config
from tarski.data import digest


def scale(probs, temperature):
    logs = [math.log(max(p, 1e-300)) for p in probs]
    top = max(logs)
    weights = [math.exp((x - top) / temperature) for x in logs]
    total = sum(weights)
    return [x / total for x in weights]


def metrics(items):
    """Items are (probabilities, gold index); ECE uses ten equal-width bins."""
    require(items, "no labelled questions to score")
    correct = 0
    brier = nll = 0.0
    bins = [[] for _ in range(10)]
    for p, y in items:
        top = max(range(len(p)), key=p.__getitem__)
        hit = int(top == y)
        correct += hit
        brier += sum((v - int(i == y)) ** 2 for i, v in enumerate(p))
        nll -= math.log(max(p[y], 1e-300))
        bins[min(int(p[top] * 10), 9)].append((p[top], hit))
    n = len(items)
    ece = sum(abs(sum(conf - hit for conf, hit in bucket)) for bucket in bins) / n
    return {"questions": n, "accuracy": correct / n, "brier": brier / n, "nll": nll / n, "ece": ece}


def canonical_reads(question):
    out, skipped = {}, []
    for name, probs in question["reads"].items():
        try:
            key = condition(parse_condition(name))
        except ValueError:
            skipped.append(name)
            continue
        require(key not in out, f"{question['qid']}: multiple names resolve to {key}")
        out[key] = probs
    return out, skipped


def reader_identity(config):
    return {"backend": config["backend"], "model": config["model"], "seed": config["read"]["seed"]}


def check_provenance(rows, config):
    expected = reader_identity(config)
    known = [row.get("reader") for row in rows]
    require(all(x is None for x in known) or all(x is not None for x in known), "mixed reads with and without reader provenance")
    for identity in known:
        if identity is not None:
            require(identity == expected, f"reader provenance does not match config: expected {expected}, got {identity}")
    return [] if known[0] is not None else ["Reads have no reader provenance; verify backend, model and seed match the config."]


def question_items(rows, selected, temperatures=None, default_temperature=1.0):
    items, by_type = [], {}
    for row in rows:
        for q in row["questions"]:
            reads, _ = canonical_reads(q)
            require(selected in reads, f"{row['id']}/{q['qid']}: missing condition {selected}")
            if q.get("gold") is None:
                continue
            p = reads[selected]
            if temperatures is not None:
                p = scale(p, temperatures.get(q["type"], default_temperature))
            item = (p, q["labels"].index(q["gold"]))
            items.append(item)
            by_type.setdefault(q["type"], []).append(item)
    require(items, "no labelled questions; supply gold labels to tune or evaluate")
    return items, by_type


def fit_temperature(items):
    # Include identity explicitly so the fitted NLL cannot be worse than raw.
    candidates = sorted({1.0, *(math.exp(-3 + 7 * i / 140) for i in range(141))})
    def objective(t):
        nll = -sum(math.log(max(scale(p, t)[y], 1e-300)) for p, y in items)
        return nll, abs(math.log(t))
    return min(candidates, key=objective)


def tune(rows, config, objective="accuracy"):
    validate_config(config)
    require(objective in ("accuracy", "brier"), "objective must be accuracy or brier")
    warnings = check_provenance(rows, config)
    candidates, skipped = None, set()
    for row in rows:
        for q in row["questions"]:
            reads, ignored = canonical_reads(q)
            skipped.update(ignored)
            if candidates is None:
                candidates = set(reads)
            require(candidates == set(reads), f"{row['id']}/{q['qid']}: candidate conditions differ; collect every candidate on every state")
    require(candidates, "no deployable read conditions found (expected rotN, rev, or format|order|slot)")
    scores = {name: metrics(question_items(rows, name)[0]) for name in sorted(candidates)}
    def rank(name):
        m = scores[name]
        return (-m["accuracy"], m["brier"], name) if objective == "accuracy" else (m["brier"], -m["accuracy"], name)
    selected = min(scores, key=rank)
    _, grouped = question_items(rows, selected)
    temperatures = {kind: fit_temperature(items) for kind, items in sorted(grouped.items())}
    for kind, items in grouped.items():
        if len(items) < 30:
            warnings.append(f"Only {len(items)} labelled {kind} questions; the fitted temperature may overfit.")
    result = deepcopy(config)
    result["read"].update(parse_condition(selected))
    result["calibration"] = {"default_temperature": 1.0, "temperatures": temperatures}
    result["fit"] = {
        "data_sha256": digest(rows), "id_hashes": sorted(digest(row["id"]) for row in rows),
        "state_hashes": sorted({row["state_sha256"] for row in rows if "state_sha256" in row}),
        "condition": selected, "objective": objective, "states": len(rows),
        "reader": reader_identity(result),
        "labelled_questions": sum(len(items) for items in grouped.values()),
    }
    report = {"scope": "calibration", "states": len(rows), "selected": selected,
              "objective": objective, "candidates": scores, "temperatures": temperatures,
              "calibrated": metrics(question_items(rows, selected, temperatures)[0]),
              "skipped_conditions": sorted(skipped), "warnings": warnings,
              "next_step": "Evaluate this config on a separate labelled reads file with tarski eval."}
    return validate_config(result), report


def evaluate(rows, config):
    validate_config(config)
    warnings = check_provenance(rows, config)
    fit = config.get("fit", {})
    seen_ids, seen_states = set(fit.get("id_hashes", [])), set(fit.get("state_hashes", []))
    overlap = [row["id"] for row in rows if digest(row["id"]) in seen_ids or row.get("state_sha256") in seen_states]
    require(not overlap, f"evaluation overlaps calibration data ({len(overlap)} states); use a separate held-out file")
    selected = condition(config["read"])
    cal = config["calibration"]
    raw, _ = question_items(rows, selected)
    items, grouped = question_items(rows, selected, cal["temperatures"], cal["default_temperature"])
    missing = set(grouped) - set(cal["temperatures"])
    if missing:
        warnings.append(f"No fitted temperature for {', '.join(sorted(missing))}; using default_temperature.")
    return {"scope": "evaluation", "states": len(rows), "condition": selected,
            "raw": metrics(raw), "calibrated": metrics(items),
            "by_type": {kind: metrics(values) for kind, values in sorted(grouped.items())},
            "warnings": warnings}
