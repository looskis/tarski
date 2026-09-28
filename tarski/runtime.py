"""Thin adapter around the readers used to produce the research results."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys

from tarski.config import condition, require, validate_config
from tarski.data import digest, gold_label, probabilities
from tarski.policy import reader_identity, scale


def doctor(config):
    modules = ["httpx", "transformers"] + (["mlx", "mlx_vlm"] if config["backend"] == "mlx" else ["torch"])
    available = {module: importlib.util.find_spec(module) is not None for module in modules}
    checkout = Path(__file__).resolve().parents[1] / "third_party" / "openjev" / "openjev" / "engine.py"
    available["openjev"] = importlib.util.find_spec("openjev") is not None or checkout.is_file()
    return {"backend": config["backend"], "model": config["model"], "dependencies": available,
            "ready": all(available.values()), "weights_checked": False,
            "note": "Dependency check only; collect/predict load weights and may download them on first use."}


def load_reader(config):
    info = doctor(config)
    missing = [key for key, value in info["dependencies"].items() if not value]
    if missing:
        raise RuntimeError(f"Missing reader dependencies: {', '.join(missing)}. See README.md 'Run on your data' for installation.")
    from dlm.backend import get_reader
    with redirect_stdout(sys.stderr):
        return get_reader(config["backend"], model_path=config["model"])


def ordered(questions, order):
    qids = list(questions)
    if order == "reversed":
        qids.reverse()
    elif order.startswith("rotate:"):
        k = int(order.split(":")[1]) % len(qids)
        qids = qids[k:] + qids[:k]
    return {qid: questions[qid] for qid in qids}


def collect(records, config, candidates, reader=None):
    """Collect raw probabilities; calibration is applied only at prediction/evaluation."""
    validate_config(config)
    for read in candidates:
        trial = {**config, "read": read}
        trial.pop("fit", None)
        validate_config(trial)
        require(read["seed"] == config["read"]["seed"], "all candidates must use the config's seed")
    from dlm.state_first import prepare
    reader = reader if reader is not None else load_reader(config)
    rows = []
    for index, record in enumerate(records, 1):
        state = record["state"] if isinstance(record["state"], str) else json.dumps(record["state"], ensure_ascii=False)
        out = {}
        for read in candidates:
            with redirect_stdout(sys.stderr):
                prep = prepare(reader, state, ordered(record["questions"], read["question_order"]), read["prompt_format"])
                require(not prep["forced"], "single-choice questions are not supported by this CLI")
                require(len(reader.engine.groups(prep["qs"], prep["fmt"])) == 1,
                        "questions need multiple canvases; reduce the question set")
                probs = reader.read(prep, "random" if read["slot"] == "native" else "mean", seed=read["seed"])
            require(len(probs) == len(prep["qs"]), "reader returned an unexpected number of answers")
            for q, p in zip(prep["qs"], probs):
                qid = q["key"]
                labels = [name for name, _ in q["choices"]]
                probabilities(p, labels, f"reader answer {qid}")
                item = out.setdefault(qid, {"qid": qid, "type": q["type"], "labels": labels,
                                           "gold": gold_label(record.get("gold", {}).get(qid)), "reads": {}})
                require(item["labels"] == labels, f"reader label order changed for {qid}")
                item["reads"][condition(read)] = p
        require(set(out) == set(record["questions"]), "reader returned a different question set")
        rows.append({"id": record["id"], "input_sha256": digest(record), "state_sha256": digest(record["state"]),
                     "reader": reader_identity(config), "questions": [out[qid] for qid in record["questions"]]})
        print(json.dumps({"progress": {"completed": index, "total": len(records)}}), file=sys.stderr, flush=True)
    return rows


def predictions(rows, config):
    key, cal = condition(config["read"]), config["calibration"]
    output = []
    for row in rows:
        decisions = {}
        for q in row["questions"]:
            temperature = cal["temperatures"].get(q["type"], cal["default_temperature"])
            p = scale(q["reads"][key], temperature)
            top = max(range(len(p)), key=p.__getitem__)
            decisions[q["qid"]] = {"label": q["labels"][top], "probabilities": dict(zip(q["labels"], p)), "confidence": p[top]}
        output.append({"id": row["id"], "decisions": decisions})
    return output
