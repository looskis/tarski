"""Noninteractive CLI: JSON results on stdout, errors/progress on stderr."""

from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys

from tarski import __version__
from tarski.config import FORMATS, MODELS, SLOTS, condition, default_config, require, validate_config
from tarski.data import load_json, load_rows, write_output
from tarski.policy import evaluate, tune


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def parser():
    p = Parser(prog="tarski", description="Fit and use diffusion decision read policies on your data.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("schema", help="print the machine-readable command and data contract")
    init = sub.add_parser("init", help="write a versioned starter config (no model load)")
    init.add_argument("--backend", choices=list(MODELS), default="mlx")
    init.add_argument("--model", help="override model ID or local path")
    init.add_argument("--seed", type=int, default=0)
    validate = sub.add_parser("validate", help="validate a config or JSONL data without loading a model")
    target = validate.add_mutually_exclusive_group(required=True)
    target.add_argument("--config")
    target.add_argument("--data", help="JSONL path, or - for stdin")
    validate.add_argument("--kind", choices=["records", "reads"], default="records")
    doctor = sub.add_parser("doctor", help="check reader dependencies without loading weights")
    doctor.add_argument("--config", required=True)
    collect = sub.add_parser("collect", help="read local records under candidate settings")
    collect.add_argument("--config", required=True)
    collect.add_argument("--data", required=True, help="records JSONL, or - for stdin")
    collect.add_argument("--formats", nargs="+", choices=FORMATS, help="default: the config's prompt format")
    collect.add_argument("--orders", nargs="+", help="given, reversed or rotate:N; default: the config's order")
    collect.add_argument("--slots", nargs="+", choices=SLOTS, help="default: the config's slot")
    fit = sub.add_parser("tune", help="select settings and fit per-type temperatures on calibration reads")
    fit.add_argument("--config", required=True)
    fit.add_argument("--reads", required=True, help="calibration reads JSONL, or - for stdin")
    fit.add_argument("--objective", choices=["accuracy", "brier"], default="accuracy")
    ev = sub.add_parser("eval", help="score a config on held-out reads; reject calibration overlap")
    ev.add_argument("--config", required=True)
    ev.add_argument("--reads", required=True, help="held-out reads JSONL, or - for stdin")
    pred = sub.add_parser("predict", help="make calibrated decisions for local records")
    pred.add_argument("--config", required=True)
    pred.add_argument("--data", required=True, help="records JSONL, or - for stdin")
    for command in (init, collect, fit):
        command.add_argument("--out", required=True, help="output path (existing files require --force)")
        command.add_argument("--force", action="store_true")
    pred.add_argument("--out", default="-", help="JSONL output; default: stdout")
    pred.add_argument("--force", action="store_true")
    return p


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))


def schema():
    return {
        "schema_version": 1, "version": __version__,
        "commands": {
            "init": {"required": ["--out"], "optional": ["--backend", "--model", "--seed", "--force"]},
            "validate": {"one_of": ["--config", "--data"], "optional": ["--kind"]},
            "doctor": {"required": ["--config"]},
            "collect": {"required": ["--config", "--data", "--out"], "optional": ["--formats", "--orders", "--slots", "--force"]},
            "tune": {"required": ["--config", "--reads", "--out"], "optional": ["--objective", "--force"]},
            "eval": {"required": ["--config", "--reads"]},
            "predict": {"required": ["--config", "--data"], "optional": ["--out", "--force"]},
        },
        "config_example": default_config(),
        "record_example": {"id": "ticket-1", "state": "I was charged twice.",
                           "questions": {"refund": {"type": "noul", "instructions": "Does this need a refund?"}},
                           "gold": {"refund": "yes"}},
        "reads_example": {"id": "ticket-1", "questions": [
            {"qid": "refund", "type": "noul", "labels": ["yes", "no"], "gold": "yes",
             "reads": {"state_first|rot0|mean": [0.9, 0.1]}}]},
        "question_types": {"noul": {"labels": ["yes", "no"]},
                           "choice": {"criteria": "object of 2..255 label: description pairs"},
                           "score": {"criteria": "array of 2..10 level descriptions", "labels": "0, 1, ... as strings"}},
        "constraints": {"record_ids": "unique nonempty strings; keep IDs stable across collect runs",
                        "gold": "optional; label string or object with a label field; null means unlabelled",
                        "state": "string, object or array", "data_format": "one JSON object per line",
                        "prompt_formats": list(FORMATS), "slots": list(SLOTS),
                        "orders": ["given", "reversed", "rotate:N (0..9999)"],
                        "native": "random vocabulary token on DiffusionGemma; mask token on LLaDA",
                        "calibration": "per-type temperatures fitted by NLL; selection metrics are in-sample",
                        "evaluation": "use disjoint states; IDs and available state hashes are checked",
                        "runtime": "one canvas per record; no automatic question splitting"},
        "io": {"stdout": "JSON result or prediction JSONL", "stderr": "JSON errors, progress, model logs",
               "stdin": "--data - or --reads -", "exit_codes": {"0": "success", "1": "runtime/dependency failure", "2": "invalid input", "130": "interrupted"},
               "writes": "atomic; existing paths require --force; collect and predict may download model weights"},
    }


def check_output(args):
    if hasattr(args, "out"):
        require(args.command == "predict" or args.out != "-", "--out must be a file path for this command")
        require(args.out == "-" or args.force or not Path(args.out).exists(), f"output exists: {args.out}; use --force to replace")
        for name in ("config", "data", "reads"):
            source = getattr(args, name, None)
            if source and source != "-" and args.out != "-":
                require(Path(source).resolve() != Path(args.out).resolve(), "output must not overwrite an input file")


def run(args):
    check_output(args)
    if args.command == "schema":
        emit(schema())
        return 0
    if args.command == "init":
        config = default_config(args.backend)
        config["read"]["seed"] = args.seed
        if args.model:
            config["model"] = args.model
        validate_config(config)
        write_output(args.out, config, force=args.force)
        emit({"config": args.out, "policy": config})
        return 0
    config = validate_config(load_json(args.config)) if getattr(args, "config", None) else None
    if args.command == "validate":
        if config is not None:
            emit({"valid": True, "kind": "config", "schema_version": config["schema_version"]})
        else:
            rows = load_rows(args.data, args.kind)
            emit({"valid": True, "kind": args.kind, "states": len(rows), "questions": sum(len(r["questions"]) for r in rows)})
        return 0
    if args.command == "doctor":
        from tarski.runtime import doctor
        result = doctor(config)
        emit(result)
        return 0 if result["ready"] else 1
    if args.command in ("tune", "eval"):
        rows = load_rows(args.reads, "reads")
        if args.command == "eval":
            emit(evaluate(rows, config))
        else:
            tuned, report = tune(rows, config, args.objective)
            write_output(args.out, tuned, force=args.force)
            emit({"config": args.out, **report})
        return 0
    records = load_rows(args.data, "records")
    from tarski.runtime import collect, predictions
    read = config["read"]
    candidates = [read]
    if args.command == "collect":
        candidates = []
        for fmt, order, slot in product(args.formats or [read["prompt_format"]],
                                       args.orders or [read["question_order"]], args.slots or [read["slot"]]):
            candidate = {**read, "prompt_format": fmt, "question_order": order, "slot": slot}
            if candidate not in candidates:
                candidates.append(candidate)
        # Validate before importing or loading the reader.
        for candidate in candidates:
            trial = deepcopy(config)
            trial.pop("fit", None)
            trial["read"] = candidate
            validate_config(trial)
    rows = collect(records, config, candidates)
    result = predictions(rows, config) if args.command == "predict" else rows
    write_output(args.out, result, lines=True, force=args.force)
    if args.out != "-":
        emit({"output": args.out, "states": len(rows), "conditions": [condition(r) for r in candidates]})
    return 0


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": {"code": "invalid_input", "message": str(exc)}}), file=sys.stderr)
        return 2
    except (RuntimeError, ImportError) as exc:
        print(json.dumps({"error": {"code": "runtime_error", "message": str(exc)}}), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(json.dumps({"error": {"code": "interrupted", "message": "Interrupted."}}), file=sys.stderr)
        return 130
