"""Versioned, editable policies. No model dependencies are imported here."""

from __future__ import annotations

import math
import re

MODELS = {
    "mlx": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
    "torch": "google/diffusiongemma-26B-A4B-it",
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
}
FORMATS = ("stock", "state_first", "user_first")
SLOTS = ("native", "mean")
TYPES = ("choice", "noul", "score")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def keys(value, allowed, required, where):
    require(isinstance(value, dict), f"{where}: expected an object")
    require(not (set(value) - set(allowed)), f"{where}: unknown keys {sorted(set(value) - set(allowed))}")
    require(set(required) <= set(value), f"{where}: missing keys {sorted(set(required) - set(value))}")


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def valid_order(order):
    return isinstance(order, str) and (order in ("given", "reversed") or re.fullmatch(r"rotate:[0-9]{1,4}", order) is not None)


def default_config(backend="mlx"):
    require(backend in MODELS, f"unknown backend: {backend}")
    return {
        "schema_version": 1,
        "backend": backend,
        "model": MODELS[backend],
        "read": {"prompt_format": "stock" if backend == "llada" else "state_first",
                 "question_order": "given", "slot": "native" if backend == "llada" else "mean", "seed": 0},
        "calibration": {"default_temperature": 1.0, "temperatures": {}},
    }


def validate_config(config):
    keys(config, ("schema_version", "backend", "model", "read", "calibration", "fit"),
         ("schema_version", "backend", "model", "read", "calibration"), "config")
    require(type(config["schema_version"]) is int and config["schema_version"] == 1, "config: schema_version must be 1")
    require(isinstance(config["backend"], str) and config["backend"] in MODELS, "config: backend must be mlx, torch or llada")
    require(isinstance(config["model"], str) and bool(config["model"].strip()), "config: model must be a nonempty string")
    read = config["read"]
    fields = ("prompt_format", "question_order", "slot", "seed")
    keys(read, fields, fields, "read")
    require(read["prompt_format"] in FORMATS, f"read.prompt_format: choose from {FORMATS}")
    require(valid_order(read["question_order"]), "read.question_order: use given, reversed or rotate:N (0..9999)")
    require(read["slot"] in SLOTS, "read.slot: use native or mean")
    require(type(read["seed"]) is int and 0 <= read["seed"] < 2**32, "read.seed: expected integer in [0, 2**32)")
    cal = config["calibration"]
    keys(cal, ("default_temperature", "temperatures"), ("default_temperature", "temperatures"), "calibration")
    require(positive(cal["default_temperature"]), "calibration.default_temperature: expected a finite positive number")
    keys(cal["temperatures"], TYPES, (), "calibration.temperatures")
    for kind, value in cal["temperatures"].items():
        require(positive(value), f"calibration.temperatures.{kind}: expected a finite positive number")
    if "fit" in config:
        fit = config["fit"]
        fields = ("data_sha256", "id_hashes", "state_hashes", "condition", "reader", "objective", "states", "labelled_questions")
        keys(fit, fields, fields, "fit")
        require(isinstance(fit["data_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", fit["data_sha256"]) is not None,
                "fit.data_sha256: expected a SHA-256 digest")
        for name in ("id_hashes", "state_hashes"):
            require(isinstance(fit[name], list) and all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) for x in fit[name]),
                    f"fit.{name}: expected a list of SHA-256 digests")
        require(fit["condition"] == condition(read), "fit.condition differs from read settings; retune or remove fit and reset calibration")
        require(fit["reader"] == {"backend": config["backend"], "model": config["model"], "seed": read["seed"]},
                "fit.reader differs from backend/model/seed; retune or remove fit and reset calibration")
        require(fit["objective"] in ("accuracy", "brier"), "fit.objective: expected accuracy or brier")
        for name in ("states", "labelled_questions"):
            require(type(fit[name]) is int and fit[name] > 0, f"fit.{name}: expected a positive integer")
    return config


def condition(read):
    order = read["question_order"]
    order = {"given": "rot0", "reversed": "rev"}.get(order, order.replace("rotate:", "rot"))
    slot = "random" if read["slot"] == "native" else "mean"
    return f"{read['prompt_format']}|{order}|{slot}"


def parse_condition(name):
    """Understand the order, state-first and Bitext experiment file formats."""
    if "|" in name:
        parts = name.split("|")
        require(len(parts) == 3, f"unsupported condition: {name}")
        fmt, order, slot = parts
    else:
        fmt, order, slot = "stock", name.removesuffix("_mean"), "mean" if name.endswith("_mean") else "random"
    order = {"given": "rot0", "reversed": "rev"}.get(order, order)
    require(fmt in FORMATS and slot in ("random", "mean"), f"unsupported condition: {name}")
    require(order == "rev" or re.fullmatch(r"rot[0-9]{1,4}", order) is not None, f"unsupported condition: {name}")
    return {"prompt_format": fmt, "question_order": "reversed" if order == "rev" else
            ("given" if int(order[3:]) == 0 else f"rotate:{int(order[3:])}"),
            "slot": "native" if slot == "random" else "mean"}
