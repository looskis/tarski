from copy import deepcopy
import math
from pathlib import Path

import pytest

from tarski.config import condition, default_config, parse_condition, validate_config
from tarski.data import digest, load_rows
from tarski.policy import evaluate, fit_temperature, metrics, scale, tune

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rows():
    return load_rows(ROOT / "examples/calibration.reads.jsonl", "reads")


def test_tune_export_and_heldout_evaluation(rows):
    original = deepcopy(rows)
    config, report = tune(rows, default_config())
    assert rows == original
    assert report["scope"] == "calibration"
    assert report["selected"] == "state_first|rot0|mean"
    assert report["calibrated"]["nll"] <= report["candidates"][report["selected"]]["nll"]
    heldout = load_rows(ROOT / "examples/evaluation.reads.jsonl", "reads")
    result = evaluate(heldout, config)
    assert result["scope"] == "evaluation"
    assert result["calibrated"]["accuracy"] == result["raw"]["accuracy"] == 1.0
    assert validate_config(config) == config


def test_evaluation_rejects_fit_ids_and_state_reuse(rows):
    rows[0]["state_sha256"] = digest("same text")
    config, _ = tune(rows, default_config())
    with pytest.raises(ValueError, match="overlaps calibration"):
        evaluate(rows, config)
    renamed = deepcopy(rows[:1])
    renamed[0]["id"] = "different-id"
    with pytest.raises(ValueError, match="overlaps calibration"):
        evaluate(renamed, config)


def test_reader_mismatch_rejected(rows):
    config = default_config("torch")
    with pytest.raises(ValueError, match="provenance does not match"):
        tune(rows, config)
    rows[0].pop("reader")
    with pytest.raises(ValueError, match="mixed reads"):
        tune(rows, default_config())


def test_partial_runs_do_not_change_scoring_population(rows):
    rows[-1]["questions"][0]["reads"].pop("stock|rot0|random")
    with pytest.raises(ValueError, match="candidate conditions differ"):
        tune(rows, default_config())


def test_missing_labels_are_not_treated_as_incorrect(rows):
    rows[0]["questions"][0]["gold"] = None
    _, report = tune(rows, default_config())
    assert report["calibrated"]["questions"] == 5
    for row in rows:
        row["questions"][0]["gold"] = None
    with pytest.raises(ValueError, match="no labelled questions"):
        tune(rows, default_config())


def test_research_names_and_skipped_hybrids(rows):
    for row in rows:
        row.pop("reader")
        q = row["questions"][0]
        q["reads"] = {"rot0": q["reads"]["stock|rot0|random"], "rev_mean": q["reads"]["state_first|rot0|mean"],
                      "prompt_rev": [0.5, 0.5]}
    config, report = tune(rows, default_config())
    assert config["read"]["question_order"] == "reversed"
    assert report["skipped_conditions"] == ["prompt_rev"]
    assert "no reader provenance" in report["warnings"][0]


@pytest.mark.parametrize("name,expected", [
    ("rot0_mean", "stock|rot0|mean"), ("rev", "stock|rev|random"),
    ("state_first|given|mean", "state_first|rot0|mean"),
    ("stock|reversed|random", "stock|rev|random"),
    ("user_first|rot3|random", "user_first|rot3|random"),
])
def test_experiment_condition_aliases(name, expected):
    assert condition(parse_condition(name)) == expected


def test_backend_defaults():
    assert default_config("llada")["read"]["slot"] == "native"
    assert default_config("llada")["read"]["prompt_format"] == "stock"
    assert default_config("torch")["read"]["slot"] == "mean"


@pytest.mark.parametrize("temp", [0.000001, 1, 2, 1e300, 1e-300])
def test_calibration_numerics_and_argmax(temp):
    p = scale([0, 0.05, 0.95], temp)
    assert all(math.isfinite(x) for x in p)
    assert sum(p) == pytest.approx(1)
    # Extreme high temperatures can round distinct probabilities to equal values.
    if temp < 1e100:
        assert max(range(3), key=p.__getitem__) == 2


def test_temperature_reduces_overconfident_nll():
    items = [([.99, .01], 0), ([.99, .01], 1), ([.99, .01], 0)]
    temp = fit_temperature(items)
    assert temp > 1
    assert metrics([(scale(p, temp), y) for p, y in items])["nll"] < metrics(items)["nll"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1, True, "2"])
def test_invalid_temperatures(bad):
    config = default_config()
    config["calibration"]["temperatures"]["choice"] = bad
    with pytest.raises(ValueError, match="finite positive"):
        validate_config(config)


def test_typos_and_changed_fitted_settings_rejected(rows):
    config = default_config()
    config["read"]["slots"] = "mean"
    with pytest.raises(ValueError, match="unknown keys"):
        validate_config(config)
    fitted, _ = tune(rows, default_config())
    fitted["read"]["question_order"] = "reversed"
    with pytest.raises(ValueError, match="retune"):
        validate_config(fitted)


@pytest.mark.parametrize("field,value", [("backend", "torch"), ("model", "different-checkpoint"), ("seed", 5)])
def test_fit_is_bound_to_its_reader(rows, field, value):
    config, _ = tune(rows, default_config())
    if field == "seed":
        config["read"][field] = value
    else:
        config[field] = value
    with pytest.raises(ValueError, match="fit.reader differs"):
        validate_config(config)
