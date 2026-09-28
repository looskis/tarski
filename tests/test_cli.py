import json
from pathlib import Path
import subprocess
import sys

import pytest

from tarski.cli import main
from tarski.data import load_rows, write_output

ROOT = Path(__file__).resolve().parents[1]


def test_offline_cli_workflow(tmp_path, capsys):
    base, fitted = tmp_path / "base.json", tmp_path / "fitted.json"
    assert main(["init", "--out", str(base)]) == 0
    assert main(["tune", "--config", str(base), "--reads", str(ROOT / "examples/calibration.reads.jsonl"),
                 "--out", str(fitted)]) == 0
    assert main(["eval", "--config", str(fitted), "--reads", str(ROOT / "examples/evaluation.reads.jsonl")]) == 0
    output = capsys.readouterr()
    assert not output.err
    assert len([json.loads(x) for x in output.out.splitlines()]) == 3
    assert main(["eval", "--config", str(fitted), "--reads", str(ROOT / "examples/calibration.reads.jsonl")]) == 2
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["error"]["code"] == "invalid_input"


def test_machine_discovery_needs_no_model_modules():
    code = "from tarski.cli import main; import sys; main(['schema']); assert not any(m in sys.modules for m in ['torch', 'mlx', 'transformers', 'numpy'])"
    process = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "predict" in json.loads(process.stdout)["commands"]


def test_stdin_records():
    process = subprocess.run([sys.executable, "-m", "tarski", "validate", "--data", "-"],
                             input=(ROOT / "examples/decisions.jsonl").read_text(), capture_output=True, text=True)
    assert process.returncode == 0
    assert json.loads(process.stdout)["states"] == 2


@pytest.mark.parametrize("argv", [["unknown"], ["tune"], ["init", "--backend", "missing"],
                                 ["validate", "--data", "/nonexistent/tarski.jsonl"]])
def test_usage_errors_are_json(argv, capsys):
    assert main(argv) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["error"]["code"] == "invalid_input"


def test_no_clobber_even_with_input_force(tmp_path, capsys):
    path = tmp_path / "config.json"
    assert main(["init", "--out", str(path)]) == 0
    original = path.read_bytes()
    assert main(["init", "--out", str(path)]) == 2
    assert path.read_bytes() == original
    assert main(["tune", "--config", str(path), "--reads", "unused", "--out", str(path), "--force"]) == 2
    assert path.read_bytes() == original
    assert main(["init", "--out", str(path), "--force"]) == 0


@pytest.mark.parametrize("text,match", [
    ("", "no records"), ('{"id":"a","id":"b"}', "duplicate JSON key"),
    ('{"id":"a","state":"x","questions":{}}', "nonempty object"),
    ('{"id":"a","state":{"count":1e400},"questions":{}}', "non-finite"),
    ('{"id":"a","state":"x","questions":{"q":{"type":"score","criteria":{}}}}', "array"),
    ('{"id":"a","state":"x","questions":{"q":{"type":"noul"}},"gold":{"q":"true"}}', "label must be"),
])
def test_record_errors_include_location(tmp_path, text, match):
    path = tmp_path / "bad.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_rows(path, "records")


def test_bad_probabilities_duplicate_ids_and_nan(tmp_path):
    path = tmp_path / "bad.jsonl"
    rows = load_rows(ROOT / "examples/calibration.reads.jsonl", "reads")[:1]
    write_output(path, rows * 2, lines=True)
    with pytest.raises(ValueError, match="duplicate id"):
        load_rows(path, "reads")
    rows[0]["questions"][0]["reads"]["stock|rot0|random"] = [.9, .9]
    write_output(path, rows, lines=True, force=True)
    with pytest.raises(ValueError, match="sum to 1"):
        load_rows(path, "reads")
    path.write_text(path.read_text().replace('0.9', 'NaN'))
    with pytest.raises(ValueError, match="non-finite"):
        load_rows(path, "reads")
