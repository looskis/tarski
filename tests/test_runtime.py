import sys
from types import SimpleNamespace

import pytest

from tarski.config import default_config
from tarski.data import labels_for
from tarski.runtime import collect, predictions


class Engine:
    def system_text(self, qs, fmt):
        return "Generic instruction\n" + "\n".join(f"Question {q['key']}: decide" for q in qs) + "\nReply in order."

    def chat_prompt_ids(self, system, user):
        return [system, user]

    def resolve_template(self, qs, fmt):
        return [], []

    def canvas_width(self, template):
        return 64

    def groups(self, qs, fmt):
        return [qs]


class Reader:
    def __init__(self):
        self.engine = Engine()
        self.calls = []

    def schema(self, questions):
        return {"questions": [{"key": qid, "type": q["type"], "choices": [(label, "") for label in labels_for(q)]}
                              for qid, q in questions.items()], "format": "lines", "forced": {}}

    def read(self, prep, mode, seed):
        self.calls.append((prep, mode, seed))
        return [[.8, .2] if q["key"] == "urgent" else [.1, .9] for q in prep["qs"]]


@pytest.fixture
def record():
    return {"id": "one", "state": "service is down", "questions": {
        "team": {"type": "choice", "criteria": {"billing": "", "support": ""}},
        "urgent": {"type": "noul"}}, "gold": {"team": {"label": "support"}, "urgent": "yes"}}


def test_prompt_order_slot_and_prediction_mapping(monkeypatch, record):
    monkeypatch.setitem(sys.modules, "openjev.engine", SimpleNamespace(FORMATS={"lines": ("", "", "Reply in order.")}))
    config = default_config()
    config["read"]["question_order"] = "rotate:3"
    config["read"]["seed"] = 17
    config["calibration"]["temperatures"] = {"noul": 2.0}
    reader = Reader()
    rows = collect([record], config, [config["read"]], reader=reader)
    prep, mode, seed = reader.calls[0]
    assert [q["key"] for q in prep["qs"]] == ["urgent", "team"]
    assert prep["prompt"][1].startswith("service is down\n\nThe questions:")
    assert "Question urgent" not in prep["prompt"][0]
    assert (mode, seed) == ("mean", 17)
    assert [q["qid"] for q in rows[0]["questions"]] == ["team", "urgent"]
    result = predictions(rows, config)[0]["decisions"]
    assert result["team"]["label"] == "support"
    assert result["urgent"]["label"] == "yes"
    assert result["urgent"]["confidence"] == pytest.approx(2 / 3)
    assert rows[0]["reader"]["seed"] == 17


def test_llada_native_and_stock_prompt(record):
    config, reader = default_config("llada"), Reader()
    collect([record], config, [config["read"]], reader=reader)
    prep, mode, _ = reader.calls[0]
    assert mode == "random"  # LLaDA's reader maps this to its native mask token.
    assert prep["prompt"][1] == record["state"]
    assert "Question team" in prep["prompt"][0]


def test_runtime_rejects_wrong_reader_shape(record):
    config, reader = default_config("llada"), Reader()
    reader.read = lambda *a, **kw: [[.8, .2]]
    with pytest.raises(ValueError, match="number of answers"):
        collect([record], config, [config["read"]], reader=reader)
