"""The user path end to end on the example file: train from a CSV, list, eval, predict, and the
out-of-scope detector, all on CPU with a small probe so the suite stays quick."""

import json
import os

from tarski.cli import main

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "tickets.csv")


def test_list_and_info(store, capsys):
    main(["--store", store, "list"])
    out = capsys.readouterr().out
    assert "team" in out and "urgent" in out and " yes" in out          # the oos column
    main(["--store", store, "--device", "cpu", "info"])
    out = capsys.readouterr().out
    assert "store:" in out and "team: probe@14" in out


def test_eval_scores_the_file(store, capsys):
    main(["--store", store, "--device", "cpu", "eval", EXAMPLE, "--max-len", "32", "--all-rows", "--json"])
    out = capsys.readouterr().out
    res = json.loads(out[out.index("{"):])
    assert res["team"]["n"] == 80 and res["team"]["acc"] > 0.8
    assert 0.0 <= res["team"]["oos_flag_rate"] <= 0.3


def test_predict_reports_label_confidence_and_scope(store, capsys):
    main(["--store", store, "--device", "cpu", "predict", "Production is down, nobody can log in",
          "--tasks", "team", "urgent"])
    row = json.loads(capsys.readouterr().out.strip().splitlines()[0])
    assert row["decisions"]["team"]["label"] == "outage"
    assert 0.0 <= row["decisions"]["team"]["confidence"] <= 1.0
    assert row["decisions"]["team"]["out_of_scope"] is False


def test_oos_scores_off_topic_messages_higher(store):
    from tarski.engine import Engine

    eng = Engine(store, device="cpu")
    in_scope = ["I was charged twice this month", "The app is down for everyone",
                "Please add dark mode", "Reset my password please"]
    off_topic = ["The recipe calls for two cups of flour and a pinch of salt",
                 "Photosynthesis converts light into chemical energy in leaves",
                 "The train to Lisbon leaves from platform nine at dawn"]
    r = eng.decide(in_scope + off_topic, ["team"])
    s = [o["team"]["score"] for o in r["oos"]]
    assert sum(s[len(in_scope):]) / len(off_topic) > sum(s[:len(in_scope)]) / len(in_scope)


def test_oos_scores_on_the_default_device(store):
    """Branches trained on CPU score on whatever device the engine picks (MPS on Apple silicon, which
    has no float64); the detector must not depend on the tap's device."""
    from tarski.engine import Engine

    r = Engine(store).decide(["Please add dark mode", "The recipe calls for two cups of flour"], ["team"])
    s = [o["team"]["score"] for o in r["oos"]]
    assert len(s) == 2 and all(isinstance(v, float) for v in s)


def test_server_exposes_scope(store):
    from fastapi.testclient import TestClient

    from tarski.engine import Engine
    from tarski.server import create_app

    client = TestClient(create_app(Engine(store, device="cpu")))
    t = client.get("/v1/tasks").json()["tasks"]
    assert t["team"]["oos"] is True
    r = client.post("/v1/decide", json={"text": "Production is down for everyone", "tasks": ["team"]}).json()
    d = r["results"][0]["team"]
    assert d["label"] == "outage" and "out_of_scope" in d and "oos_score" in d
