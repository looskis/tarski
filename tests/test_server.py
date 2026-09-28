"""End to end on a toy labelled file: train from CSV, serve, answer /v1/decide and Jev's /v1/systemone."""

import csv
import random

import pytest
from fastapi.testclient import TestClient

from tarski import data, train
from tarski.engine import Engine
from tarski.server import create_app
from tarski.trunk import Trunk

ROWS = {
    ("billing", "no"): ["I was charged twice this month", "Refund the duplicate payment please",
                        "My invoice shows the wrong amount", "Why did my card get billed again"],
    ("outage", "yes"): ["Everything is down and our demo is at noon", "The site is not loading for any customer",
                        "Production is broken, nobody can log in", "All requests are failing right now"],
    ("feature", "no"): ["Could you add dark mode someday", "It would be nice to export to CSV",
                        "Feature idea: keyboard shortcuts", "Please consider a bulk edit option"],
}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    d = tmp_path_factory.mktemp("t")
    path = d / "tickets.csv"
    rng = random.Random(0)
    rows = [(f"{t} #{i}" if i else t, team, urgent) for (team, urgent), texts in ROWS.items()
            for t in texts for i in range(5)]
    rng.shuffle(rows)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "team", "urgent"])
        w.writerows(rows)
    ds = data.load_table(str(path), max_len=32)
    assert set(ds.tasks) == {"team", "urgent"}
    trunk = Trunk(device="cpu", max_len=32)
    out = d / "branches"
    res = train.fit(trunk, ds, kind="probe", split=11, epochs=15, lr_head=3e-3, store=str(out), log=lambda s: None)
    assert res["team"]["acc"] > 0.8
    return str(out)


@pytest.fixture(scope="module")
def client(store):
    return TestClient(create_app(Engine(store, device="cpu", max_loaded=1)))


def test_tasks_listed(client):
    t = client.get("/v1/tasks").json()["tasks"]
    assert set(t) == {"team", "urgent"} and t["team"]["labels"] == ["billing", "feature", "outage"]


def test_decide_many_tasks_one_trunk_pass(client):
    r = client.post("/v1/decide", json={"texts": ["The whole app is down for everyone", "I was charged twice"],
                                        "tasks": ["team", "urgent"]}).json()
    down, billing = r["results"]
    assert down["team"]["label"] == "outage" and billing["team"]["label"] == "billing"
    assert down["urgent"]["probabilities"]["yes"] > billing["urgent"]["probabilities"]["yes"]
    assert set(r["timing_ms"]["branches"]) == {"team", "urgent"}


def test_lru_eviction_and_reload(client):
    client.post("/v1/decide", json={"text": "charged twice", "tasks": ["team"]})
    client.post("/v1/decide", json={"text": "charged twice", "tasks": ["urgent"]})
    t = client.get("/v1/tasks").json()
    assert t["tasks"]["urgent"]["loaded"] and not t["tasks"]["team"]["loaded"]   # max_loaded=1
    assert t["stats"]["evictions"] >= 1


def test_systemone_jev_shapes(client):
    body = {"state": "I was charged twice, refund the duplicate payment", "model": "jev-latest", "questions": {
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "charges", "outage": "down", "feature": "requests"}},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    }}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    a = r.json()["answers"]
    assert a["team"]["choice"] == "billing" and 0 <= a["team"]["confidence"] <= 1
    assert abs(sum(a["team"]["probabilities"].values()) - 1) < 1e-6
    assert a["urgent"]["type"] == "noul" and a["urgent"]["noul"] < 0.5
    assert r.json()["usage"]["input_tokens"] > 0


def test_systemone_option_subset_renormalises(client):
    body = {"state": "Please add an export button", "questions": {
        "team": {"type": "choice", "criteria": {"billing": "", "feature": ""}}}}
    a = client.post("/v1/systemone", json=body).json()["answers"]["team"]
    assert set(a["probabilities"]) == {"billing", "feature"} and a["choice"] == "feature"


def test_systemone_unknown_question_without_fallback(client):
    body = {"state": "hello", "questions": {"sentiment": {"type": "noul"}}}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 400 and "sentiment" in r.text


def test_branch_refuses_other_base(store, monkeypatch):
    from tarski.branches import Branch

    trunk = Trunk(device="cpu")
    monkeypatch.setattr(trunk, "fingerprint", lambda: "not-the-same")
    with pytest.raises(ValueError, match="retrain"):
        Branch.load(f"{store}/team", trunk)
