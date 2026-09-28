"""The labelling page: queue order from the branches' uncertainty, saving in the shape `tarski train`
reads, resuming, and new labels typed by the labeller."""

import json

from fastapi.testclient import TestClient

from tarski import data
from tarski.engine import Engine
from tarski.label import LabelQueue, create_label_app, parse_task_specs, read_messages

MESSAGES = ["Production is down, nobody can log in",                 # clearly in scope
            "The recipe calls for two cups of flour and a pinch of salt",
            "asdf qwer zxcv 12345",                                   # gibberish: flagged, so first
            "I was charged twice for the same invoice"]


def test_specs_and_readers(tmp_path):
    assert parse_task_specs(["team=billing,outage", "urgent"]) == {"team": ["billing", "outage"], "urgent": []}
    p = tmp_path / "m.txt"
    p.write_text("one\n\ntwo\n")
    assert read_messages(str(p)) == ["one", "", "two"]
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"text": "three"}) + "\n")
    assert read_messages(str(p)) == ["three"]


def test_queue_without_a_store_uses_the_specs(tmp_path):
    q = LabelQueue(MESSAGES, str(tmp_path / "labels.jsonl"), parse_task_specs(["team=billing,outage"]))
    assert q.stats()["remaining"] == 4 and q.next(1)[0]["suggest"] == {}
    row = q.save(q.next(1)[0]["id"], {"team": "outage"})
    assert row == {"text": MESSAGES[0], "team": "outage"}


def test_queue_orders_by_uncertainty_and_saves_trainable_rows(store, tmp_path):
    out = tmp_path / "labels.jsonl"
    q = LabelQueue(MESSAGES, str(out), {}, engine=Engine(store, device="cpu"))
    assert set(q.tasks) == {"team", "urgent"} and q.tasks["team"] == ["account", "billing", "feature", "outage"]
    first = q.next(4)
    assert first[0]["text"] == "asdf qwer zxcv 12345"                        # out-of-scope flag comes first
    rest = [min(v["confidence"] for v in it["suggest"].values()) for it in first[1:]]
    assert rest == sorted(rest)                                              # then least confident first
    ids = {it["text"]: it["id"] for it in first}
    q.save(ids[MESSAGES[0]], {"team": "outage", "urgent": "yes"})
    q.save(ids[MESSAGES[3]], {"team": "billing", "urgent": "no"})
    q.save(ids[MESSAGES[1]], {"team": "other"})                              # a new label
    assert "other" in q.tasks["team"]
    q.skip(ids[MESSAGES[2]])
    s = q.stats()
    assert (s["done"], s["skipped"], s["remaining"]) == (3, 1, 0)
    ds = data.load_table(str(out))                                           # the output trains as is
    assert set(ds.tasks) == {"team", "urgent"} and len(ds.train) + len(ds.val) + len(ds.test) == 3
    # resuming skips what was labelled
    q2 = LabelQueue(MESSAGES, str(out), {"team": []})
    assert q2.stats()["remaining"] == 1 and q2.next(1)[0]["text"] == MESSAGES[2]


def test_label_api(tmp_path):
    q = LabelQueue(MESSAGES[:2], str(tmp_path / "labels.jsonl"), parse_task_specs(["urgent=yes,no"]))
    c = TestClient(create_label_app(q))
    assert "tarski label" in c.get("/").text
    assert c.get("/api/tasks").json()["tasks"] == {"urgent": ["yes", "no"]}
    it = c.get("/api/next?n=1").json()["items"][0]
    r = c.post("/api/save", json={"id": it["id"], "decisions": {"urgent": "yes"}})
    assert r.status_code == 200, r.text
    assert r.json()["stats"]["done"] == 1
    assert c.post("/api/save", json={"id": "nope", "decisions": {"urgent": "yes"}}).status_code == 404
    assert c.post("/api/save", json={"id": it["id"], "decisions": {"team": "x"}}).status_code == 400
    it2 = c.get("/api/next?n=1").json()["items"][0]
    assert c.post("/api/skip", json={"id": it2["id"]}).json()["stats"]["remaining"] == 0
