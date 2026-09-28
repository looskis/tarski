"""H6: adding or retraining a branch cannot change another branch's answers (the trunk is frozen)."""

import csv
import random

import torch

from tarski import data, train
from tarski.engine import Engine
from tarski.trunk import Trunk

ROWS = [("I was charged twice", "billing", "no"), ("Refund the duplicate payment", "billing", "no"),
        ("Everything is down before the demo", "outage", "yes"), ("Nobody can log in", "outage", "yes"),
        ("Please add dark mode", "feature", "no"), ("Export to CSV would help", "feature", "no")]
PROBE = ["The dashboard is down for all users", "My card was billed again", "Add keyboard shortcuts"]


def _probs(store):
    eng = Engine(store, device="cpu")
    return eng.decide(PROBE, ["team"])["results"]


def test_adding_and_retraining_other_branches_leaves_a_branch_unchanged(tmp_path):
    path = tmp_path / "t.csv"
    rows = [(f"{t} ({i})", team, u) for t, team, u in ROWS for i in range(8)]
    random.Random(0).shuffle(rows)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([("text", "team", "urgent"), *rows])
    ds = data.load_table(str(path), max_len=32)
    trunk = Trunk(device="cpu", max_len=32)
    store = str(tmp_path / "branches")

    train.fit(trunk, ds, ["team"], kind="probe", split=6, epochs=3, store=store, log=lambda s: None)
    before = _probs(store)
    train.fit(trunk, ds, ["urgent"], kind="blocks", split=11, depth=1, epochs=1, store=store, log=lambda s: None)
    train.fit(trunk, ds, ["urgent"], kind="probe", split=3, epochs=2, seed=7, store=store, log=lambda s: None)
    after = _probs(store)
    for a, b in zip(before, after):
        assert torch.equal(torch.as_tensor(a["team"]), torch.as_tensor(b["team"]))
    assert trunk.fingerprint() == Trunk(device="cpu").fingerprint()      # the base itself never changes


def test_fit_temperature_never_returns_nan():
    from tarski.train import fit_temperature

    logits = torch.tensor([[1e6, -1e6], [-1e6, 1e6], [1e6, -1e6]])       # separable to the point of overflow
    t = fit_temperature(logits, torch.tensor([0, 1, 1]))
    assert t == t and 0.05 <= t <= 20.0
