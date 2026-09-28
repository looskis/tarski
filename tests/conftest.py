"""Shared fixtures: a branch store trained on the example file, once per session, on CPU with a
small probe so the suite stays quick."""

import os

import pytest

from tarski.cli import main

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "tickets.csv")


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("cli") / "branches")
    main(["--store", d, "--device", "cpu", "train", EXAMPLE, "--kind", "probe", "--max-len", "32"])
    assert sorted(os.listdir(d)) == ["team", "urgent"]
    assert os.path.exists(os.path.join(d, "team", "oos.safetensors"))
    return d
