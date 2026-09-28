"""Parity: the split trunk + branch layers reproduce ModernBERT's own forward pass."""

import os
import tempfile

import pytest
import torch

from tarski.branches import BlockBranch, Branch, ProbeBranch, mean_pool
from tarski.trunk import Trunk

TEXTS = ["I was charged twice for my card this month, please refund it.",
         "hi",
         "The deploy to staging is failing on the migration step again and blocking the release."]


@pytest.fixture(scope="module")
def trunk():
    return Trunk(device="cpu")


def test_full_depth_matches_stock_forward(trunk):
    b = trunk.tokenize(TEXTS)
    ref = trunk.model(**b).last_hidden_state
    taps, _ = trunk.taps(b["input_ids"], b["attention_mask"], [trunk.n_layers])
    valid = b["attention_mask"].bool()
    assert (ref - taps[trunk.n_layers])[valid].abs().max() < 1e-4


def test_branch_continuing_from_split_matches_stock(trunk):
    """A block branch holding copies of layers [k, L) plus the final norm is the stock model."""
    b = trunk.tokenize(TEXTS)
    ref = mean_pool(trunk.model(**b).last_hidden_state, b["attention_mask"])
    k = 11
    taps, ctx = trunk.taps(b["input_ids"], b["attention_mask"], [k])
    br = BlockBranch(k, ["a", "b"], trunk.hidden, trunk.n_layers - k, trunk).eval()
    with torch.no_grad():
        h = br.norm(ctx.run(br.layers, taps[k]))
    assert (mean_pool(h, b["attention_mask"]) - ref).abs().max() < 1e-4


def test_multiple_taps_one_pass(trunk):
    b = trunk.tokenize(TEXTS)
    taps, _ = trunk.taps(b["input_ids"], b["attention_mask"], [4, 11, 16])
    solo, _ = trunk.taps(b["input_ids"], b["attention_mask"], [11])
    assert set(taps) == {4, 11, 16}
    assert torch.equal(taps[11], solo[11])


def test_save_load_roundtrip(trunk):
    b = trunk.tokenize(TEXTS)
    taps, ctx = trunk.taps(b["input_ids"], b["attention_mask"], [8])
    for br in (ProbeBranch(8, ["x", "y", "z"], trunk.hidden), BlockBranch(8, ["x", "y", "z"], trunk.hidden, 2, trunk)):
        br.eval()
        br.temperature.fill_(1.7)
        with tempfile.TemporaryDirectory() as d:
            br.save(os.path.join(d, "t"), trunk.fingerprint(), trunk.base)
            again = Branch.load(os.path.join(d, "t"), trunk)
        assert float(again.temperature) == pytest.approx(1.7)
        with torch.no_grad():
            p1, p2 = br.probs(taps[8], ctx), again.probs(taps[8], ctx)
        assert (p1 - p2).abs().max() < 5e-3        # weights are stored in fp16
