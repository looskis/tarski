"""Fused multi-branch execution matches running the branches one by one."""

import torch

from tarski.branches import BlockBranch
from tarski.fused import FusedBlocks
from tarski.trunk import Trunk


def test_fused_matches_loop():
    torch.manual_seed(0)
    trunk = Trunk(device="cpu")
    b = trunk.tokenize(["the deploy is failing on staging", "refund please", "ok"])
    for split, depth in ((11, 2), (0, 1), (20, 2)):
        brs = [BlockBranch(split, [f"l{i}" for i in range(5)], trunk.hidden, depth, trunk).eval() for _ in range(4)]
        for br in brs:                                  # make branches differ from each other and from the base
            with torch.no_grad():
                for p in br.parameters():
                    p.add_(torch.randn_like(p) * 0.02)
            br.temperature.fill_(float(torch.rand(()) + 0.5))
        taps, ctx = trunk.taps(b["input_ids"], b["attention_mask"], [split])
        with torch.no_grad():
            loop = torch.stack([br.probs(taps[split], ctx) for br in brs])
            fused = FusedBlocks(brs).probs(taps[split], ctx)
        assert (loop - fused).abs().max() < 1e-4, (split, depth, float((loop - fused).abs().max()))
