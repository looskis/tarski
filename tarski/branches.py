"""Task branches: the small, per-decision parts that sit on the shared trunk.

A branch reads the trunk's hidden states at its split depth and returns logits over its labels.

  probe   LayerNorm + mean pool + linear. KB-sized; costs almost nothing per extra decision.
  blocks  `depth` transformer layers copied from the base at [split, split + depth), fine-tuned for this
          task, then a final norm, mean pool and linear. The branch is "the rest of the network, cut
          short and specialised": each extra decision costs `depth` layers instead of a full pass.

Branches are saved as `<store>/<task>/branch.safetensors` + `meta.json`, and refuse to load against a
base whose fingerprint differs from the one they were trained on.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from tarski.trunk import Context, Trunk


def mean_pool(h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    m = attention_mask.to(h.dtype)[..., None]
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


class Branch(nn.Module):
    kind = "base"

    def __init__(self, split: int, labels: List[str], hidden: int):
        super().__init__()
        self.split = int(split)
        self.labels = list(labels)
        self.hidden = hidden
        self.register_buffer("temperature", torch.ones(()))
        self.meta: Dict = {}
        self.oos = None      # tarski.oos.OOSStats when the branch was trained with out-of-scope detection

    @property
    def n_labels(self) -> int:
        return len(self.labels)

    def logits(self, h: torch.Tensor, ctx: Context) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, h: torch.Tensor, ctx: Context) -> torch.Tensor:
        return self.logits(h, ctx)

    def probs(self, h: torch.Tensor, ctx: Context) -> torch.Tensor:
        return torch.softmax(self.logits(h, ctx).float() / self.temperature.float(), -1)

    # -- persistence ------------------------------------------------------------------------------

    def config(self) -> Dict:
        return {"kind": self.kind, "split": self.split, "labels": self.labels, "hidden": self.hidden}

    def save(self, path: str, fingerprint: str, base: str, extra: Optional[Dict] = None) -> None:
        os.makedirs(path, exist_ok=True)
        sd = {k: v.detach().to("cpu", torch.float16 if v.is_floating_point() else v.dtype).contiguous()
              for k, v in self.state_dict().items()}
        sd["temperature"] = self.temperature.detach().float().cpu()      # keep calibration in full precision
        save_file(sd, os.path.join(path, "branch.safetensors"))
        meta = {**self.config(), "base": base, "fingerprint": fingerprint, "saved_at": time.time(),
                "params": sum(p.numel() for p in self.parameters()), **(extra or {})}
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        self.meta = meta

    @staticmethod
    def load(path: str, trunk: Trunk) -> "Branch":
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        if meta["fingerprint"] != trunk.fingerprint():
            raise ValueError(f"branch {path!r} was trained on base {meta['base']} ({meta['fingerprint']}), "
                             f"not the loaded base {trunk.base} ({trunk.fingerprint()}); retrain it")
        branch = build_branch(meta, trunk)
        sd = load_file(os.path.join(path, "branch.safetensors"))
        branch.load_state_dict({k: v.float() for k, v in sd.items()})
        branch.meta = meta
        from tarski.oos import OOSStats

        branch.oos = OOSStats.load(path)
        return branch.to(trunk.device).eval()


class ProbeBranch(Branch):
    kind = "probe"

    def __init__(self, split: int, labels: List[str], hidden: int, dropout: float = 0.1):
        super().__init__(split, labels, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, len(labels))

    def logits(self, h, ctx):
        return self.out(self.drop(mean_pool(self.norm(h), ctx.attention_mask)))


class BlockBranch(Branch):
    kind = "blocks"

    INITS = ("next", "top", "random")

    def __init__(self, split: int, labels: List[str], hidden: int, depth: int, trunk: Trunk, dropout: float = 0.1,
                 init: str = "next"):
        """`init`: "next" copies base layers [split, split+depth) (the layers the trunk would have run
        next); "top" copies the base's last `depth` layers; "random" re-initialises the copied layers."""
        super().__init__(split, labels, hidden)
        if split + depth > trunk.n_layers:
            raise ValueError(f"split {split} + depth {depth} exceeds the base's {trunk.n_layers} layers")
        if init not in self.INITS:
            raise ValueError(f"init must be one of {self.INITS}")
        self.depth, self.init = int(depth), init
        lo = trunk.n_layers - depth if init == "top" else split
        self.layers = trunk.copy_layers(lo, lo + depth)
        if init == "random":
            self.layers.apply(trunk.model._init_weights)
        self.norm = trunk.copy_final_norm()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, len(labels))

    def config(self):
        return {**super().config(), "depth": self.depth, "init": self.init}

    def logits(self, h, ctx):
        h = self.norm(ctx.run(self.layers, h))
        return self.out(self.drop(mean_pool(h, ctx.attention_mask)))


def build_branch(cfg: Dict, trunk: Trunk) -> Branch:
    kind = cfg["kind"]
    if kind == "probe":
        return ProbeBranch(cfg["split"], cfg["labels"], trunk.hidden)
    if kind == "blocks":
        return BlockBranch(cfg["split"], cfg["labels"], trunk.hidden, cfg["depth"], trunk, init=cfg.get("init", "next"))
    raise ValueError(f"unknown branch kind {kind!r}")
