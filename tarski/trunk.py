"""A frozen base encoder used as a shared trunk.

Every message runs through the trunk once. Its hidden states at chosen depths ("taps") are what task
branches read, so N decisions about one message cost one trunk pass plus N small branches, instead of
N passes through the whole model.

The layer loop mirrors `ModernBertModel.forward` and builds its attention masks with the same
transformers helpers, so running layers [0, L) here reproduces the stock forward pass.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

DEFAULT_BASE = "answerdotai/ModernBERT-base"


def pick_device(pref: Optional[str] = None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Context:
    """Per-batch attention masks and rotary embeddings, shared by the trunk and every branch."""

    def __init__(self, masks: Dict[str, Optional[torch.Tensor]], rope: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                 attention_mask: torch.Tensor):
        self.masks = masks
        self.rope = rope
        self.attention_mask = attention_mask

    def run(self, layers: Iterable[nn.Module], h: torch.Tensor) -> torch.Tensor:
        for layer in layers:
            h = layer(h, attention_mask=self.masks[layer.attention_type],
                      position_embeddings=self.rope[layer.attention_type])
        return h


class Trunk(nn.Module):
    def __init__(self, base: str = DEFAULT_BASE, device: Optional[str] = None, max_len: int = 256):
        super().__init__()
        self.base = base
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModel.from_pretrained(base, attn_implementation="sdpa").to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.cfg = self.model.config
        self.n_layers = self.cfg.num_hidden_layers
        self.hidden = self.cfg.hidden_size
        self.max_len = max_len

    # -- identity -------------------------------------------------------------------------------

    def fingerprint(self) -> str:
        """Identifies the exact base weights a branch was trained against."""
        h = hashlib.sha256(self.base.encode())
        h.update(json.dumps(self.cfg.to_dict(), sort_keys=True, default=str).encode())
        for name in ("embeddings.tok_embeddings.weight", "layers.0.attn.Wqkv.weight", "final_norm.weight"):
            h.update(self.model.get_parameter(name).detach().float().cpu().numpy().tobytes()[:65536])
        return h.hexdigest()[:16]

    # -- inputs ---------------------------------------------------------------------------------

    def tokenize(self, texts: Sequence[str], max_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=max_len or self.max_len,
                       return_tensors="pt")
        return {"input_ids": enc["input_ids"].to(self.device), "attention_mask": enc["attention_mask"].to(self.device)}

    def token_ids(self, texts: Sequence[str], max_len: Optional[int] = None) -> List[List[int]]:
        return self.tok(list(texts), truncation=True, max_length=max_len or self.max_len)["input_ids"]

    def context(self, h: torch.Tensor, attention_mask: torch.Tensor) -> Context:
        kw = {"config": self.cfg, "inputs_embeds": h, "attention_mask": attention_mask}
        masks = {"full_attention": create_bidirectional_mask(**kw),
                 "sliding_attention": create_bidirectional_sliding_window_mask(**kw)}
        pos = torch.arange(h.shape[1], device=h.device)[None]
        rope = {t: self.model.rotary_emb(h, pos, t) for t in set(self.cfg.layer_types)}
        return Context(masks, rope, attention_mask)

    # -- forward --------------------------------------------------------------------------------

    @torch.no_grad()
    def taps(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
             depths: Iterable[int]) -> Tuple[Dict[int, torch.Tensor], Context]:
        """Hidden states after layer depth-1 for each requested depth (0 = embeddings), in one pass
        that stops at the deepest one. Depth n_layers returns the final-normed output, as the stock model."""
        want = sorted(set(int(d) for d in depths))
        if not want or want[0] < 0 or want[-1] > self.n_layers:
            raise ValueError(f"depths must be within [0, {self.n_layers}], got {want}")
        h = self.model.embeddings(input_ids=input_ids)
        ctx = self.context(h, attention_mask)
        out = {}
        if 0 in want:
            out[0] = h
        for i in range(want[-1]):
            h = ctx.run([self.model.layers[i]], h)
            if i + 1 in want:
                out[i + 1] = h
        if self.n_layers in out:
            out[self.n_layers] = self.model.final_norm(out[self.n_layers])
        return out, ctx

    def copy_layers(self, lo: int, hi: int) -> nn.ModuleList:
        """Trainable copies of base layers [lo, hi), to initialise a branch."""
        layers = nn.ModuleList(copy.deepcopy(self.model.layers[i]) for i in range(lo, hi))
        for p in layers.parameters():
            p.requires_grad_(True)
        return layers

    def copy_final_norm(self) -> nn.Module:
        norm = copy.deepcopy(self.model.final_norm)
        for p in norm.parameters():
            p.requires_grad_(True)
        return norm
