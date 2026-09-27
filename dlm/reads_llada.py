"""Second-model reader: LLaDA-8B-Instruct (masked diffusion, single bidirectional stack, no encoder),
through the same OpenJev canvas construction. Used to test whether the order effect and the
state-first fix are properties of diffusion readers in general or of DiffusionGemma's causal encoder.

Reads: the whole sequence (chat prompt + canvas) in one forward pass; the slots hold the model's own
mask token (`mode="random"` maps to this, the model's native undecided input, so the experiment
scripts' "random" column means "native slot" here; `mode="randtok"` writes an OpenJev-style random
token), `mode="mean"` puts the vocabulary-mean embedding in the slots via inputs_embeds. Label
distributions are read at the slot positions exactly as for DiffusionGemma (renormalised over the
label token ids). Needs transformers 4.x (the checkpoint's custom code): .venv_llada on the box.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional, Sequence

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("openjev", "jevbench"):
    p = os.path.join(ROOT, "third_party", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from openjev.config import Settings          # noqa: E402
from openjev.engine import Engine  # noqa: E402

DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"


class Reader:
    def __init__(self, model_path: str = DEFAULT_MODEL, dtype=torch.bfloat16, device: str = "cuda", **_):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mask_id = int(getattr(self.model.config, "mask_token_id", 126336))
        self.settings = Settings()
        self.engine = Engine(self.settings, self.tok)
        self.engine.scaffold = []                  # no Gemma thought block on the canvas
        self.embed_layer = next(m for m in self.model.modules() if isinstance(m, torch.nn.Embedding))
        self.hidden = self.embed_layer.weight.shape[1]
        self.vocab = self.embed_layer.weight.shape[0]
        self._mean_embed = None
        self._prefills = {}

    # -- request -> prompt, template, slots ---------------------------------------------------------

    def schema(self, questions: Dict[str, Dict]) -> Dict:
        return self.engine.build_schema(questions)

    def prepare(self, state_text: str, questions: Dict[str, Dict]) -> Dict:
        schema = self.schema(questions)
        qs, fmt = schema["questions"], schema["format"]
        sys_text = self.engine.system_text(qs, fmt)
        prompt = self.engine.chat_prompt_ids(sys_text, state_text)
        template, slots = self.engine.resolve_template(qs, fmt, head=[])
        return {"qs": qs, "fmt": fmt, "forced": schema["forced"], "prompt": prompt,
                "template": template, "slots": slots, "width": len(template)}

    def canvas(self, prep: Dict, seed: Optional[int] = None, randtok: bool = False) -> List[int]:
        ids = list(prep["template"])
        rng = random.Random(seed or 0)
        for s in prep["slots"]:
            ids[s["pos"]] = rng.randrange(self.vocab) if randtok else self.mask_id
        return ids

    def prefill(self, prompt):
        return None                                # single stack: no separate prefill

    # -- embeddings ---------------------------------------------------------------------------------

    @torch.inference_mode()
    def embed(self, ids):
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, device=self.device)
        return self.embed_layer(ids.to(self.device))

    @torch.inference_mode()
    def mean_embedding(self):
        if self._mean_embed is None:
            self._mean_embed = self.embed_layer.weight.float().mean(dim=0).to(self.embed_layer.weight.dtype)
        return self._mean_embed

    def with_slots(self, h, positions: Sequence[int], rows):
        h = h.clone()
        for i, p in enumerate(positions):
            h[0, p] = rows[i].to(h.dtype)
        return h

    # -- masks over the full sequence (prompt + canvas) ---------------------------------------------

    def masks(self, cache, width: int):
        return None

    def slot_masks(self, cache, width: int, positions: Sequence[int], mode: str = "isolate", prompt_len: int = 0):
        """Additive attention bias [1, 1, L, L] over the full sequence; positions are canvas-relative."""
        L = prompt_len + width
        pos = {prompt_len + p for p in positions}
        allow = torch.ones((L, L), dtype=torch.bool, device=self.device)
        col_is_slot = torch.tensor([j in pos for j in range(L)], device=self.device)
        row_is_slot = col_is_slot.clone()
        same = torch.eye(L, dtype=torch.bool, device=self.device)
        if mode == "default":
            block = torch.zeros((L, L), dtype=torch.bool, device=self.device)
        elif mode == "isolate":
            block = row_is_slot[:, None] & col_is_slot[None, :] & ~same
        elif mode == "invisible":
            block = (col_is_slot[None, :] & ~same).expand(L, L)
        else:
            raise ValueError(mode)
        allow = allow & ~block
        neg = torch.finfo(self.model.dtype).min
        return torch.zeros((1, 1, L, L), dtype=self.model.dtype, device=self.device).masked_fill(~allow[None, None], neg)

    # -- the read -----------------------------------------------------------------------------------

    @torch.inference_mode()
    def logits(self, ids=None, inputs_embeds=None, attention_bias=None):
        out = self.model(input_ids=ids, inputs_embeds=inputs_embeds, attention_bias=attention_bias)
        return out.logits.float()

    def label_logprobs(self, logits, label_ids: Sequence[Sequence[int]]):
        out = []
        for i, ids in enumerate(label_ids):
            lp = torch.log_softmax(logits[i], dim=-1)
            sub = lp[list(ids)]
            out.append(sub - torch.logsumexp(sub, dim=-1))
        return out

    @torch.inference_mode()
    def read(self, prep: Dict, mode: str = "random", seed: int = 0, queries=None, masks=None, with_entropy: bool = False):
        prompt = list(prep["prompt"])
        P = len(prompt)
        positions = [P + s["pos"] for s in prep["slots"]]
        if mode == "randtok":
            ids = torch.tensor([prompt + self.canvas(prep, seed, randtok=True)], device=self.device)
            lg = self.logits(ids=ids, attention_bias=masks)
        elif mode == "random":                       # native: mask tokens in the slots
            ids = torch.tensor([prompt + self.canvas(prep, None)], device=self.device)
            lg = self.logits(ids=ids, attention_bias=masks)
        elif mode in ("mean", "query"):
            ids = torch.tensor([prompt + self.canvas(prep, None)], device=self.device)
            h = self.embed(ids)
            rows = [self.mean_embedding()] * len(positions) if mode == "mean" else [torch.as_tensor(q, device=self.device) for q in queries]
            h = self.with_slots(h, positions, rows)
            lg = self.logits(inputs_embeds=h, attention_bias=masks)
        else:
            raise ValueError(mode)
        slot_logits = lg[0, positions]
        label_ids = [s["label_ids"] for s in prep["slots"]]
        lps = self.label_logprobs(slot_logits, label_ids)
        probs = [[float(v) for v in torch.exp(lp).tolist()] for lp in lps]
        if with_entropy:
            ents = []
            for i, ids_ in enumerate(label_ids):
                lp = torch.log_softmax(slot_logits[i], dim=-1)
                keep = sorted(set(torch.topk(lp, 20).indices.tolist()) | set(ids_))
                p = torch.exp(lp[keep])
                ents.append(float(-(p * torch.log(torch.clamp(p, min=1e-30))).sum()))
            return probs, ents
        return probs
