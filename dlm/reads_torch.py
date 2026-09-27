"""CUDA port of dlm/reads.py: the same reads of DiffusionGemma (google/diffusiongemma-26B-A4B-it,
bf16, transformers>=5.17) through OpenJev's canvas construction, with the same hooks: embedding-path
slots (random token / vocabulary mean / query), per-layer canvas attention masks, an allowed-expert
mask on every router (simulated carve), a routing recorder (census) and an attention recorder
(leakage). Method names and return shapes match dlm/reads.py so the experiment scripts can pick a
backend with `--backend torch`.

Differences from the MLX reader: the model is bf16, not the 4-bit MLX conversion, so label
distributions are close but not identical (dlm/parity_torch.py measures the gap); no gradients here.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("openjev", "jevbench"):
    p = os.path.join(ROOT, "third_party", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from openjev.config import Settings          # noqa: E402
from openjev.engine import PAD, TURN_CLOSE, VOCAB, Engine  # noqa: E402

DEFAULT_MODEL = "google/diffusiongemma-26B-A4B-it"


def _patch_router():
    """Router.forward with an optional allowed-expert mask (`router._dlm_allowed`, bool [E]) and an
    optional recorder of (indices, weights) per call, in layer order."""
    from transformers.models.diffusion_gemma import modeling_diffusion_gemma as M

    R = M.DiffusionGemmaTextRouter
    if getattr(R, "_dlm_patched", False):
        return

    def forward(self, hidden_states):
        h = self.norm(hidden_states)
        h = h * self.scale * self.scalar_root_size
        scores = self.proj(h)
        allowed = getattr(self, "_dlm_allowed", None)
        if allowed is not None:
            scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        w, idx = torch.topk(probs, k=self.config.top_k_experts, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        w = w * self.per_expert_scale[idx]
        rec = R._dlm_record
        if rec is not None:
            rec.append((idx, w))
        return probs, w, idx

    R.forward = forward
    R._dlm_patched = True
    R._dlm_record = None


class RoutingRecorder:
    """Collect (indices [N, k], weights [N, k]) from every router call inside the block."""

    def __enter__(self):
        from transformers.models.diffusion_gemma import modeling_diffusion_gemma as M
        self.calls: List = []
        M.DiffusionGemmaTextRouter._dlm_record = self.calls
        return self

    def __exit__(self, *exc):
        from transformers.models.diffusion_gemma import modeling_diffusion_gemma as M
        M.DiffusionGemmaTextRouter._dlm_record = None


class AttentionRecorder:
    """Collect the decoder layers' attention weights [B, heads, Q, K] (needs attn_implementation="eager")."""

    def __init__(self, reader, layers: Optional[Sequence[int]] = None):
        self.r = reader
        self.layers = list(layers) if layers is not None else list(range(len(reader.dec.layers)))
        self.weights: Dict[int, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        self.weights = {}
        for i in self.layers:
            def hook(module, args, output, i=i):
                w = output[1]
                if w is not None:
                    self.weights[i] = w.detach().float()
            self._handles.append(self.r.dec.layers[i].self_attn.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


class Reader:
    def __init__(self, model_path: str = DEFAULT_MODEL, dtype=torch.bfloat16, device: str = "cuda",
                 attn_implementation: str = "eager", prefill_cache_tokens: int = 8192):
        from transformers import AutoTokenizer
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

        _patch_router()
        self.torch = torch
        self.device = device
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            model_path, dtype=dtype, device_map=device, attn_implementation=attn_implementation).eval()
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.settings = Settings()
        self.engine = Engine(self.settings, self.tok)
        self.dec = self.model.model.decoder
        self.enc = self.model.model.encoder
        self.text_config = getattr(self.dec, "text_config", None) or self.model.config.get_text_config(decoder=True)
        self.hidden = self.text_config.hidden_size
        self.attn_implementation = attn_implementation
        self._prefills: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._prefill_budget = prefill_cache_tokens
        self._mean_embed = None
        self._override = None
        self.dec.embed_tokens.register_forward_hook(self._embed_hook)

    def _embed_hook(self, module, args, output):
        return self._override if self._override is not None else output

    # -- request -> prompt, template, slots ---------------------------------------------------------

    def schema(self, questions: Dict[str, Dict]) -> Dict:
        return self.engine.build_schema(questions)

    def prepare(self, state_text: str, questions: Dict[str, Dict]) -> Dict:
        schema = self.schema(questions)
        qs, fmt = schema["questions"], schema["format"]
        groups = self.engine.groups(qs, fmt)
        if len(groups) != 1:
            raise ValueError(f"{len(qs)} questions need {len(groups)} canvases; keep a request to one")
        sys_text = self.engine.system_text(qs, fmt)
        prompt = self.engine.chat_prompt_ids(sys_text, state_text)
        template, slots = self.engine.resolve_template(qs, fmt)
        return {"qs": qs, "fmt": fmt, "forced": schema["forced"], "prompt": prompt,
                "template": template, "slots": slots, "width": self.engine.canvas_width(template)}

    def canvas(self, prep: Dict, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            return self.engine.build_canvas(prep["template"], prep["slots"], seed)
        c = list(prep["template"]) + [TURN_CLOSE]
        return c + [PAD] * (prep["width"] - len(c))

    # -- prefill cache ------------------------------------------------------------------------------

    @torch.inference_mode()
    def prefill(self, prompt: Sequence[int]):
        from transformers import DynamicCache
        key = tuple(prompt)
        hit = self._prefills.get(key)
        if hit is not None:
            self._prefills.move_to_end(key)
            return hit[0]
        cache = DynamicCache(config=self.model.config.get_text_config(decoder=True))
        ids = torch.tensor([list(prompt)], device=self.device)
        out = self.enc(input_ids=ids, past_key_values=cache)
        cache = out.past_key_values
        self._prefills[key] = (cache, len(prompt))
        while len(self._prefills) > 1 and sum(n for _, n in self._prefills.values()) > self._prefill_budget:
            self._prefills.popitem(last=False)
        return cache

    # -- embeddings ---------------------------------------------------------------------------------

    @torch.inference_mode()
    def embed(self, canvas_ids):
        """Input embeddings as the decoder computes them (already scaled), before self-conditioning."""
        if not torch.is_tensor(canvas_ids):
            canvas_ids = torch.tensor(canvas_ids, device=self.device)
        return self.dec.embed_tokens(canvas_ids.to(self.device))

    @torch.inference_mode()
    def mean_embedding(self, chunk: int = 32768):
        if self._mean_embed is None:
            total = torch.zeros((self.hidden,), dtype=torch.float32, device=self.device)
            for s in range(0, VOCAB, chunk):
                ids = torch.arange(s, min(s + chunk, VOCAB), device=self.device)
                total += self.embed(ids).float().sum(dim=0)
            self._mean_embed = (total / VOCAB).to(self.model.dtype)
        return self._mean_embed

    def with_slots(self, h, positions: Sequence[int], rows):
        h = h.clone()
        for i, p in enumerate(positions):
            h[0, p] = rows[i].to(h.dtype)
        return h

    # -- masks --------------------------------------------------------------------------------------

    def _enc_len(self, cache, layer_type: str) -> int:
        i = self.text_config.layer_types.index(layer_type)
        return cache.layers[i].keys.shape[-2]

    def masks(self, cache, width: int):
        dummy = torch.zeros((1, width, self.hidden), dtype=self.model.dtype, device=self.device)
        return self.dec.create_diffusion_decoder_attention_mask(
            config=self.text_config, inputs_embeds=dummy, past_key_values=cache, decoder_attention_mask=None)

    def _allowed(self, m):
        return m if m.dtype == torch.bool else (m == 0)

    def _as_mask(self, allow, like):
        if like is not None and like.dtype != torch.bool:
            neg = torch.finfo(like.dtype).min
            return torch.zeros(allow.shape, dtype=like.dtype, device=allow.device).masked_fill(~allow, neg)
        if like is None and self.attn_implementation == "eager":
            neg = torch.finfo(self.model.dtype).min
            return torch.zeros(allow.shape, dtype=self.model.dtype, device=allow.device).masked_fill(~allow, neg)
        return allow

    def slot_masks(self, cache, width: int, positions: Sequence[int], mode: str = "isolate"):
        """Same semantics as dlm/reads.py: "isolate" (slot rows do not see other slots), "invisible"
        (no canvas row sees any slot but itself), "default"."""
        base = self.masks(cache, width)
        if mode == "default":
            return base
        out = {}
        pos = set(positions)
        dev = self.device
        for layer_type, m in base.items():
            if m is None:
                enc_len = self._enc_len(cache, layer_type)
                allow = torch.ones((1, 1, width, enc_len + width), dtype=torch.bool, device=dev)
            else:
                allow = self._allowed(m)
                if allow.shape[-2] != width:
                    allow = allow.expand(-1, -1, width, -1)
            key_len = allow.shape[-1]
            enc_len = key_len - width
            col_is_slot = torch.tensor([False] * enc_len + [j in pos for j in range(width)], device=dev)
            row_is_slot = torch.tensor([j in pos for j in range(width)], device=dev)
            same = torch.cat([torch.zeros((width, enc_len), dtype=torch.bool, device=dev),
                              torch.eye(width, dtype=torch.bool, device=dev)], dim=1)
            if mode == "isolate":
                block = row_is_slot[:, None] & col_is_slot[None, :] & ~same
            elif mode == "invisible":
                block = (col_is_slot[None, :] & ~same).expand(width, key_len)
            else:
                raise ValueError(mode)
            out[layer_type] = self._as_mask(allow & ~block[None, None], m)
        return out

    # -- the read -----------------------------------------------------------------------------------

    @torch.inference_mode()
    def slot_logits(self, cache, inputs_embeds, positions: Sequence[int], masks):
        """Softcapped vocabulary logits at the slot rows, from input embeddings [1, L, D]: float32 [n, V]."""
        L = inputs_embeds.shape[1]
        dummy_ids = torch.zeros((1, L), dtype=torch.long, device=self.device)
        self._override = inputs_embeds
        try:
            out = self.dec(decoder_input_ids=dummy_ids, past_key_values=cache, decoder_attention_mask=masks)
        finally:
            self._override = None
        rows = out.last_hidden_state[0, list(positions)]
        logits = self.model.lm_head(rows).float()
        cap = self.model.final_logit_softcapping
        return torch.tanh(logits / cap) * cap

    @torch.inference_mode()
    def decoder_logits(self, cache, inputs_embeds, masks, self_conditioning_logits=None):
        """Softcapped logits for every canvas row, float32 [1, L, V], with an optional self-conditioning
        signal (the previous step's logits, as the generation loop passes them)."""
        L = inputs_embeds.shape[1]
        dummy_ids = torch.zeros((1, L), dtype=torch.long, device=self.device)
        self._override = inputs_embeds
        try:
            out = self.dec(decoder_input_ids=dummy_ids, past_key_values=cache, decoder_attention_mask=masks,
                           self_conditioning_logits=self_conditioning_logits)
        finally:
            self._override = None
        logits = self.model.lm_head(out.last_hidden_state).float()
        cap = self.model.final_logit_softcapping
        return torch.tanh(logits / cap) * cap

    def label_logprobs(self, logits, label_ids: Sequence[Sequence[int]]):
        out = []
        for i, ids in enumerate(label_ids):
            lp = torch.log_softmax(logits[i], dim=-1)
            sub = lp[list(ids)]
            out.append(sub - torch.logsumexp(sub, dim=-1))
        return out

    def topk_entropy(self, logits, label_ids: Sequence[Sequence[int]], k: int = 20) -> List[float]:
        out = []
        for i, ids in enumerate(label_ids):
            lp = torch.log_softmax(logits[i], dim=-1)
            keep = sorted(set(torch.topk(lp, k).indices.tolist()) | set(ids))
            p = torch.exp(lp[keep])
            out.append(float(-(p * torch.log(torch.clamp(p, min=1e-30))).sum()))
        return out

    @torch.inference_mode()
    def read(self, prep: Dict, mode: str = "random", seed: int = 0, queries=None, masks=None,
             with_entropy: bool = False):
        cache = self.prefill(prep["prompt"])
        positions = [s["pos"] for s in prep["slots"]]
        ids = torch.tensor([self.canvas(prep, seed if mode == "random" else None)], device=self.device)
        h = self.embed(ids)
        if mode == "mean":
            m = self.mean_embedding()
            h = self.with_slots(h, positions, [m] * len(positions))
        elif mode == "query":
            h = self.with_slots(h, positions, [torch.as_tensor(q, device=self.device) for q in queries])
        elif mode != "random":
            raise ValueError(mode)
        if masks is None:
            masks = self.masks(cache, prep["width"])
        logits = self.slot_logits(cache, h, positions, masks)
        label_ids = [s["label_ids"] for s in prep["slots"]]
        lps = self.label_logprobs(logits, label_ids)
        probs = [[float(v) for v in torch.exp(lp).tolist()] for lp in lps]
        if with_entropy:
            return probs, self.topk_entropy(logits, label_ids)
        return probs

    # -- carve --------------------------------------------------------------------------------------

    def routers(self):
        from transformers.models.diffusion_gemma import modeling_diffusion_gemma as M
        seen, out = set(), []
        for _, m in self.model.named_modules():
            if isinstance(m, M.DiffusionGemmaTextRouter) and id(m) not in seen:
                seen.add(id(m))
                out.append(m)
        return out

    def set_carve(self, sets):
        """`sets`: per decoder layer, the allowed expert set (applied to the router of that layer, which
        the encoder shares), or None to restore the full model. Clears the prefill cache."""
        n_experts = self.text_config.num_experts
        layer_routers = []
        from transformers.models.diffusion_gemma import modeling_diffusion_gemma as M
        for layer in self.dec.layers:
            rs = [m for _, m in layer.named_modules() if isinstance(m, M.DiffusionGemmaTextRouter)]
            assert len(rs) == 1
            layer_routers.append(rs[0])
        if sets is None:
            for r in self.routers():
                r._dlm_allowed = None
        else:
            for r, s in zip(layer_routers, sets):
                r._dlm_allowed = torch.tensor([i in s for i in range(n_experts)], device=self.device)
            # encoder routers, if they are separate objects, get the same mask by layer index
            enc_layers = getattr(self.enc, "layers", None) or getattr(getattr(self.enc, "language_model", None), "layers", None)
            if enc_layers is not None:
                for layer, s in zip(enc_layers, sets):
                    for _, m in layer.named_modules():
                        if isinstance(m, M.DiffusionGemmaTextRouter):
                            m._dlm_allowed = torch.tensor([i in s for i in range(n_experts)], device=self.device)
        self._prefills.clear()
