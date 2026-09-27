"""Reads of DiffusionGemma on Apple silicon, with control over what sits in the answer slots.

OpenJev reads a decision by seeding each answer slot of a 256-token canvas with a random vocabulary
token, running one read-only denoise pass, and taking the logits at the slot positions over the
label tokens. DiffusionGemma is uniform-state diffusion, so one such read is one sample of the
marginal conditioned on the noise token, not the marginal; OpenJev averages up to four reads with
fresh seeds when a slot is uncertain.

This module reproduces that read exactly (same prompt, template, slot discovery and canvas as
OpenJev's engine, which it imports) and adds two controls the experiments need:

  * what the slot holds: a random token (`seed`), the vocabulary-mean embedding, or an arbitrary
    continuous embedding such as a learned query, injected at the decoder's input-embedding level
    so that gradients flow to it and nothing else;
  * the canvas attention mask, so slots can be isolated from one another.

The prefill (the encoder pass over the state and questions) is cached per prompt and never
receives gradients. Logits are computed only at the slot rows.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("openjev", "jevbench"):
    p = os.path.join(ROOT, "third_party", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from openjev.config import Settings          # noqa: E402
from openjev.engine import PAD, TURN_CLOSE, VOCAB, Engine  # noqa: E402

DEFAULT_MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
PREFILL_CACHE_TOKENS = 32768


def _stop_gradient_on_router_indices():
    """mlx-vlm only stop-gradients the MoE router's top-k indices in training mode; in eval mode a
    backward pass through the experts fails ("[gather] Cannot calculate VJP with respect to indices").
    Expert choice is discrete either way, so cutting the graph at the indices changes no forward value."""
    import mlx.core as mx
    from mlx_vlm.models.diffusion_gemma import language as L

    if getattr(L.Router, "_dlm_patched", False):
        return

    def patched(self, x):
        # mlx-vlm's Router.__call__, with the indices cut from the graph before they index anything,
        # an optional per-router allowed-expert mask (simulated carve, dlm/carve.py), and an optional
        # recorder for the expert census (one entry per layer call, in layer order)
        x = mx.fast.rms_norm(x, None, self.eps)
        x = x * self.scale * self._root_size
        scores = self.proj(x)
        allowed = getattr(self, "_dlm_allowed", None)      # carve: experts outside the set never win
        if allowed is not None:
            scores = mx.where(allowed, scores, mx.array(-float("inf"), dtype=scores.dtype))
        top_k = self.config.top_k_experts
        indices = mx.stop_gradient(mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:])
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = mx.softmax(weights, axis=-1, precise=True)
        weights = weights * self.per_expert_scale[indices]
        rec = L.Router._dlm_record
        if rec is not None:
            rec.append((indices, weights))
        return indices, weights

    L.Router.__call__ = patched
    L.Router._dlm_patched = True
    L.Router._dlm_record = None


class RoutingRecorder:
    """Collect (indices, weights) from every router call inside the block: 30 entries per decoder pass
    (one per layer, in order), each [tokens, top_k]."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        from mlx_vlm.models.diffusion_gemma import language as L
        L.Router._dlm_record = self.calls
        return self

    def __exit__(self, *exc):
        from mlx_vlm.models.diffusion_gemma import language as L
        L.Router._dlm_record = None


class Reader:
    def __init__(self, model_path: str = DEFAULT_MODEL, prefill_cache_tokens: int = PREFILL_CACHE_TOKENS):
        import mlx.core as mx
        from mlx_vlm import load

        _stop_gradient_on_router_indices()
        self.mx = mx
        self.model, self.processor = load(model_path, trust_remote_code=False)
        self.tok = getattr(self.processor, "tokenizer", self.processor)
        self.settings = Settings()
        self.engine = Engine(self.settings, self.tok)
        self.dec = self.model.model.decoder
        self.hidden = self.dec.config.hidden_size
        self._prefills: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._prefill_budget = prefill_cache_tokens
        self._mean_embed = None

    # -- request -> prompt, template, slots ---------------------------------------------------------

    def schema(self, questions: Dict[str, Dict]) -> Dict:
        return self.engine.build_schema(questions)

    def prepare(self, state_text: str, questions: Dict[str, Dict]) -> Dict:
        """Everything a read needs, built exactly as OpenJev builds it: chat prompt ids, the answer
        template, and each question's slot position and label token ids."""
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
        """The canvas token ids: template, turn close, padding; slots seeded from `seed` as OpenJev does,
        or left as the template's own token when seed is None (they are overwritten by embeddings)."""
        if seed is not None:
            return self.engine.build_canvas(prep["template"], prep["slots"], seed)
        c = list(prep["template"]) + [TURN_CLOSE]
        return c + [PAD] * (prep["width"] - len(c))

    # -- prefill cache ------------------------------------------------------------------------------

    def prefill(self, prompt: Sequence[int]):
        key = tuple(prompt)
        hit = self._prefills.get(key)
        if hit is not None:
            self._prefills.move_to_end(key)
            return hit[0]
        cache = self.model.diffusion_prefill_cache(input_ids=self.mx.array([list(prompt)]))
        self.mx.eval([c.state for c in cache])
        self._prefills[key] = (cache, len(prompt))
        while len(self._prefills) > 1 and sum(n for _, n in self._prefills.values()) > self._prefill_budget:
            self._prefills.popitem(last=False)
        return cache

    # -- embeddings ---------------------------------------------------------------------------------

    def embed(self, canvas_ids):
        """Input embeddings as the decoder computes them, before its input norm."""
        return self.dec.embed_tokens(canvas_ids) * self.dec.embed_scale

    def mean_embedding(self, chunk: int = 16384):
        """The vocabulary-mean input embedding (already scaled): the trivial deterministic slot."""
        if self._mean_embed is None:
            mx = self.mx
            total = mx.zeros((self.hidden,), dtype=mx.float32)
            for s in range(0, VOCAB, chunk):
                ids = mx.arange(s, min(s + chunk, VOCAB))
                total = total + self.embed(ids).astype(mx.float32).sum(axis=0)
                mx.eval(total)
            self._mean_embed = total / VOCAB
        return self._mean_embed

    def with_slots(self, h, positions: Sequence[int], rows):
        """`h` [1, L, D] with the rows at `positions` replaced by `rows` [n, D] (differentiably)."""
        mx = self.mx
        L = h.shape[1]
        keep = mx.ones((L, 1), dtype=h.dtype)
        put = mx.zeros((L, self.hidden), dtype=h.dtype)
        for i, p in enumerate(positions):
            onehot = (mx.arange(L) == p).astype(h.dtype)[:, None]
            keep = keep * (1 - onehot)
            put = put + onehot * rows[i].astype(h.dtype)[None, :]
        return h * keep[None] + put[None]

    # -- the read -----------------------------------------------------------------------------------

    def masks(self, cache, width: int, decoder_attention_mask=None):
        h_shape = self.mx.zeros((1, width, 1))
        return self.dec._make_decoder_masks(h_shape, cache, decoder_attention_mask)

    def slot_masks(self, cache, width: int, positions: Sequence[int], mode: str = "isolate"):
        """Canvas attention masks that control what sees the answer slots.

          "isolate"    a slot row attends to the state and the template but not to other slots
          "invisible"  no canvas row attends to any slot but itself: the template rows cannot carry
                       another slot's noise either, so each slot's read depends on its own content only
          "default"    the stock masks

        Returned as the dict `_make_decoder_masks` accepts unchanged. Keys are laid out as
        [encoder positions..., canvas positions...]; the sliding layers keep only the window's tail."""
        mx = self.mx
        from mlx_vlm.models.diffusion_gemma.language import _cache_offset, _cache_state

        base = self.masks(cache, width)
        if mode == "default":
            return base
        out = {}
        pos = set(positions)
        for layer_type in set(self.dec.config.layer_types):
            c = next((c for c, layer in zip(cache, self.dec.layers) if layer.layer_type == layer_type), None)
            state = _cache_state(c)
            enc_len = state[0].shape[2] if state is not None else 0
            valid = min(_cache_offset(c), enc_len)
            key_len = enc_len + width
            # start from what the stock mask allows (None means everything valid)
            if base.get(layer_type) is None:
                allow = mx.concatenate([mx.arange(enc_len) < valid, mx.ones((width,), dtype=mx.bool_)])
                allow = mx.broadcast_to(allow[None, None, None, :], (1, 1, width, key_len))
            else:
                allow = base[layer_type]
            col_is_slot = mx.array([False] * enc_len + [j in pos for j in range(width)])
            row_is_slot = mx.array([j in pos for j in range(width)])
            same = mx.concatenate([mx.zeros((width, enc_len), dtype=mx.bool_), mx.eye(width, dtype=mx.bool_)], axis=1)
            if mode == "isolate":
                # slot rows: drop other slot columns; template rows: unchanged
                block = row_is_slot[:, None] & col_is_slot[None, :] & ~same
            elif mode == "invisible":
                block = col_is_slot[None, :] & ~same
                block = mx.broadcast_to(block, (width, key_len))
            else:
                raise ValueError(mode)
            out[layer_type] = allow & ~block[None, None, :, :]
        return out

    def slot_logits(self, cache, inputs_embeds, positions: Sequence[int], masks):
        """Softcapped vocabulary logits at the slot rows, from input embeddings [1, L, D]."""
        mx = self.mx
        dec = self.dec
        h = dec.self_conditioning(inputs_embeds, mx.zeros_like(inputs_embeds))   # no self-conditioning signal
        offset = 0
        if cache:
            from mlx_vlm.models.diffusion_gemma.language import _cache_offset
            offset = _cache_offset(cache[0])
        for layer, c in zip(dec.layers, cache):
            h = layer(h, masks.get(layer.layer_type), c, decoder=True, offset=offset)
        h = dec.norm(h)
        rows = mx.take(h, mx.array(list(positions)), axis=1)          # [1, n, D]
        logits = dec.embed_tokens.as_linear(rows)
        return self.model._softcap(logits)[0]                          # [n, V] float32

    def label_logprobs(self, logits, label_ids: Sequence[Sequence[int]]):
        """Per slot: log-probabilities over that slot's label tokens, renormalised over them
        (OpenJev's slot_distribution at temperature 1)."""
        mx = self.mx
        out = []
        for i, ids in enumerate(label_ids):
            row = logits[i]
            lp = row - mx.logsumexp(row)
            sub = mx.take(lp, mx.array(list(ids)))
            out.append(sub - mx.logsumexp(sub))
        return out

    def topk_entropy(self, logits, label_ids: Sequence[Sequence[int]], k: int = 20) -> List[float]:
        """OpenJev's re-read trigger: the entropy of the full-vocabulary distribution restricted to the
        top-k tokens and the slot's labels (its `slot_distribution` entropy)."""
        mx = self.mx
        out = []
        for i, ids in enumerate(label_ids):
            row = logits[i]
            lp = row - mx.logsumexp(row)
            keep = sorted(set(mx.argpartition(-lp, k)[:k].tolist()) | set(ids))
            p = mx.exp(mx.take(lp, mx.array(keep)))
            out.append(float(-(p * mx.log(mx.maximum(p, 1e-30))).sum()))
        return out

    def read(self, prep: Dict, mode: str = "random", seed: int = 0, queries=None, masks=None,
             with_entropy: bool = False):
        """One read. `mode`: "random" (OpenJev: slot = random token from `seed`), "mean" (slot =
        vocabulary-mean embedding), "query" (slot i = queries[i], a [D] embedding). Returns a list of
        probability lists in label order, one per question (and the top-k entropies with
        `with_entropy`)."""
        mx = self.mx
        cache = self.prefill(prep["prompt"])
        positions = [s["pos"] for s in prep["slots"]]
        ids = mx.array([self.canvas(prep, seed if mode == "random" else None)])
        h = self.embed(ids)
        if mode == "mean":
            m = self.mean_embedding()
            h = self.with_slots(h, positions, [m] * len(positions))
        elif mode == "query":
            h = self.with_slots(h, positions, list(queries))
        elif mode != "random":
            raise ValueError(mode)
        if masks is None:
            masks = self.masks(cache, prep["width"])
        logits = self.slot_logits(cache, h, positions, masks)
        label_ids = [s["label_ids"] for s in prep["slots"]]
        lps = self.label_logprobs(logits, label_ids)
        mx.eval(lps)
        probs = [[float(v) for v in mx.exp(lp).tolist()] for lp in lps]
        if with_entropy:
            return probs, self.topk_entropy(logits, label_ids)
        return probs

    def reference_slot_logits(self, prep: Dict, seed: int = 0):
        """The slot logits by the stock path (`diffusion_decoder_logits` over token ids), for parity."""
        mx = self.mx
        cache = self.prefill(prep["prompt"])
        ids = mx.array([self.canvas(prep, seed)])
        masks = self.model.diffusion_decoder_masks(ids, cache, None)
        logits = self.model.diffusion_decoder_logits(ids, cache=cache, self_conditioning=None,
                                                     decoder_attention_mask=masks)
        pos = mx.array([s["pos"] for s in prep["slots"]])
        return logits[0, pos].astype(mx.float32)


def answer_dicts(prep: Dict, probs: List[List[float]]) -> Dict[str, Dict[str, float]]:
    """Question key -> {label name: probability}, using the engine's option names (yes/no for noul,
    the criteria keys for choice, level indices for score)."""
    out = {}
    for q, p in zip(prep["qs"], probs):
        out[q["key"]] = {name: float(v) for (name, _), v in zip(q["choices"], p)}
    return out
