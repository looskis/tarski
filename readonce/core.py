"""Read-once decision heads: split a trained laya cross-encoder at depth k.

Laya encodes one sequence per question:  [CLS] question [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] | state [SEP]
                                          '---------------- segment A (question) -----------'   '-- B (text) --'

With split depth k, encoder layers 0..k-1 see A and B separately (block-diagonal attention), so both
segments' layer-k states can be computed once and cached: B once per message, A once per deployment.
Layers k..27 run jointly on the concatenation, then laya's appended decision head scores the options.
k=0 is the original cross-encoder; k=28 means only the appended head ever sees both segments.

ModernBERT uses RoPE and index-distance sliding windows, both relative, so masking a segment inside the
concatenated sequence is numerically the same as encoding it alone from position 0. The accuracy
experiments use the mask form (simpler batching); `bench.py` uses true caching for latency.
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.checkpoint
from datasets import load_dataset
from torch import nn

import laya
from laya.common import QTYPES

DEVICE = torch.device(os.environ.get("READONCE_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))


def autocast():
    """fp16 autocast on CUDA and MPS (laya's README: closer to fp32 than bf16); a no-op on CPU."""
    return torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type != "cpu")


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def empty_cache():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


def mem_report() -> str:
    if DEVICE.type == "cuda":
        return f"cuda alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
    if DEVICE.type == "mps":
        return (f"mps alloc {torch.mps.current_allocated_memory() / 2**30:.1f} GiB, "
                f"driver {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB")
    return "cpu"


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

@dataclass
class Item:
    case: int            # index of the state (message) this question belongs to
    workflow: str
    qid: str
    ids: List[int]       # laya token ids, question segment first
    a_len: int           # length of segment A (through the SEP after the options)
    markers: List[int]   # option [MASK] positions, all inside A
    qtype: int
    target: np.ndarray   # gold soft distribution over options, laya option order
    label: int           # gold argmax label index


def _option_keys(q: Dict) -> List[str]:
    if q["type"] == "choice":
        return list(q["criteria"].keys())
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def build_items(agent, split: str, sink: bool = False) -> List[Item]:
    """Tokenize every (state, question) pair of typed-decisions with laya's own sequence builder.

    `sink=True` inserts a [CLS] at the start of segment B, giving the independently-encoded text an
    attention-sink token of its own (the question segment already starts with [CLS]).
    """
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split=split)
    tok = agent.tok
    items: List[Item] = []
    for ci, row in enumerate(ds):
        state, qs, gold = json.loads(row["state"]), json.loads(row["questions"]), json.loads(row["gold"])
        qids = list(qs)
        internal = {k: agent._to_internal(qs[k]) for k in qids}
        encoded = agent._encode_state(state, qids, internal)
        for qid, enc in zip(qids, encoded):
            ids = list(enc["ids"])
            seps = [i for i, t in enumerate(ids) if t == tok.sep_token_id]
            a_len = seps[1] + 1
            if sink:
                ids = ids[:a_len] + [tok.cls_token_id] + ids[a_len:]
            keys = _option_keys(qs[qid])
            probs = gold[qid]["probabilities"]
            target = np.array([float(probs.get(k, 0.0)) for k in keys], dtype=np.float32)
            target = target / target.sum()
            items.append(Item(ci, row["workflow"], qid, ids, a_len, list(enc["markers"]), int(enc["qtype"]),
                              target, keys.index(str(gold[qid]["label"]))))
    return items


def split_train_val(items: List[Item], val_frac: float = 0.1, seed: int = 0):
    cases = sorted({it.case for it in items})
    rng = random.Random(seed)
    rng.shuffle(cases)
    val_cases = set(cases[: int(len(cases) * val_frac)])
    return [it for it in items if it.case not in val_cases], [it for it in items if it.case in val_cases]


def collate(items: List[Item], pad_id: int, device=DEVICE) -> Dict[str, torch.Tensor]:
    n, L = len(items), max(len(it.ids) for it in items)
    L = -(-L // 32) * 32                                 # few distinct shapes keep the MPS allocator cache small
    K = max(len(it.markers) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    seg = torch.ones((n, L), dtype=torch.long)          # 0 = question segment, 1 = text segment/pad
    mpos = torch.zeros((n, K), dtype=torch.long)
    mmask = torch.zeros((n, K), dtype=torch.bool)
    target = torch.zeros((n, K))
    for i, it in enumerate(items):
        ids[i, : len(it.ids)] = torch.tensor(it.ids)
        att[i, : len(it.ids)] = 1
        seg[i, : it.a_len] = 0
        k = len(it.markers)
        mpos[i, :k] = torch.tensor(it.markers)
        mmask[i, :k] = True
        target[i, :k] = torch.from_numpy(it.target)
    qtype = torch.tensor([it.qtype for it in items])
    label = torch.tensor([it.label for it in items])
    out = dict(input_ids=ids, attention_mask=att, segment=seg, marker_pos=mpos, marker_mask=mmask,
               target=target, qtype=qtype, label=label)
    return {k: v.to(device) for k, v in out.items()}


def batches(items: List[Item], bs: int, shuffle: bool = False, seed: int = 0):
    order = sorted(range(len(items)), key=lambda i: len(items[i].ids))   # length-bucketed
    chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        random.Random(seed).shuffle(chunks)
    for c in chunks:
        yield [items[i] for i in c]


# ---------------------------------------------------------------------------------------------
# Split encoder forward
# ---------------------------------------------------------------------------------------------

def layer_masks(encoder, attention_mask: torch.Tensor, segment: torch.Tensor, k: int):
    """Per-layer boolean 4D masks (True = attend). Layers < k are block-diagonal over segments.
    `encoder` may be a ModernBertModel or its config."""
    cfg = getattr(encoder, "config", encoder)
    L = attention_mask.shape[1]
    valid_k = attention_mask.bool()[:, None, None, :]                       # key padding
    idx = torch.arange(L, device=attention_mask.device)
    window = (idx[:, None] - idx[None, :]).abs() <= cfg.sliding_window      # ModernBERT local attention
    same_seg = (segment[:, :, None] == segment[:, None, :])[:, None]        # (n,1,L,L)
    eye = torch.eye(L, dtype=torch.bool, device=attention_mask.device)      # keep pad rows finite
    masks = []
    for li, lt in enumerate(cfg.layer_types):
        m = valid_k.expand(-1, 1, L, L)
        if lt == "sliding_attention":
            m = m & window
        if li < k:
            m = m & same_seg
        masks.append(m | eye)
    return masks


def run_layers(config, rotary, layers, h, attention_mask, segment, first: int, k: int,
               checkpoint: bool = False) -> torch.Tensor:
    """Run encoder layers first, first+1, ... (given as `layers`) with split depth k.
    `checkpoint` recomputes each layer's activations in backward (training many layers on a small GPU)."""
    pos = torch.arange(h.shape[1], device=h.device)[None]
    rope = {lt: rotary(h, pos, lt) for lt in set(config.layer_types)}
    masks = layer_masks(config, attention_mask, segment, k)
    for i, layer in enumerate(layers):
        kw = dict(attention_mask=masks[first + i], position_embeddings=rope[layer.attention_type])
        if checkpoint:
            h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False, **kw)
        else:
            h = layer(h, **kw)
    return h


def encode_lower(encoder, input_ids, attention_mask, segment, k: int) -> torch.Tensor:
    """Hidden states entering layer k; segments never see each other, so these are cacheable per segment."""
    h = encoder.embeddings(input_ids=input_ids)
    return run_layers(encoder.config, encoder.rotary_emb, encoder.layers[:k], h, attention_mask, segment, 0, k)


def encode_split(encoder, input_ids, attention_mask, segment, k: int) -> torch.Tensor:
    """ModernBertModel.forward with per-layer attention masks; returns final-normed hidden states."""
    h = encode_lower(encoder, input_ids, attention_mask, segment, k)
    h = run_layers(encoder.config, encoder.rotary_emb, encoder.layers[k:], h, attention_mask, segment, k, k)
    return encoder.final_norm(h)


class Upper(nn.Module):
    """Encoder layers k..end plus the final norm, as a trainable per-task module over cached lower states.
    These layers run once per question anyway, so tuning them costs nothing extra at inference."""

    def __init__(self, encoder, k: int):
        super().__init__()
        self.config, self.k = encoder.config, k
        self.layers = copy.deepcopy(encoder.layers[k:])
        self.final_norm = copy.deepcopy(encoder.final_norm)
        self.rotary = copy.deepcopy(encoder.rotary_emb)

    def forward(self, h, attention_mask):
        seg = torch.zeros_like(attention_mask)          # every layer here is joint
        ckpt = self.training and torch.is_grad_enabled() and len(self.layers) > 8
        h = run_layers(self.config, self.rotary, self.layers, h, attention_mask, seg, self.k, self.k, checkpoint=ckpt)
        return self.final_norm(h)


def head_forward(model, h, attention_mask, marker_pos, marker_mask, qtype) -> torch.Tensor:
    """Laya's appended decision head (DecisionModel.forward after the encoder); returns option logits."""
    h = h + model.type_emb(qtype)[:, None, :]
    if model.head is not None:
        pad = ~attention_mask.bool()
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    logits = model.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
    return logits.masked_fill(~marker_mask, -1e4)


def load_agent(name: str = "english"):
    sub = {"english": None, "typed-decisions": "typed-decisions", "multilingual": "multilingual"}[name]
    return laya.load("convaiinnovations/laya", subfolder=sub, device=str(DEVICE))


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------

def metrics(logits: List[np.ndarray], items: List[Item]) -> Dict[str, float]:
    """Accuracy vs gold argmax (laya's benchmark metric), Brier vs gold soft labels, top-label ECE."""
    acc, brier, conf, corr = [], [], [], []
    by_type: Dict[str, List[float]] = {t: [] for t in QTYPES}
    for z, it in zip(logits, items):
        z = z[: len(it.markers)]
        p = np.exp(z - z.max()); p /= p.sum()
        ok = float(int(p.argmax()) == it.label)
        acc.append(ok)
        by_type[[t for t, v in QTYPES.items() if v == it.qtype][0]].append(ok)
        brier.append(float(((p - it.target) ** 2).sum()))
        conf.append(float(p.max())); corr.append(ok)
    conf, corr = np.array(conf), np.array(corr)
    bins = np.linspace(0, 1, 16)
    ece = sum(abs(conf[(conf > lo) & (conf <= hi)].mean() - corr[(conf > lo) & (conf <= hi)].mean())
              * ((conf > lo) & (conf <= hi)).mean() for lo, hi in zip(bins[:-1], bins[1:])
              if ((conf > lo) & (conf <= hi)).any())
    out = {"acc": float(np.mean(acc)), "brier": float(np.mean(brier)), "ece": float(ece), "n": len(items)}
    out.update({f"acc_{t}": float(np.mean(v)) for t, v in by_type.items() if v})
    return out
