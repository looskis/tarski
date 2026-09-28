"""Skeptic lens, idea 2: does a task branch's split depth track raw layer count, or the number of
ModernBERT's GLOBAL (full-attention) layers that ran below the split?

ModernBERT-base alternates attention types (confirmed via AutoConfig): layers 0, 3, 6, 9, 12, 15, 18, 21
(0-indexed) are `full_attention`; the other 14 are `sliding_attention` with a 64-token radius
(`config.sliding_window == 64`, i.e. a ~129-token window). banking77 and clinc150 both use
`max_len=64` (tarski/data.py load_banking77 / load_clinc) -- shorter than the window diameter -- so
EVERY "sliding" layer behaves as full attention on those two datasets, and none of the depth-accuracy
curves measured on them (results/tarski/tables.md) can say anything about windowing. typed-decisions
uses `max_len=512` (median 238 tokens; 72% of examples exceed 128 tokens, measured directly from
tarski.data.load_typed_decisions()), so it is the only dataset in the current results where a token's
receptive field can actually be capped by the window. At split depth k, only floor-ish counts of the 8
global layers have run (2 at k=6, 4 at k=11, 6 at k=16, 8 at k=22 -- the exact splits tables.md reports
typed-decisions probes at) -- so two JSON fields more than ~65-130 tokens apart may simply not have had
a chance to interact yet at the shallower splits, independent of how "deep" k sounds.

This script builds a fully synthetic, order-sensitive "field-order" task modelled directly on typed-
decisions tasks that must compare two possibly-distant fields (e.g. invoice_processing.matches_order,
invoice_processing.duplicate): two marker codes are placed near the start and end of a message with a
controlled amount of filler between them, and the label is whether the first marker's fixed rank is
below the second's. This is deliberately NOT a "do these two tokens match" task, because a mean-pooled,
position-invariant probe could solve token-repetition matching as a bag-of-words statistic at depth 0,
with no attention at all -- that would test nothing. Order/rank comparison cannot be recovered from an
unordered token multiset (the pair {a, b} is identical text-content-wise whichever one comes first;
only the label differs), so a linear probe can only succeed once the trunk has actually bound each
marker's identity to its position and propagated that across the gap.

For several gaps (fits in one window vs several window-radii) and several decoy-field counts, it probes
accuracy at every depth (the same cheap linear probe autosplit.py itself uses for its split-depth
curves) and reports it alongside the number of global layers that have run by that depth.

Closest prior work: the l * w receptive-field growth of stacked local/sliding-window attention is
textbook (Beltagy, Peters & Cohan, "Longformer", 2020); needle-style controlled probes of usable context
by depth/position are standard in the long-context literature (e.g. Liu et al., "Lost in the Middle",
TACL 2023/2024). Applying this specifically to whether a PER-TASK STATIC split-depth selector (as in
tarski/autosplit.py) should account for global-layer count rather than raw depth for long inputs, in a
hybrid local/global attention encoder used for multi-task branch serving, was not found in the sources
reviewed in docs/research/notes/novelty_check.md -- the closest relative there (Wei et al., ACL 2022) uses
plain BERT, which has no local/global distinction at all.

IMPORTANT -- what a CPU pilot run of this exact script already showed (see docs/research/notes/explore/
skeptic.md item on this hypothesis): with 0 decoys, accuracy jumps from ~chance at depth 0 to ~0.9-1.0
at depth 1 for EVERY gap tested, including gap=400 (several window-radii). This refutes the hypothesis
in its strong form: ModernBERT-base's layer 0 is itself `full_attention` (global), so every depth >= 1
already gives every token a direct, unwindowed attention step to every other token -- no split depth
actually used anywhere in tarski's experiments (all >= 4) is ever short of at least one full sequence-
wide mixing round. Adding up to 8 decoys did not change this. The remaining, weaker and still-open
question -- whether SELECTIVE routing among many similarly-formatted fields (as in a real, cluttered
JSON state with dozens of keys, not 2-10 decoys) needs more than one global layer's worth of refinement
-- is why this script sweeps `--decoys` as a list: the full run pushes decoy count much higher to give
that weaker hypothesis a real chance to show up before concluding depth truly doesn't matter here.

Usage:
  python -m explore.skeptic_receptive_field --smoke
  python -m explore.skeptic_receptive_field --out results/tarski/explore_receptive_field.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch

from tarski.autosplit import pooled_by_depth, probe_accuracy
from tarski.trunk import Trunk

CODES = ["Q7F", "K3M", "Z9P", "L8R", "T6N", "V2X", "H5D", "W4C"]   # fixed rank = list index
FILLER = ("The nightly batch job completed without errors and the queue drained normally. "
         "Support tickets from the previous shift were all closed before the handover meeting. ")
DECOYS = ["note field is {c}.", "internal tag is {c}.", "batch code is {c}.", "queue label is {c}."]


def make_example(gap_tokens: int, tok, rng: random.Random, n_decoys: int = 0) -> Tuple[str, int]:
    """Two distinct codes, one near the start and one near the end; label = does the first one's
    fixed rank precede the second's. The token *set* {a, b} carries no label information -- only
    which one is first vs second does -- so this cannot be solved as a bag-of-words statistic.

    `n_decoys` scatters similarly-formatted but irrelevantly-named fields (same code vocabulary)
    through the filler: a real JSON state has many fields, not two, so a branch also has to learn to
    single out the two that matter by NAME, not just detect that *some* order-sensitive pair exists
    reachable within its depth. This checks whether that selective-routing task (as opposed to bare
    reachability) is what actually gets harder at shallow depth on long inputs."""
    ia, ib = rng.sample(range(len(CODES)), 2)
    ca, cb = CODES[ia], CODES[ib]
    filler = ""
    while len(tok(filler)["input_ids"]) < gap_tokens:
        filler += FILLER
    if n_decoys:
        chunks = [c for c in filler.split(". ") if c]
        decoys = [rng.choice(DECOYS).format(c=rng.choice(CODES)) for _ in range(n_decoys)]
        step = max(1, len(chunks) // (n_decoys + 1))
        out, di = [], 0
        for i, ch in enumerate(chunks):
            out.append(ch)
            if di < len(decoys) and (i + 1) % step == 0:
                out.append(decoys[di])
                di += 1
        out.extend(decoys[di:])
        filler = ". ".join(out)
    text = f"first marker is {ca}. " + filler + f" second marker is {cb}."
    return text, int(ia < ib)


def make_dataset(gap_tokens: int, n: int, tok, seed: int, n_decoys: int = 0) -> Tuple[List[str], np.ndarray]:
    rng = random.Random(seed * 1000 + gap_tokens)
    rows = [make_example(gap_tokens, tok, rng, n_decoys) for _ in range(n)]
    return [r[0] for r in rows], np.array([r[1] for r in rows])


def global_layers_below(trunk: Trunk, depth: int) -> int:
    return sum(1 for i, t in enumerate(trunk.cfg.layer_types) if t == "full_attention" and i < depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny subset, finishes in well under 2 minutes")
    ap.add_argument("--gaps", type=int, nargs="*", default=None, help="filler length in tokens per condition")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--depths", type=int, nargs="*", default=None)
    ap.add_argument("--decoys", type=int, nargs="*", default=None,
                    help="distractor-field counts to sweep, scattered through the filler with the same "
                         "code vocabulary but irrelevant field names -- tests selective routing amid "
                         "clutter, not bare reachability (see module docstring)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    device = "cpu" if a.smoke else a.device
    gaps = a.gaps or ([0, 200] if a.smoke else [0, 32, 150, 400])
    decoys = a.decoys if a.decoys is not None else ([0] if a.smoke else [0, 8, 24])
    n_train = a.n_train or (48 if a.smoke else 240)
    n_val = a.n_val or (24 if a.smoke else 80)
    max_len = 512

    trunk = Trunk(device=device, max_len=max_len)
    # depth 0 = raw embeddings, no attention at all -- the only depth with provably zero position-aware
    # mixing, since ModernBERT's layer 0 is itself a full_attention (global) layer (see
    # global_layers_below_depth below): every depth >= 1 already has at least one global attention pass.
    depths = a.depths or (list(range(0, 9)) if a.smoke else [0] + list(range(1, trunk.n_layers + 1)))
    depths = [d for d in depths if 0 <= d <= trunk.n_layers]

    t0 = time.time()
    curves = {}
    for n_dec in decoys:
        for gap in gaps:
            key = f"gap={gap},decoys={n_dec}"
            texts_tr, y_tr = make_dataset(gap, n_train, trunk.tok, a.seed, n_dec)
            texts_va, y_va = make_dataset(gap, n_val, trunk.tok, a.seed + 1, n_dec)
            lens = [len(trunk.tok(t)["input_ids"]) for t in texts_tr[:5]]
            feats_tr = pooled_by_depth(trunk, texts_tr, depths, max_len)
            feats_va = pooled_by_depth(trunk, texts_va, depths, max_len)
            ytr_t, yva_t = torch.tensor(y_tr, dtype=torch.long), torch.tensor(y_va, dtype=torch.long)
            curve = {}
            for d in depths:
                acc = probe_accuracy(feats_tr[d], ytr_t, feats_va[d], yva_t, 2, trunk.device,
                                     steps=150 if a.smoke else 300)
                curve[d] = round(acc, 4)
            curves[key] = curve
            print(f"{key:22s} (~{lens[0]} tok, base rate {y_tr.mean():.2f}) " +
                 " ".join(f"d{d}:{curve[d]:.2f}" for d in depths))

    global_counts = {str(d): global_layers_below(trunk, d) for d in depths}
    # simple summary: for each condition, the shallowest depth reaching 0.75 accuracy (or None)
    onset = {}
    for key, curve in curves.items():
        hit = [d for d in depths if curve[d] >= 0.75]
        onset[key] = {"onset_depth": (min(hit) if hit else None),
                     "global_layers_at_onset": (global_layers_below(trunk, min(hit)) if hit else None)}

    results = {"base": trunk.base, "n_layers": trunk.n_layers, "depths": depths, "decoys": decoys,
              "global_layers_below_depth": global_counts, "gaps": gaps,
              "n_train": n_train, "n_val": n_val, "curves": curves, "onset_at_0.75_acc": onset,
              "seconds": round(time.time() - t0, 1)}
    print(json.dumps(results, indent=2))

    default_out = "results/tarski/explore_receptive_field_smoke.json" if a.smoke \
        else "results/tarski/explore_receptive_field.json"
    out = a.out or default_out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
