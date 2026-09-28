# Skeptic lens: where the thesis breaks, and what to do about it

Method: read `tarski/train.py`, `tarski/data.py`, `tarski/branches.py`, `tarski/trunk.py`,
`tarski/autosplit.py`, `tarski/fused.py`, `tarski/engine.py`, all of `experiments/`, and
`results/tarski/tables.md`; then measured things directly (dataset stats, ModernBERT's actual
`layer_types`/`sliding_window`, and recomputations from the JSON result files) rather than trusting the
summary in `results/tarski/tables.md` alone, and ran two small stress-test scripts against the real
model on CPU. Everything below with a specific number was verified by running code in this repo, not
inferred from the prose in `BRIEF.md`.

**Follow-up pass (same session, after the GPU queue ran the first round of experiments).** The
early-stopping bug (item 4) was fixed in `tarski/train.py`, ideas 1 and 2 were run to completion on the
GPU, and `experiments/autosplit_eval.py` / `experiments/ablation_init.py` were finally run (item 5 was
previously unverified; it no longer is). This pass: (a) folds those real results back into every item
below with a **Status** line (tested / queued / not testable), (b) implements ideas 3 and 6 with a
testable prediction each, (c) quantifies bug 2 directly, and (d) adds one new item (7) found while
re-checking `results/tarski/ablation_init.json`, now that it exists. Ideas 4 and 5 are out of scope for
this pass (the coordinator is covering idea 4 as a separate follow-up; idea 5 is covered by the
token-attention pooling in `explore/farfield_coarsegrain.py`) — their entries are left as originally
written, marked skipped.

---

## (a) Actual bugs / methodological problems

### 1. CLINC's `oos` branch has a large, uncorrected train/test class-prior shift — this is most of "weak OOS detection"

**Status: TESTED at full GPU scale, confirmed.** `results/tarski/lambda/explore_oos_priorshift.json`
(probe@11, full 15250/3100/5500 CLINC150, 20 epochs): the Saerens-EM correction moved
**oos-F1 0.292 → 0.570, macro-F1 0.603 → 0.739**, and matched the oracle-prior upper bound almost
exactly (oracle oos-F1 0.566, macro-F1 0.740) *without ever seeing test labels* — the EM prior estimate
(0.769/0.231) converged close to the true test prior (0.818/0.182). Confirms the CPU-scale numbers
below were not a fluke.

**Evidence.** Measured directly from `tarski.data.load_clinc()`: the out-of-scope fraction is
**1.64% in train, 3.23% in val, but 18.18% in test** (this is CLINC150's own "plus"-split design,
Larson et al., EMNLP 2019 — not a tarski bug in the data loader). `tarski/train.py` trains the `oos`
branch with plain cross-entropy (`train_branch`, `tarski/train.py:150-200`, loss at line 180-182) and
calibrates it with **one scalar temperature** (`fit_temperature`, `tarski/train.py:103-115`, called at
`tarski/train.py:242-244`). A single temperature can sharpen or flatten a softmax; it cannot move mass
between classes to correct an asymmetric prior shift. Result: `results/tarski/tables.md`'s oos accuracy
(85-87%) barely beats the 81.8% "always say in-scope" baseline — a classic symptom of a classifier whose
implicit decision threshold reflects the ~2% training prevalence being run against 18% test prevalence.
This exact CLINC150 train/test OOS mismatch is independently documented in two 2024-2025 papers found
while checking novelty (see idea 1 below), which corroborates the numbers above are a known artifact of
this benchmark, not something specific to how tarski split-loads it.
**Fix implemented and empirically confirmed** — see idea 1 and `explore/skeptic_oos_priorshift.py`.

### 2. Calibration is fit to two different targets depending on the dataset, but reported with one ECE metric

**Status: QUANTIFIED, direction confirmed; full run queued.** Implemented as idea 3,
`explore/skeptic_dual_calibration.py` (fits `T_hard` and `T_soft` from the same validation logits and
reports test ECE/Brier under both). Smoke run (3 invoice_processing tasks, real val sizes, CPU) already
shows the predicted direction: mean hard-ECE under today's default `T_soft` = 0.275 vs. `T_hard` = 0.258
— today's calibration is measurably worse at the thing ECE actually reports, even at toy scale. New
corroborating evidence from the now-fixed training loop (item 4): after the early-stopping fix, typed-
decisions' mean mean-ECE across probe depths got **worse**, not better, even as accuracy improved
(0.085-0.104 pre-fix → 0.129-0.141 post-fix, from `sweep_typed_buggy_earlystop.json` vs the current
`sweep_typed.json`) — consistent with branches that now train for their full, intended number of steps
producing sharper (less-regularized-by-accident) soft-fit posteriors that drift further from hard-ECE
calibration. The full 20-task run is queued (`explore/QUEUE_skeptic.txt`) for a trustworthy number.

**Evidence.** `fit_temperature(logits, y, soft=None)` (`tarski/train.py:103-115`) minimizes cross-entropy
against `soft` when it is available, and against hard one-hot otherwise. `tarski/train.py:242` passes
`soft["val"]` whenever it exists. Banking77/CLINC150 examples carry no `soft` field
(`tarski/data.py` `load_banking77`/`load_clinc`), so their temperature is fit the textbook way (Guo et
al., ICML 2017: minimize NLL of hard correctness). typed-decisions is the **only** dataset with soft
teacher labels, so it is the only one whose temperature is fit to match a teacher's *label
distribution* instead of the branch's own *correctness*. But `ece_score` (`tarski/train.py:81-88`) always
measures `|confidence − hard-correctness|`, regardless of what the temperature was optimized for. This
mismatch (optimize-for-soft-report-as-hard) is a plausible, previously unstated reason typed-decisions'
ECE (0.085-0.104 in `tables.md`) is 5-10x worse than banking77's (0.010-0.021) even after "calibration" —
it may not reflect worse-calibrated models so much as calibration aimed at a different target. Not
tested further here (would need a core-file change to `fit_temperature` to compare both objectives);
flagged as idea 3 below.

### 3. There is no same-architecture "full fine-tune" reference for CLINC or typed-decisions — the headline "51-55% vs 76.6%" compares two different models

**Status: PARTIALLY RESOLVED.** banking77 and CLINC150 now have proper full-fine-tune baselines with
3-seed variance (`results/tarski/lambda/sweep_banking77_s{0,1,2}.json`,
`sweep_clinc150_s{0,1,2}.json`): banking77 full mean acc 93.14/93.89/94.25% (seeds 1/2/0), CLINC150 full
mean acc 90.58/90.89/90.87% (seeds 0/1/2) — both now directly comparable to the blocks/probe rows in the
same files. typed-decisions is **still unresolved**, and for a concrete reason: a full 22-layer fine-tune
of ModernBERT-base at this batch size **OOMs on a 24GB A10**
(`results/tarski/lambda/run_typed_cs_s0.out`: `CUDACachingAllocator ... memory allocation failed with
OOM on device 0`, `total: 23696375808` bytes ≈ 22 GiB, during the backward pass of
`customer_service.action`'s full-depth `BlockBranch`). The coordinator reports a customer_service
headline of `blocks@14+2` = 78.6% vs. `full` = 79.4% (seed 0, 3 seeds "finishing now" — presumably
re-run with a smaller batch size or on an A100); I could not independently verify 79.4% from a saved
JSON at the time of this check, only the `blocks@14+2` side (`results/tarski/lambda/sweep_typed_cs_s0.json`:
78.59%, matches). Two things worth flagging for whoever finishes this: (a) full fine-tuning ModernBERT-
base needs more than 24GB at the batch size `experiments/sweep.py` uses by default — an A10 is
insufficient for the `full` config specifically, even though every `blocks`/`probe` config runs fine on
it; (b) once the 20-task full baseline lands, the "78.6 vs 79.4" gap (0.8 points, on customer_service
alone) is far smaller than the pre-fix "51-55% vs 76.6%" headline suggested — most of that original gap
looks explained by item 4's bug, not by the split-trunk architecture.

**Evidence (original finding, banking77/CLINC150 half now stale — see Status above).** `results/tarski/sweep_clinc150.json` and `results/tarski/sweep_typed_buggy_earlystop.json`
(the typed-decisions sweep file — renamed during this session from `sweep_typed.json`, see item 4)
contain **no `"full"` key at all** (checked directly — only `probe@*`/`blocks@*` rows exist;
`sweep_clinc150.json` is also mid-run per its own file timestamp). Only `sweep_banking77.json` has one, and even there only for
its single task (`experiments/sweep.py:32-33`: `--full-tasks` defaults to "the first task only, since a
full fine-tune per task is the slow part"; the config itself is `experiments/sweep.py:58-59`, 5 epochs,
`lr_layers=5e-5` — notably *lower* than the 1e-4 given to every `blocks` config in the same sweep, and
the embedding table is never fine-tuned in any of these "full" runs either, since `BlockBranch` only
ever copies encoder layers, never `trunk.model.embeddings`). So the "typed-decisions probes collapse:
mean 51-55% vs laya 76.6%" framing in `BRIEF.md` compares tarski's ModernBERT-**base** (110M, frozen
trunk + probe/blocks head) against **laya**, a completely different, larger model
(ModernBERT-**large** cross-encoder, ~395M, question-in-input architecture, its own training recipe).
Without a full fine-tune of the *same* ModernBERT-base trunk on typed-decisions, there is no way to
attribute the gap between "split trunk hurts," "ModernBERT-base is just smaller/weaker than laya-large,"
and "a mean-pool classification head is worse than a cross-encoder's." This is the single most important
missing baseline for interpreting the project's central negative result.

### 4. `train_branch`'s early-stopping/warmup interaction throws away most of the epoch budget on typed-decisions — likely the dominant cause of "probe collapse," not depth or pooling

**Status: FIXED and confirmed.** `tarski/train.py:150-216` now has `min_steps=300` (raises `epochs` so
small datasets get enough optimizer steps regardless of batch count), `min_val=50` (below that, val
accuracy is too noisy to select or stop on at all — the branch just runs the full schedule and keeps the
last epoch, `select="last_epoch"`), and `min_epochs = (epochs+1)//2` (never stop before half the
schedule, i.e. never stop still inside the LR warm-up). Effect, from the now-current
`results/tarski/sweep_typed.json`: mean accuracy across all 20 typed-decisions tasks went from
**51.5/51.8/50.4/55.0% (buggy, non-monotonic in depth) to 57.6/58.5/59.9/61.6% (fixed, monotonic in
depth)** for probe@6/11/16/22 — a real, substantial improvement, and the erratic depth-dependence is
gone. Still well below laya's 76.6% and (once available) the same-architecture full-fine-tune reference
in item 3, so this was a real contributor but evidently not the whole story. **New, currently-unresolved
side effect: mean ECE got worse after the fix (0.085-0.104 → 0.129-0.141), even though accuracy
improved** — see item 2 / idea 3, which was built specifically to check whether this is the calibration-
target mismatch showing through now that training is no longer accidentally truncated.

**Evidence (own analysis of `results/tarski/sweep_typed_buggy_earlystop.log` — note this file was
renamed by another exploration pass during this session, from `sweep_typed.log`, flagging the same
thing; I verified the mechanism independently rather than taking the rename on faith, and the
farfield-lens notes at `research_notes/explore/farfield.md` document the same finding from a different
angle).** `train_branch` (`tarski/train.py:150-200`) tracks a `OneCycleLR` schedule with
`pct_start=0.1` (`tarski/train.py:165-166` — for `epochs=20`, ~2 epochs of rising learning rate before
the real, high-LR training phase even starts) and stops with `patience=3` on a **strict** `va > best`
comparison (`tarski/train.py:192-197` — a *tied* validation accuracy counts as "no improvement"). typed-
decisions' per-task validation sets are 22-41 rows (`tarski/data.py:56-60`, `_carve` with
`val_frac=0.1`), where ties between epochs are common (only ~30-40 distinct achievable accuracy values).
Parsing every probe run in the log directly: **of 80 branch-runs (20 tasks x 4 depths), 70% stop at
epoch ≤5, and 62.5% (50/80) stop at exactly epoch 4 — the earliest point `patience=3` allows** (best at
epoch 1, three flat/non-improving epochs after it, break) — out of a 20-epoch budget. Concrete example,
`customer_service.churn_risk` probe@11: val acc is **flat at 0.5161 for epochs 1-4** (each epoch tied,
not strictly better, so `bad` increments every time), training stops at epoch 4, and the "best" checkpoint
kept is whatever epoch 1 produced — before the OneCycle schedule has even finished warming up, let alone
annealed to a useful learning rate. `customer_service.category` probe@11 similarly stops at epoch 8 with
a checkpoint from epoch 5 (val acc 0.4839, test acc only 0.32 — one of the "below majority baseline"
results in item 5 below). This is a plausible, code-level, quantified explanation for most of the
reported "probe collapse" — an artifact of premature early stopping interacting with tiny validation
sets and a warmup-heavy LR schedule, not evidence about what the representations at any given depth can
actually support. This directly answers the brief's "is it a bug in the harness" question about the
typed-decisions probe collapse: yes, and it has now been fixed (see Status above) without touching the
architecture. **Seed variance, now measured (previously flagged here as missing):**
`results/tarski/ablation_init.json` and 3-seed sweeps now exist
(`results/tarski/lambda/sweep_{banking77,clinc150,typed_cs}_s{0,1,2}.json`). The spread scales with
dataset/task size roughly as item 4's own diagnosis would predict: banking77 full-fine-tune mean acc
across 3 seeds spans 93.14-94.25% (**1.1-point spread**); CLINC150 full spans 90.58-90.89% (**0.3-point
spread**); typed-decisions customer_service (the smallest data, 259-278 train rows/task) spans
72.8-75.4% for `blocks@20+2` (**2.6-point spread**) and 71.0-72.6% for `probe@22` (**1.6-point spread**)
across only the 3 seeds tried. So every single-seed number anywhere in `tables.md`, including the depth
curves the project's headline claims rest on, should be read with roughly this much uncertainty attached
— worse for typed-decisions than for banking77/CLINC150, consistent with the tiny-validation-set
concern above compounding with ordinary training-seed noise.

### 5. The probe-guided split selector (`tarski/autosplit.py`) can pick a depth that is measurably bad for the actual blocks branch — and this has never been checked against the sweep grid

**Status: CONFIRMED by the actual experiment — worse than my hand-estimate.**
`experiments/autosplit_eval.py` has now been run (`results/tarski/autosplit_banking77.json`,
`autosplit_clinc150.json`). On the real, fine-grained probe curve (not the 6 coarse points I recomputed
by hand below), the selector is even more aggressive: for banking77 intent it picks **depth 1-3 at every
tolerance tested (0.005/0.01/0.02)**, and the resulting blocks accuracy is 90.6-90.9% — a **2.2-2.5 point
gap** below the true best (93.1% at depth 18, from `tables.md`). For CLINC150 it picks **depth 1 for
all three tasks (intent, domain, oos) at every tolerance tested**; `blocks@1`'s domain accuracy is
85.4% vs. the sweep's `blocks@11+2` domain accuracy of 87.6% — a **2.2 point gap**, and vs. the now-
available full-fine-tune ceiling (item 3) the gap is much larger still. This is real, not a hypothetical:
the selector, run exactly as designed, would hand a deployment a meaningfully worse branch than the
sweep grid already knows is achievable, on two different datasets.

**Evidence (the hand-recomputation that predicted this, before the real run existed).** `choose_split` (`tarski/autosplit.py:78-80`) picks the *shallowest* depth whose *probe*
accuracy is within `tol` (default 0.01) of the probe curve's max — using probe accuracy only, never
blocks accuracy. `tables.md`'s own banking77 probe row is close to flat: 87.9 / 88.6 / 88.1 / 87.3 /
87.5 / 88.7 at depths 4/8/11/14/18/22. Recomputing `choose_split` on exactly these six points at the
documented default `tol=0.01`: best = 88.7 (depth 22), threshold = 87.7, and depths {4, 8, 11, 22} all
clear it (87.9, 88.6, 88.1, 88.7 ≥ 87.7) — so the selector returns **depth 4**, the shallowest. But the
**blocks** branch at that same depth 4 (`blocks@4+1` = 90.2) is **2.9 accuracy points worse** than
`blocks@18+1` (93.1) — a gap the probe-based selector is structurally blind to, because it never looks
at blocks accuracy at all. This is not a code bug (the function does exactly what it says, and the
`tol` parameter exists precisely so a caller CAN ask for a deeper, safer split) — but the *default*
`tol=0.01` used throughout, combined with a probe curve that is close to flat almost everywhere,
means the out-of-the-box selector picks the shallowest depth that clears a very forgiving bar, and (per
the Status above) really does hand back a measurably worse branch than the sweep grid already knows is
available. Recommendation: either raise the default `tol` substantially, or change `choose_split` to
require the chosen depth's *blocks* accuracy (a cheap 1-2 epoch check, idea 4 below / the coordinator's
H4 follow-up) to also clear a bar before returning it.

### 6. Sliding-window attention is architecturally a no-op for banking77 and CLINC150 — depth findings there don't transfer to typed-decisions

**Status: the windowing observation stands; the receptive-field prediction it fed (idea 2) is now
refuted at full GPU scale too** (`results/tarski/lambda/explore_receptive_field.json`, gaps up to 400
tokens, decoys up to 24, all 23 depths 0-22) — see idea 2's updated section below for the numbers.
The batch=1 latency confound (second half of this item) has not been re-checked this pass.

**Evidence.** Confirmed via `AutoConfig.from_pretrained("answerdotai/ModernBERT-base")`:
`global_attn_every_n_layers=3` and **`sliding_window=64`** (one-sided radius — a ~129-token window).
`load_banking77`/`load_clinc` both set `max_len=64` (`tarski/data.py:129,154`). Since no example can
exceed 64 tokens, every `sliding_attention` layer is structurally indistinguishable from
`full_attention` for these two datasets — the local/global split, and hence anything about *how much of
it has run by depth k*, is irrelevant to any result in the banking77/CLINC150 tables. typed-decisions
uses `max_len=512` (`tarski/data.py:190`); measured directly, median length is **238 tokens, 71.75% of
examples exceed 128 tokens**, and 41.75% exceed 256. So typed-decisions is the *only* dataset in the
current results where windowing could matter at all, and any intuition or pattern carried over from
banking77's smooth depth curves to typed-decisions' collapsed one is comparing two different attention
regimes. (See idea 2 for what this predicts, and why the strong version of the prediction turned out to
be false when tested directly.) **Separately, a scope confound rather than a bug:**
`experiments/latency.py` times only batch=1 (`SHORT`/`LONG`, loop at line 78) throughout. That is a fair
micro-benchmark for "cost of the Nth decision on one message," but the relative advantage of a shared
trunk vs. N separate full passes is expected to shrink as batch size grows and MPS/GPU dispatch overhead
amortizes (compare AdapterDrop's per-shared-layer speedup shrinking with batch size, Table 2). If these
latency numbers are used to argue for production viability under concurrent multi-tenant load (where
requests are batched), a batch-size sweep is needed; currently there is none.

### 7. NEW — the "copy the base's own next layers" init default is not clearly better than random init at shallow splits, undermining the truncate-and-fork assumption

**Status: TESTED (found while re-checking `results/tarski/ablation_init.json`, now populated for the
first time this session).** `tarski/branches.py:106-109`'s `BlockBranch` defaults to `init="next"`
(copy the base's own layers `[k, k+d)`); `"random"` re-initializes them instead. The project's own
novelty notes (`research_notes/novelty_check.md`, Claim 5) already flag Zhang et al.'s ICLR 2021 finding
that *re-initializing* BERT's top layers can help few-sample fine-tuning as "counter-evidence... not a
foregone conclusion." The real ablation now bears this out at tarski's own shallow splits: at split 4
(depth 2), **`random` beats `next`** on both datasets tested — banking77 intent 92.04% (random) vs.
91.48% (next); CLINC150 intent 87.47% (random) vs. 85.44% (next), a **2-point gap in random's favor**.
At split 14, `next` (92.69%) edges out `random` (92.43%) on banking77, so the direction is not even
consistent across depths. What IS consistent: `top` init (copy the base's *last* d layers, wherever the
split) is reliably worst everywhere tested (e.g. CLINC150 split 4: `top` 84.24% vs. `next` 85.44% vs.
`random` 87.47%). So "truncate-and-fork" (`next`) is not straightforwardly the safe default the project's
design narrative implies — it beats the clearly-wrong choice (`top`) but is a coin flip against plain
random init at shallow splits on small-to-medium data, exactly where the brief's target users (small
end-user label sets) would be using it. Not deeply investigated further here (would need more seeds and
splits, and this touches `tarski/branches.py`'s documented default, out of scope for `explore/`); flagged
as a finding for whoever owns that file.

---

## (b) Research ideas

### Idea 1 — Prior-shift-corrected OOS gate — **implemented, empirically confirmed**

**Mechanism.** Apply Saerens-Latinne-Decaestecker (2002) EM prior-shift correction to the already-trained,
already-temperature-scaled `oos` branch's test-time softmax outputs: estimate the test class prior from
the classifier's own posteriors via EM (no test labels used), then re-weight posteriors by
`prior_test_est / prior_train` and renormalize. Purely post-hoc — no retraining, no core-file change.

**Novelty check.** Not novel as a technique: Saerens, Latinne & Decaestecker, *"Adjusting the outputs of
a classifier to new a priori probabilities: a simple procedure,"* Neural Computation 14(1), 2002; Lipton,
Wang & Smola, *"Detecting and Correcting for Label Shift with Black Box Predictors"* (BBSE), ICML 2018.
Two recent papers already flag *this exact* CLINC150 OOS prior mismatch and fix it with a tuned
*decision threshold* instead: *"Improved Out-of-Scope Intent Classification with Dual Encoding and
Threshold-based Re-Classification,"* arXiv:2405.19967 (2024) — reports the identical shape of mismatch,
"1.4% to 2.0%" train vs. "17% to 19%" test OOS — and DROID, arXiv:2510.14110 (2025). **Verdict: known
fix for a known problem, not previously wired into tarski.** The contribution here is narrow: it costs
nothing beyond softmax outputs tarski already computes, needs no test labels, and is a fully
label-shift-general alternative to hand-tuning a threshold.

**Result — CONFIRMED at full GPU scale.** `explore/skeptic_oos_priorshift.py` trains one `oos` branch
the normal tarski way and reports accuracy/macro-F1/oos-recall/precision before and after correction.
The queued full run (probe@11, all 15250/3100/5500 CLINC150 rows, 20 epochs) completed:
`results/tarski/lambda/explore_oos_priorshift.json` —
**macro-F1 0.603 → 0.739, oos-F1 0.292 → 0.570, oos-recall 0.173 → 0.553**, matching the oracle-prior
upper bound (macro-F1 0.740, oos-F1 0.566, using the true test prior) almost exactly, and the EM's own
prior estimate (0.769/0.231) landed close to the true prior (0.818/0.182) without ever seeing a test
label. A rougher, 1-epoch CPU-only sanity check on the same full data (done while building the smoke
test, before the real run) previously showed the same direction (macro-F1 0.525→0.706, oos-F1
0.143→0.520) — both checks agree. **Caveat found while building the smoke test:** the correction cannot
manufacture signal that was never learned — at extreme few-shot scale (a handful of true OOS training
rows) the branch's posteriors are degenerate (near-identical for every example) and no prior correction,
oracle included, can fix that; it only helps once the branch has learned *some* usable, if
miscalibrated, separation.

**Wildness: 2.**

### Idea 2 — Receptive-field-aware split depth for long inputs — **implemented, hypothesis partly refuted by direct testing**

**Mechanism (as originally proposed).** ModernBERT-base's global (`full_attention`) layers sit at
0-indexed positions 0, 3, 6, …, 21. At split depth k, only `floor`-ish counts of those 8 have run (2 at
k=6, 4 at k=11, 6 at k=16, 8 at k=22 — exactly the depths `tables.md` reports typed-decisions probes at).
For inputs exceeding the 64-token window radius (item 6: 72% of typed-decisions examples do), the
hypothesis was that two distant JSON fields simply have not had a chance to interact yet at shallow
splits, and that `autosplit.py` should therefore weight "global layers run" rather than raw depth when
choosing a split for long inputs.

**What actually happened when tested.** `explore/skeptic_receptive_field.py` builds a synthetic,
order-sensitive two-marker task designed so it is *provably unsolvable* from a position-invariant,
mean-pooled representation (label = whether the first marker's fixed rank precedes the second's; the
unordered token pair carries zero label information, only which position held which value does — unlike
a naive "do these two tokens match" task, which a bag-of-words probe could solve at depth 0 with no
attention at all and would have tested nothing). Running it (CPU, `--smoke` variants, several gaps up to
400 filler tokens — several window-radii): accuracy is at chance at **depth 0** (0.29-0.54, matching the
~50% base rate — confirming the task design is sound and genuinely needs position-aware mixing), but
jumps to **0.79-1.0 at depth 1 already**, for every gap tested including gap=400, and stays roughly flat
from there to depth 22. The reason: **layer 0 itself is a `full_attention` (global) layer** — "every 3rd
layer is global" starts counting at position 0, not position 3 — so *every* split depth ≥ 1 already
gives every token one full, unwindowed attention pass to every other token in the sequence. No split
depth actually used anywhere in the real experiments (all ≥ 4) is ever short of that. Adding up to 8
decoy fields (distractors with the same code vocabulary but different, irrelevant names, to test
*selective* routing rather than bare reachability) did not change this at the scale tested.
**This refutes the hypothesis in its strong form.** It also strengthens items 4 and 3 above as the more
likely explanations for typed-decisions' probe collapse, by ruling out a competing, architecturally
appealing but wrong explanation.

**Status: TESTED at full GPU scale, hypothesis refuted (both strong and weak forms).**
`results/tarski/lambda/explore_receptive_field.json` ran the full grid: gaps {0, 32, 150, 400} x decoys
{0, 8, 24} x all 23 depths (0-22), 240 train / 80 val rows per condition. At decoys=0 and decoys=8 the
CPU-scale finding holds exactly: accuracy jumps from ~chance at depth 0 to 0.875-1.0 by depth 1 for every
gap, including gap=400, and stays there (or wobbles noisily within that band) through depth 22 — no
depth-dependent trend at all once depth ≥ 1. At decoys=24 (the condition designed to give the *weaker*
"selective routing needs more refinement" hypothesis its best chance): accuracy is lower everywhere
(consistent with the task being genuinely harder with more clutter) but still shows **no monotonic
improvement with depth** — e.g. gap=400,decoys=24 goes chance (0.45) → 0.71-0.73 by depth 1-3, then
*drifts down* to 0.58-0.68 through depth 8-22, never exceeding ~0.73 at any depth including 22. So even
under heavy clutter, more trunk depth does not resolve the task better than a shallow pass already does
— the weak version of the hypothesis is refuted too, at least at this decoy count and this synthetic
task's difficulty. This further strengthens items 3 and 4 above (missing full-fine-tune baseline;
early-stopping bug, now fixed) as the real drivers of typed-decisions' collapse, since receptive field
demonstrably is not one, at any depth actually used.

**Novelty check.** The underlying mechanism (receptive field of stacked windowed attention grows with
depth) is textbook: Beltagy, Peters & Cohan, *"Longformer,"* 2020. Needle-style controlled probes of
usable context by depth are standard in long-context evaluation (e.g. Liu et al., *"Lost in the
Middle,"* TACL 2023/2024). Using this to argue a per-task static split-depth selector should be
receptive-field-aware for long inputs, in a hybrid local/global encoder used for multi-task branch
serving, was not found in `research_notes/novelty_check.md`'s review (its closest relative, Wei et al.
ACL 2022, uses plain BERT with no local/global distinction to speak of). The idea itself turned out not
to hold, which is a legitimate (if negative) empirical contribution: it rules out a plausible,
architecturally-motivated explanation the project might otherwise have chased.

**Wildness: 3** (as originally framed, before testing).

### Idea 3 — Decouple correctness-calibration from soft-label-fidelity-calibration — **implemented, smoke passes, direction confirmed at smoke scale, full run queued**

**Status: IMPLEMENTED.** `explore/skeptic_dual_calibration.py`. No core-file change needed after all:
`tarski/train.py`'s own `fit_temperature` already accepts an optional `soft` argument and already
computes hard-only fits when it's omitted (that's exactly what banking77/CLINC get), so this script just
calls it twice per task with the same validation logits — once as `tarski.train.fit()` does today
(`soft` passed when available → `T_soft`), once forcing `soft=None` (→ `T_hard`) — and reports test
metrics under both, plus an uncalibrated `T=1` floor. Accuracy is identical across all three by
construction (temperature scaling never changes argmax); ECE (vs. hard correctness) and Brier-vs-soft
are the metrics that move.

**Mechanism.** `T_hard` = `fit_temperature(val_logits, y_val)` (what `ece_score` actually measures, and
what banking77/CLINC already get for free by having no soft labels — item 2); `T_soft` =
`fit_temperature(val_logits, y_val, soft_val)` (today's typed-decisions default, matching the teacher's
distribution / Brier score). Report both, so a downstream consumer picks the one matching their use
(abstention thresholds need correctness-calibration; distillation-style consistency needs distributional
fidelity).

**Result so far (smoke, CPU, 3 invoice_processing tasks, real 41-row validation sets, probe@4, tiny
train/test subsets — not the queued full run).** The predicted direction shows up already: mean hard-ECE
under `T_soft` (today's default) = 0.275 vs. `T_hard` = 0.258 — today's default is measurably worse at
the thing ECE reports. Brier-vs-soft did NOT show the expected opposite direction at this toy scale
(T_soft 0.236 vs. T_hard 0.223, i.e. also slightly worse for T_soft here) — plausibly just noise from 3
tasks / tiny data / a very shallow, briefly-trained probe; the queued full run (all 20 tasks, probe@22,
proper training) is needed before trusting the Brier-vs-soft side of the comparison. Command in
`explore/QUEUE_skeptic.txt`.

**Novelty check.** Standard temperature scaling calibrates against hard correctness (Guo et al., ICML
2017). Calibrating under label noise / soft targets has some related literature — Müller, Kornblith &
Hinton, *"When Does Label Smoothing Help?"*, NeurIPS 2019; Lukasik et al., *"Does label smoothing
mitigate label noise?"*, ICML 2020 — but a *dual*-temperature report that explicitly separates the two
objectives for a distilled/soft-label classifier does not appear to be a named, standard practice.
**A modest, mostly-methodological contribution**, not a new algorithm.

**Wildness: 2.**

### Idea 4 — Cheap confirmatory re-rank for autosplit

**Status: SKIPPED per coordinator instruction** — the coordinator is handling this as a separate "H4"
follow-up. Left as originally written below; item 5's now-confirmed real numbers (depth 1-3 picked,
2.2-2.5 point accuracy gap) are exactly the evidence this idea would need to justify itself.

**Mechanism.** Given item 5 (probe curve and blocks accuracy can disagree sharply), run one cheap
low-epoch blocks branch (d=1, 1-2 epochs) at just 2-3 candidate depths — the probe-chosen one, plus one
shallower and one deeper — before committing to the probe's answer. Trades a small, bounded amount of
extra compute (still far less than Wei et al.'s ~10x fine-tuning sweep) for protection against exactly
the failure demonstrated in item 5.

**Novelty check.** Wei et al. (ACL 2022) do a full sweep; `tarski/autosplit.py` already improves on that
with a zero-training probe curve. A middle-ground "cheap 1-epoch confirmatory check over a few candidate
depths" does not appear in the reviewed prior art. Not implemented (would touch `tarski/autosplit.py`,
which is out of scope for `explore/`); described here as a suggested core-code change.

**Wildness: 2.**

### Idea 5 — Attention-pooling probe for structured/long inputs

**Status: SKIPPED per coordinator instruction** — covered by the token-attention pooling already
implemented in `explore/farfield_coarsegrain.py` (a different exploration lens's script). Left as
originally written below for context.

**Mechanism.** Replace `ProbeBranch`'s `mean_pool` (`tarski/branches.py:28-30`, used identically by both
probes and, at the very end, blocks branches) with a tiny single-head learned attention-pooling layer —
still O(hidden) extra parameters, far cheaper than an extra transformer block — so a "cheap" probe can
selectively weight informative tokens (e.g. JSON values, not braces/keys/punctuation) instead of
uniformly averaging everything. This targets the "mean pooling over JSON" half of the brief's collapse
question directly, independent of depth or receptive field (which idea 2 mostly ruled out).

**Novelty check.** Attention pooling for sentence embeddings is well established (Lin et al.,
*"A Structured Self-Attentive Sentence Embedding,"* ICLR 2017; SetFit-style pooled heads). Using it
specifically as a near-zero-cost alternative to mean pooling to close the probe-vs-blocks gap on
structured/JSON inputs in a frozen-trunk branch-serving harness is a small, believable, incremental
contribution, not a new mechanism. Not implemented in this pass (time budget went to ideas 1-2); a
reasonable next script.

**Wildness: 2.**

### Idea 6 — Per-instance depth gating on top of per-task static depth — **implemented, smoke passes, full run queued**

**Status: IMPLEMENTED.** `explore/skeptic_depth_gating.py`. Does not touch `tarski/engine.py`'s serving
path (out of scope); instead validates the accuracy/depth trade-off the mechanism would need in order to
be worth building, using `FeatureCache` at two depths from tarski's own API (no core-file changes).

**Mechanism.** Combine per-task branch specialization (tarski's existing design) with per-*instance*
early exit: train a cheap shallow branch (depth k1) and tarski's own more accurate deep branch
(depth k2 > k1) for the SAME task; at serving time, run the shallow branch first, and only escalate to
depth k2 for instances whose (temperature-calibrated) shallow confidence falls below a threshold. Report
accuracy vs. average trunk depth run, swept over the threshold, against the two STATIC baselines tarski
already reports in `tables.md` ("always k1", "always k2"). A win looks like: near-`k2` accuracy at an
average depth much closer to k1 than k2. The threshold is picked on validation accuracy only; test
accuracy is reported at that picked threshold plus the full sweep for inspection.

**Result so far (smoke, CPU, tiny banking77 subset, probe@2 shallow / blocks@6+1 deep, 300/100/150
rows — not the queued full run).** Mechanically correct end-to-end (static baselines: always-2 = 34.0%,
always-6 = 35.3%; the val-picked threshold reached 36.0% test accuracy at an average depth of 3.12,
"capturing" more than 100% of the small shallow-to-deep gain at this toy scale). The magnitudes are not
meaningful yet — 100+ training examples and 2-4 epochs is far too little for either branch to reach a
stable accuracy, and the shallow-to-deep gap itself is tiny at these toy depths (k=2 vs. k=6). The queued
full run uses depths and a task combination directly comparable to two rows already in `tables.md`
(`probe@4` = 87.9% vs. `blocks@18+1` = 93.1% on banking77 intent — the largest, cleanest depth-accuracy
gap in the existing results) so the gated system's Pareto position can be read directly against numbers
already trusted in this project. Command in `explore/QUEUE_skeptic.txt`.

**Novelty check.** `research_notes/novelty_check.md` (Claim 2) already states plainly: "I found no NLP
paper that fixes a static exit layer per *task* and serves many such tasks from one trunk, other than
Wei et al." — and per-*instance* early exit (DeeBERT, PABEE, F-PABEE) is single-task/single-model, never
combined with per-task heterogeneous branches sharing one trunk. **Combining both axes (per-task AND
per-instance adaptivity) appears to be a genuinely open combination**, not covered by Wei et al. 2022 or
Mainstream 2018 (both apply one fixed depth per task to every instance).

**Wildness: 4.**

---

## Ranked top 3 (revised after the follow-up pass — all three now have real results, not predictions)

1. **Bug 4 — the early-stopping/warmup interaction in `train_branch`.** Now **FIXED and confirmed**:
   typed-decisions mean accuracy went from 51.5-55.0% (erratic, non-monotonic in depth) to 57.6-61.6%
   (monotonic) after `tarski/train.py` gained `min_steps=300`/`min_val=50`/`min_epochs=50%`. This was the
   single highest-leverage finding of the whole pass, and it is done — but it surfaced a second, still-
   open problem: mean ECE got *worse* after the fix (0.085-0.104 → 0.129-0.141), which idea 3
   (`explore/skeptic_dual_calibration.py`, queued) exists to diagnose.
2. **Idea 1 — prior-shift-corrected OOS gate.** **CONFIRMED at full GPU scale**
   (`results/tarski/lambda/explore_oos_priorshift.json`): oos-F1 0.292 → 0.570, macro-F1 0.603 → 0.739,
   matching the oracle-prior upper bound almost exactly without ever seeing a test label. The technique
   is 20+ years old and already applied to this exact CLINC OOS mismatch elsewhere; the contribution is
   wiring it into tarski's branch harness for free, post-hoc. Implemented:
   `explore/skeptic_oos_priorshift.py`.
3. **Bug 5 — autosplit's probe-only selector picks a much worse depth for blocks branches.** **CONFIRMED
   by the actual experiment**, and worse than the hand-estimate that originally flagged it: on the real
   probe curve, the selector picks depth 1-3 for banking77 intent (2.2-2.5 points worse than the true
   best) and depth 1 for *all three* CLINC150 tasks at *every* tolerance tested (2.2+ points worse on
   domain). This is the project's own flagship split-selection contribution failing its own validation
   experiment now that the experiment has actually been run
   (`results/tarski/autosplit_banking77.json`, `autosplit_clinc150.json`). The coordinator is following
   up on a fix (idea 4 / "H4").

Also now resolved: **Idea 2 — receptive-field stress test** was run to full GPU scale
(`results/tarski/lambda/explore_receptive_field.json`, all depths 0-22, gaps to 400, decoys to 24) and
the hypothesis is refuted in both its strong and weak forms — receptive field is not the bottleneck for
typed-decisions at any depth actually used, at any clutter level tested. Combined with bug 4, this closes
off two competing architectural explanations for the original "probe collapse" headline and points
squarely at the training loop (now fixed) and the still-missing same-architecture full-fine-tune baseline
(item 3, resolved for banking77/CLINC150, still open for typed-decisions) as the real story. **New this
pass:** item 7 (copy-vs-random init is not a clear win at shallow splits) is a small but genuine surprise
worth a closer look by whoever owns `tarski/branches.py`.
