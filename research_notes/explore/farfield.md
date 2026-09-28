# Far-field transplants: mechanisms from other fields mapped onto "one frozen trunk, many per-decision branches"

Written 2026-09-27 by the far-field exploration agent. Every idea below was checked against the literature
with 1–2 web searches; the "already done in ML?" verdict is stated plainly. Experiments are sized for the
M6 GPU (MPS) with a CPU `--smoke` mode. Two ideas are implemented in `explore/farfield_syndrome.py` and
`explore/farfield_coarsegrain.py`.

Scoring: wildness 1 (incremental) to 5 (nobody would think of this from inside NLP). The ranking at the end
uses novelty x feasibility x relevance to local-first routing, not wildness.

A finding that is not an idea but matters for the harness: the typed-decisions "probe collapse" in
`results/tarski/sweep_typed.log` looks like a training-loop artefact, not a representation failure. Each
probe stops after 4–5 epochs (`patience=3`) because validation accuracy on 26–31 messages sits at the majority
class during the OneCycle warm-up (pct_start 0.1 of 20 epochs = 2 epochs), so `train_branch` returns the
epoch-1 weights. Idea 2's script recomputes the probe baseline without early stopping, standardised and
full-batch, so the coarse-graining effect is measured against a fair probe. If that baseline alone is far
above 51–55%, the fix belongs in `tarski/train.py` (early stopping on validation NLL with a minimum number
of epochs), not in a new branch type.

---

## 1. Syndrome decoding of decision bundles (coding theory; immunology's danger theory)

**Source mechanism.** In an error-correcting code the receiver never sees the channel noise directly: it
computes the *syndrome*, the pattern of violated parity checks among the received symbols, and decodes to
the nearest valid codeword. Matzinger's danger theory (1994) in immunology says the same thing about the
body: the immune system reacts less to "non-self" than to *inconsistency* signals from tissue (damage
without a matching cause).

**Mapping.** A message gets several decisions from independent branches: intent (150), domain (10),
out-of-scope, urgency, team. The decisions are redundant: domain is a function of intent; team is nearly a
function of category. So a bundle of branch outputs is a noisy received word from a code whose valid
codewords are the consistent bundles. Two things follow. (a) *Decoding*: choose the consistent bundle with
the highest joint branch log-probability, `argmax_i log P_int(i) + log P_dom(dom(i))`, instead of taking each
argmax separately. (b) *Syndrome as an out-of-scope signal*: the amount of inconsistency (e.g. JS
divergence between the domain branch's distribution and the domain distribution implied by the intent
branch) needs no OOS training data at all. In-scope messages land on a codeword; out-of-scope messages fall
between codewords because each branch is forced to pick something, and they pick incompatibly.

**Novelty.** The mechanism exists in vision: Zamir et al., "Robust Learning Through Cross-Task
Consistency", CVPR 2020 (https://openaccess.thecvf.com/content_CVPR_2020/papers/Zamir_Robust_Learning_Through_Cross-Task_Consistency_CVPR_2020_paper.pdf)
define a "consistency energy" over depth/normals/etc. predictions and report AUC 0.99 for OOD detection
with it. Hierarchical OOD: Linderman et al., "Fine-grain Inference on Out-of-Distribution Data with
Hierarchical Classification", CoLLAs 2023 (https://arxiv.org/abs/2209.04493) backs off to coarser labels
when uncertain (different mechanism). Joint domain+intent+OOS modelling: "Out-of-Scope Domain and Intent
Classification through Hierarchical Joint Modeling", IWSDS 2021 (https://arxiv.org/abs/2104.14781) shares
OOS supervision across the two heads but does not use domain/intent disagreement as a score. Error-
correcting output codes (Dietterich & Bakiri, JAIR 1995) build *artificial* redundant binary tasks. Not
found: using the *natural* redundancy among user-requested decisions, produced by independently trained
frozen-trunk branches, for consistent joint decoding and for label-free OOS scoring in text routing. Verdict:
mechanism known in vision, application to decision bundles / OOS new-ish; modest but real.

**Experiment (implemented: `explore/farfield_syndrome.py`).** CLINC150. Trunk pooled states at depths
{4, 8, 12, 16, 22}; linear probes for intent (150, trained on in-scope only), domain (10, in-scope only),
plus the supervised OOS binary probe and the 151-way probe as references. Scores on the 5500-message test
set (1000 OOS): syndrome-JS, syndrome-agreement (1 − Σ_i P_int(i) P_dom(dom(i))), joint-decoded
in-scope score, max-softmax (MSP), kNN distance and Mahalanobis (the immunology reduction, idea 3),
depth-disagreement (idea 4), and rank-average combinations. Metric: AUROC and FPR@95%TPR for OOS,
binary OOS accuracy at a validation-chosen threshold vs the sweep's 85.8–87.6% and the 81.8% always-
in-scope baseline; in-scope intent accuracy with vs without consistent decoding. Expected signal: syndrome
alone below Mahalanobis but above MSP at mid depth; syndrome + MSP above either; joint decoding +0.2–0.5
intent points. Runtime: one trunk pass over 23.9k short messages plus full-batch probes, ~3–6 min on MPS.
**Wildness 3.**

## 2. Structure-following coarse-graining with dendritic field gates (renormalisation group; neuroscience)

**Source mechanism.** Block-spin renormalisation (Kadanoff 1966) replaces groups of microscopic degrees of
freedom by one coarse variable per block and asks which couplings survive. In cortex, task-dependent gain
modulation (and "active dendrites" in Numenta's models) multiplicatively gates which inputs a neuron
listens to, so one population serves many tasks by changing gains rather than weights.

**Mapping.** A typed-decisions message is a JSON state: 240 tokens, most of them keys and punctuation
identical across messages of a workflow. Mean pooling over tokens is a block-spin transform with one block
= the whole message, which washes out the few informative fields. The RG-faithful choice is to make the
blocks follow the data's own structure: pool the trunk's token states per JSON field (key path to depth 2),
giving an F x 768 "coarse state" per message. Each decision then gets a *gain vector* over fields (softmax of
F learned logits, per task: urgency listens to `alert.evidence`, `duplicate` to `invoice.number`), i.e. the
dendritic gate, and a linear read-out. Variants: input-dependent field attention (learned query), and the
un-coarse-grained control (attention over tokens), to separate "structure helps" from "attention helps".

**Novelty.** Cell-level mean pooling of transformer states is in TaBERT (Yin et al., ACL 2020,
https://arxiv.org/abs/2005.08314); hierarchical attention over segments is HAN (Yang et al., NAACL 2016);
per-task static gates over inputs are (IA)^3-like (Liu et al., NeurIPS 2022, https://arxiv.org/abs/2205.05638).
Token pruning for upper layers (PoWER-BERT 2020, LTP 2022) is the compute-side cousin. Not found:
schema-derived pooling of *frozen trunk* states with per-decision static field gates as a KB-sized branch
type for JSON decision models. Verdict: components known; combination and target (probe collapse on JSON
states) new to this setting; primarily diagnostic.

**Experiment (implemented: `explore/farfield_coarsegrain.py`).** typed-decisions, depths {6, 11, 16, 22}.
Heads trained on cached states with soft-label cross-entropy, full batch, no early stopping: `mean`
(fair probe), `fields_gate`, `fields_attn`, `fields_concat`, `token_attn`. Metrics: acc, macro-F1, NLL,
Brier vs soft labels, per task and mean over 20 tasks; majority-class and sweep probe@k (51–55) as
baselines; laya 76.6 as the ceiling. Expected signal: `mean` already well above the sweep's probes (the
artefact), `fields_*` +3–8 points over `mean` on field-local decisions (duplicate, matches_order,
credential_compromise), little on holistic ones (urgency, action). Runtime: one trunk pass over 1600
messages (~20 s on MPS) plus seconds of head training; < 5 min. **Wildness 3.**

## 3. Negative selection for out-of-scope (immunology)

**Source mechanism.** T-cells are generated with random receptors; any that bind self-antigens in the
thymus are deleted (negative selection). The survivors detect anything that is not self, without ever
seeing a pathogen (Forrest et al., 1994; real-valued NSA / V-detector, Ji & Dasgupta 2004).

**Mapping.** Generate random detectors (points with radii) in the pooled trunk space at depth k; delete
every detector that covers an in-scope training message; a test message is out-of-scope if a surviving
detector covers it. No OOS training examples needed, which is the routing reality.

**Novelty.** NSA is widely used in intrusion detection (review: "Negative Selection Algorithm Research and
Applications in the last decade", 2021, https://arxiv.org/abs/2105.06109); no application to LM embeddings
for intent OOS was found. But the honest analysis kills it: in 768 dimensions a detector that survives
negative selection is one whose centre is farther than r from every self point, so "covered by a surviving
detector" reduces to "k-NN distance to the training set exceeds a threshold", which is Sun et al.,
"Out-of-Distribution Detection with Deep Nearest Neighbors", ICML 2022 (https://arxiv.org/abs/2204.06507),
and the Mahalanobis variant is the intent-OOS state of the art (Podolskiy et al., AAAI 2021,
https://arxiv.org/abs/2101.03778). Verdict: already done under another name; the transplant adds nothing.
Kept as a baseline in idea 1's script (kNN and Mahalanobis at each depth). **Wildness 4, novelty 1.**

## 4. Prediction-error across depth as an OOS signal (predictive coding)

**Source mechanism.** In predictive coding, higher cortical areas predict lower-area activity and only the
residual (surprise) propagates upward; large sustained surprise means the generative model does not
fit the input.

**Mapping.** Probes at several depths give a trajectory of decisions over depth. In-scope messages settle
early and stay put; out-of-scope messages keep changing their mind, because no layer's features support a
stable answer. Score = disagreement across depths (JS between adjacent-depth probe outputs, or the mutual
information of the depth ensemble).

**Novelty.** Done in spirit: MOOD (Lin et al., CVPR 2021, https://arxiv.org/abs/2104.14726) exploits early
exits for OOD; BLOOD (Jelenić et al., ICLR 2024, https://arxiv.org/abs/2310.02832) scores text OOD by
between-layer transformation smoothness on Transformers; "Mysteries of the Deep" (2025,
https://arxiv.org/abs/2510.05782) finds mid layers best for OOD. Verdict: already done. Included in idea
1's script as a cheap extra signal because the probes are already there. **Wildness 2.**

## 5. Anytime decision bundles with performance profiles (real-time systems)

**Source mechanism.** Anytime algorithms (Zilberstein, AI Magazine 1996) carry a *performance profile*
(quality vs compute) and a meta-level controller stops when the marginal quality per unit time falls below
the deadline's price; imprecise computation (Liu et al. 1991) returns a usable result at any interruption.

**Mapping.** The autosplit probe curve *is* each decision's performance profile. For a bundle of requested
decisions, the controller runs the trunk layer by layer, emits provisional decisions from shallow probes,
and stops when every requested decision's expected change (learned per task from validation: P(argmax
flips before depth 22 | margin at depth d)) is below a threshold. Urgent messages can be acted on at depth
4 and refined at depth 16 if the deadline allows.

**Novelty.** Per-instance early exit is well covered (DeeBERT, PABEE, F-PABEE 2023 for multi-label; EE-Tuning
2024 for frozen LLMs; one-step-lookahead stopping in "Optimal Depth of Neural Networks", 2025,
https://arxiv.org/abs/2506.16862). What is not written up is the bundle-level rule (stop when *all*
requested decisions have converged) with profiles measured on a frozen trunk. Verdict: mostly done; the
bundle twist is small. Expected signal is also small here: banking77 probes are flat over depth (87.3–88.7),
so the profile has no knee to exploit. **Experiment:** CLINC/banking77, probes at all depths, cascade with a
validation-fitted flip-probability model; plot accuracy vs mean depth against fixed-depth probes and
blocks@k+d; ~5 min on MPS. **Wildness 2.**

## 6. Free-rider depth (public-goods economics)

**Source mechanism.** A public good, once paid for by whoever values it most, is free for everyone else to
use; rational agents consume it at the level provided, not the level they would have paid for.

**Mapping.** In a request bundle the trunk runs to the deepest requested split. Every shallower decision
in the bundle can then read a deeper state for free. Store each task's branch at 2–3 depths (probes cost
KB; blocks 5–10 MB) and, per request, use the deepest branch whose split is already paid for. Zero extra
trunk compute; the only cost is storage and the branch itself.

**Novelty.** Not a published technique; Wei et al. (ACL 2022) run every task at its fixed depth, and ZTW
(NeurIPS 2021, https://arxiv.org/abs/2106.05409) recycles *earlier* exits into later ones, the opposite
direction. Verdict: new but a serving policy, not a method. **Experiment:** post-hoc from
`results/tarski/sweep_clinc150.json`: if intent is served at 16+2, domain moves from 87.9 (6+2) to 88.6 and
oos from 86.1 to 86.9. Small; worth one paragraph in the thesis, not a paper. **Wildness 1.**

## 7. Inverse-variance fusion over depth (control theory, Kalman filtering)

**Source mechanism.** A Kalman filter fuses a sequence of noisy measurements of a latent state with weights
inversely proportional to each measurement's noise; the posterior variance gives a stopping rule.

**Mapping.** Treat each depth's probe logits as a noisy measurement of the final decision; estimate
per-depth noise on validation; fuse depths 1..d with inverse-variance weights. The fused prediction at any
cut-off d dominates the single probe at d, and the posterior variance is an anytime stopping signal.

**Novelty.** Already done: ZTW (2021) ensembles earlier exits; ELMo's scalar mix (2018) is a learned
low-pass filter over depth; Meyoyan & Del Corro, "A BERTology View of LLM Orchestrations" (ACL 2026,
https://arxiv.org/abs/2601.13288) learns layer-selective probes. Verdict: done. **Wildness 2.**

## 8. Alternative splicing of branch layers (molecular genetics)

**Source mechanism.** One gene yields many proteins by splicing different subsets of its exons together;
the exons need not be contiguous.

**Mapping.** A blocks branch at split k copies base layers [k, k+d). Splice instead: copy any d base layers
{a, b} (e.g. layers 8 and 19 on top of h_8), chosen per task by a cheap score (a probe on the output of
each candidate layer applied to h_k). `init=top` in `experiments/ablation_init.py` is one fixed splice;
this searches the space.

**Novelty.** "Transformer Layers as Painters" (Sun et al., AAAI 2025, https://arxiv.org/abs/2407.09298)
shows middle layers can be skipped or reordered with graceful degradation, and non-contiguous layer
pruning fine-tunes back better than contiguous (patent literature, 2025); Aletheia (2026,
https://arxiv.org/abs/2604.15351) selects layers to tune by gradient norm. Not found: choosing which base
layers to *copy into a branch* from a non-contiguous set. Verdict: partially done; cheap to test.
**Experiment:** banking77, blocks@8 with d=2 from {next, top, best-2-by-probe, random-pair}; ~1 min per
branch on MPS, 10 branches = 10 min. Expected: best-by-probe within noise of `next`; a null result is
informative for H5. **Wildness 3.**

## 9. Shared attention, private MLP (cortical columns; task-MoE)

**Source mechanism.** Cortical columns share thalamic input and lateral connectivity but each column's
output cells compute a different function of it; in task-MoE transformers attention is shared and only the
FFN is task-specific.

**Mapping.** In `tarski/fused.py`, N branches at (k, d) each recompute attention over the same trunk state.
Freeze the attention sub-blocks in every branch at the base's weights and train only the MLP, norms and
head. Then the fused module computes Q, K, V and the attention output *once* for all N decisions and only the
MLPs are per-branch: the marginal cost of a decision drops from a full layer to an MLP, and branch files
shrink by ~40%.

**Novelty.** Architecture done: M3ViT (ICLR 2023, https://arxiv.org/abs/2210.14793), UniAV 2024, MoTE 2026
(https://arxiv.org/abs/2608.24763) all share attention and specialise FFNs; FFN-only tuning is a standard
PEFT ablation. Not found: the serving consequence for independently trained frozen-trunk branches (one
attention pass for N branches). Verdict: mechanism done, engineering consequence new to this harness.
**Experiment:** banking77 and CLINC, blocks@11+2 with attention frozen vs full; latency of fused N=20
branches with shared attention vs current; ~10 min. Expected: −0.3 to −0.8 accuracy, ~2x cheaper
branches. **Wildness 2.**

## 10. Sleep-like consolidation of branches (neuroscience)

**Source mechanism.** Hippocampal replay during sleep consolidates many episodic traces into cortical
weights without the original stimuli.

**Mapping.** Many independently trained branches at the same (k, d) are consolidated into one multi-head
branch by replaying cached trunk states (no raw messages needed) and matching each teacher branch's
outputs; new branches join by another consolidation round.

**Novelty.** Done: BAM! (Clark et al., ACL 2019, https://arxiv.org/abs/1907.04829) distils single-task
teachers into a multi-task student; RegMean (Jin et al., ICLR 2023, https://arxiv.org/abs/2212.09849)
merges linear layers in closed form from activation statistics; task arithmetic/TIES are in the brief's
prior-art list. Verdict: done. **Wildness 2.**

## 11. Template cancellation (motor neuroscience, efference copy)

**Source mechanism.** The motor system sends a copy of each command to sensory cortex so self-generated
input is subtracted before perception; only unexpected input is processed.

**Mapping.** A workflow's JSON schema is self-generated input: keys, punctuation and boilerplate are
identical across its messages. Subtract the per-field mean trunk state (over training messages) from each
token before the branch, so the branch spends its capacity on variation, not the template.

**Novelty.** Mean removal is "All-but-the-Top" (Mu & Viswanath, ICLR 2018, https://arxiv.org/abs/1702.01417)
and whitening for sentence embeddings (Su et al. 2021). For a linear probe with a bias the subtraction is
absorbed exactly, so it can only matter for the nonlinear blocks branches. Verdict: known trick, untested
for copied-layer branches. **Experiment:** typed-decisions blocks@8+2 with vs without per-field mean
subtraction (fields from idea 2's span code); ~5 min per config on MPS. **Wildness 3.**

## 12. Population search over branch topology (evolution)

**Source mechanism.** A population of variants competes under a fixed resource budget; niches partition
the resource.

**Mapping.** Evolve (split, depth, splice, pooling) per task under a total-compute budget shared across the
bundle.

**Novelty.** Done: "Learning to Branch" (ICML 2020), TreeMTL (2022), Wei et al.'s validation sweep;
autosplit is already the cheap replacement. Verdict: done; listed for completeness. **Wildness 1.**

---

## Ranked top 3 (novelty x feasibility x relevance to local-first routing)

1. **Syndrome decoding of decision bundles (idea 1).** Novelty medium (vision precedent, no NLP-bundle
   precedent), feasibility high (post-hoc on probe outputs), relevance high: it attacks the weakest result
   (OOS at 85.8–87.6% vs 81.8% trivial) with a signal that needs no OOS labels and that only exists
   *because* the system answers several decisions per message. Implemented: `explore/farfield_syndrome.py`.
2. **Structure-following coarse-graining with field gates (idea 2).** Novelty medium-low, feasibility high,
   relevance high: typed-decisions is the dataset that mirrors the product (JSON states, soft labels) and is
   where the harness currently fails. Also settles whether the probe collapse is an artefact. Implemented:
   `explore/farfield_coarsegrain.py`.
3. **Shared attention, private MLP for fused branches (idea 9).** Novelty low (task-MoE), feasibility high,
   relevance high for the cost thesis (H2): halves the marginal cost of a decision at a measurable accuracy
   price. Not implemented; a 10-minute follow-up on the sweep code.

Honourable mention: alternative splicing (idea 8) as a cheap extension of the H5 ablation. Ideas 3, 4, 7,
10, 12 are already done in ML under other names and should not be claimed.

## Status per idea (updated 2026-09-27, after the lambda runs)

Every idea now has a script with a passing CPU `--smoke` and a full mode; the full-run commands are in
`explore/QUEUE_farfield.txt` (one line per run: name | GPU | est. minutes | command). Shared helpers live in
`explore/farfield_common.py`. The custom training loops were checked against the under-training finding:
`farfield_syndrome.py` (300 full-batch steps) and `farfield_coarsegrain.py` (400 full-batch steps) were never
under-trained, so their lambda results stand and no rerun is queued; every new blocks-training script goes
through `tarski.train.train_branch` with `min_steps=300` (smoke modes pass a smaller `min_steps`).

| # | Idea | Status | Result / testable prediction |
|---|---|---|---|
| 1 | Syndrome decoding | **tested** (lambda, `explore_syndrome.json`) | OOS AUROC: syn_agree 0.919, syn_joint 0.929, msp_int 0.929, syn_js+msp+maha 0.946, supervised stack 0.957; cross-depth ensemble 0.94. Joint decoding +0.1-0.4 in-scope intent points (93.0 -> 93.2 at depth 4). Competitive and label-free, not better than the supervised probe. |
| 2 | Coarse-graining + field gates | **tested** (lambda, `explore_coarsegrain.json`) | fields_gate 71.4 / fields_attn 71.4 vs mean-pool probe 66.1 over all 20 typed tasks at depth 16 (harness probe@22 after the min-steps fix: 65.7); a blocks@k+2 branch reaches ~75. Interpretable gates. |
| 3 | Negative selection | **queued** (`farfield_negsel.py`, 3 min) | Prediction: NSA margin <= 1-NN distance - self radius (triangle inequality, checked in the script), AUROC <= kNN. Smoke already shows the curse of dimensionality: with 4k candidates in 8-, 64- and 768-d, detectors cover 0-20% of test points and the score is at or below chance; OOS messages sit *inside* the in-scope cloud in the top PCA directions (kNN in PCA-8 is 0.52 vs 0.86 full-dim). Full run confirms with 40k candidates / 15k self points. |
| 4 | Prediction error across depth | **queued** (`farfield_predcode.py`, 4 min) | Prediction: ridge R^2 of z_{k+j} from z_k > 0.9, so the residual carries little; residual-only probe loses tens of points, error-norm AUROC < MSP and < kNN. Smoke: R^2 0.91-0.93, residual probe 25% vs 76%, error AUROC 0.75 vs MSP 0.83. Cross-depth disagreement was already measured in the syndrome run (ensemble 0.94). |
| 5 | Anytime decision bundles | **queued** (`farfield_anytime.py`, 8 min, with idea 7) | Prediction: bundle stops around depth 4-8 on flat profiles at no cost; CLINC waits for intent. Smoke (CLINC subset): VOI flip-rule at tau 0.3 stops at mean depth 6.0 with bundle acc 0.900 vs 0.878 at depth 22; MSP>0.5 and patience-1 exits similar. |
| 6 | Free-rider depth | **tested** (post hoc from lambda sweeps, `explore_freerider_smoke.json`; the full command is the same computation) | CLINC (3 seeds, blocks +2): riding to the intent branch's split 16 lifts the bundle mean from 87.0 to 88.1 (+1.1), ride-best<=D +0.6. typed_cs: +4.4 at D=14, but a naive ride to 20 loses 6.0 on urgency, which ride-best<=D avoids. Banking77 is a single task (degenerate). Serving policy, not a method. |
| 7 | Inverse-variance fusion over depth | **queued** (in `farfield_anytime.py`) | Prediction: inverse-NLL fusion dominates the single-depth probe at every cut-off. Smoke: 7/7 depths for all three CLINC tasks; fused@6 0.804 vs single@6 0.762 vs single@22 0.764 (intent). Known mechanism (ZTW 2021); here it costs nothing because the probes exist. |
| 8 | Alternative splicing | **queued** (`farfield_splice.py`, 15 min) | Prediction: `next` within noise of `probe_best2`, 1-2 points over `random_pair`/`top`; `next_reversed` costs little. The probe-curve selector works in smoke (picks layers 10 and 18 for banking77 at split 8). |
| 9 | Shared attention, private MLP | **queued** (`farfield_sharedattn.py`, 12 min) | Prediction: freezing attention costs 0.3-0.8 points; trainable params 5.37M of 10.09M; first-layer attention sharing saves ~24% of a d=2 branch (attention is 48% of a layer's MACs at 64 tokens, analytic). Smoke: SharedAttnFused == FusedBlocks == loop (max diff 0), 1.25x faster than FusedBlocks at N=4 on CPU. Honest limit found while implementing: only the first branch layer's attention can be shared, because residual streams diverge after the first per-branch MLP. |
| 10 | Sleep-like consolidation | **queued** (`farfield_consolidate.py`, CLINC 15 min + typed_cs 5 min) | Prediction: replay consolidation (no labels) within 1 point of independent branches and above weight merging (sibling `merge_branches.py`: CLINC TIES+refit 0.858 vs independent 0.876; typed_cs 0.722 vs 0.776) and joint training; incremental round forgets < 0.5 points. |
| 11 | Template cancellation | **queued** (`farfield_template.py`, 22 min) | Prediction: per-field mean subtraction helps probes (the per-token LayerNorm makes it non-trivial) on field-local decisions and does nothing or hurts for blocks. 20 tasks x {probe, blocks} x {none, global, field} at split 14. |
| 12 | Population search | **queued** (`farfield_evolve.py`, 15 min) | Prediction: evolution prefers shallower genomes than the sweep's 14+2 under a layer price and beats autosplit by < 1 point at 3-5x its training cost; fitness on 26-31 validation rows is noisy (the honest limit). 3 customer_service tasks, pop 6 x 3 generations over (split, depth, init). |

Not testable here: nothing. Ideas 3, 4, 7, 10, 12 are known mechanisms; their scripts exist to put numbers
on the "already done" verdicts rather than to claim novelty.

