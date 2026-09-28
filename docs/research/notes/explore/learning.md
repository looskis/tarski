# Learning dynamics, objectives and data: exploration notes

Lens: why do typed-decisions probes collapse to the majority class, and what in the *objective*, the
*parameterisation*, or the *data pipeline* (not the architecture, which `research_notes/novelty_check.md`
already covers) would fix that or make branches learn faster from fewer labels. Read against
`research_notes/explore/BRIEF.md`; code pointers are to `tarski/branches.py`, `tarski/train.py`,
`tarski/data.py`, and to `laya` (installed locally at
`.venv/lib/python3.12/site-packages/laya`, with a real cached checkpoint at
`~/.cache/huggingface/hub/models--convaiinnovations--laya`, subfolder `typed-decisions` -- a
ModernBERT-large cross-encoder, 28 layers, hidden 1024).

Each idea below: mechanism, closest prior art (1-2 web searches each, title/year/URL), a cheap harness
experiment, and a wildness score (1 = incremental, 5 = likely to not work but worth 30 minutes of GPU
time to find out). Follow-up from the coordinator: every idea (A-K) is now implemented, per the "Status"
section immediately below and the "Implemented scripts" section at the end.

---

## Status (after the coordinator's GPU pass and training-length fix)

The coordinator ran an earlier version of this file's scripts on the GPU and found `tarski.train.
train_branch`'s epoch/patience defaults were badly undertrained for typed-decisions' scale (a few hundred
messages at batch 32 is only ~10 steps/epoch; 8-10 epochs with patience=3-4 stops at 30-50 steps). Most of
the "collapse" the brief originally reported turned out to be that, not an objective or capacity problem:
`tarski.train.train_branch` now forces >=300 optimiser steps and only trusts best-epoch selection with
>=50 validation rows (see its docstring). With that fixed, `blocks@14+2` reaches 78.6% vs a 79.4% full
fine-tune on the customer-service decisions -- branches were never far behind once trained for long enough.
This file's scripts have all been updated to force the same >=300-step schedule in any custom training
loop (idea A/B/C/D's, F's, H's, I's, J's) and to reuse `tarski.train.train_branch` UNCHANGED wherever a
loop doesn't need a custom loss (E, K, and parts of F/H/I), so it inherits the fix automatically.

| Idea | Status | Result so far |
|---|---|---|
| A (direct proper-score loss) | tested (smoke) + queued (full) | Ran once, pre-fix, mixed in with B/C/D below; queued for a clean re-run at the corrected schedule. |
| B (REINFORCE policy-gradient) | tested (smoke) + queued (full) | Same run as A: at the (pre-fix) schedule, `reinforce` was the objective most prone to collapsing onto a single class even in a quick post-fix smoke check -- worth watching in the real run whether that persists at 300+ steps or was itself a step-count artifact. |
| C (bias-only, K+1 params) | tested (smoke) + queued (full) | The coordinator's pre-fix GPU run found bias-only branches at 42% vs. a 50.4% majority baseline, i.e. *worse than doing nothing* -- the first genuinely negative result in this file. Queued for a re-run at 300+ steps to see whether that was under-training or a real capacity ceiling; if it persists post-fix, idea C is disproved as stated (log2(K)-bit capacity is not sufficient here in practice, whatever the theoretical bit-budget argument says). |
| D (class-balanced / logit-adjusted) | tested (smoke) + queued (full) | Same pre-fix run as A-C; queued for re-run. |
| E (active distillation, live teacher) | tested (smoke) + queued (full, scaled up) | First GPU run (2 tasks, 6 rounds) was inconclusive: `active_soft` 38-58% vs `random_soft` ~39-49%, within noise at that sample size. Rerun scaled to all 5 customer-service decisions and 10 rounds (`explore/QUEUE_learning.txt`) for a less noisy comparison. |
| F (self-training, no teacher) | tested (smoke) + queued (full) | Not yet run on GPU. Implemented in `explore/learning_self_training.py`. |
| G (branch merging, TIES) | tested (full, by the coordinator, not this agent) | **Negative.** The coordinator implemented this directly (`explore/merge_branches.py`, not written by this agent) and ran it: TIES-merged-and-refit lags independently trained branches on both benchmarks (CLINC 87.6 independent vs. 85.4-85.8 merged; customer-service 77.6 vs. 72.2), and a jointly trained shared body is worse still (CLINC 84.9, customer-service 56). Short forked stacks are not redundant enough for TIES/merging to pay off the way it does on full fine-tuned LLMs -- the wildness-4 caveat in idea G's writeup below was right to flag this as the likeliest idea to fail. |
| H (route imprinting) | tested (smoke) + queued (full) | Implemented in `explore/learning_route_imprinting.py`; not yet run on GPU. |
| I (probe-difficulty curriculum) | tested (smoke) + queued (full) | Implemented in `explore/learning_curriculum.py`; not yet run on GPU. Given G and the training-length finding, our prior is this shows at most a convergence-speed effect, not a final-accuracy one. |
| J (outlier exposure, cross-task negatives) | tested (smoke) + queued (full) | Implemented in `explore/learning_outlier_exposure.py`; not yet run on GPU. |
| K (soft-label leverage / branch-vs-full-fine-tune curve) | tested (smoke) + queued (full) | Implemented in `explore/learning_label_efficiency.py`, built specifically to reproduce and extend the coordinator's 78.6-vs-79.4 finding across a range of N; not yet run on GPU. |

---

## A. Direct proper-scoring-rule loss instead of soft cross-entropy ("RLCD-lite")

**Mechanism.** Replace `train_branch`'s soft cross-entropy, `-(soft * log_softmax(z)).sum(-1)`, with
`-proper_reward(softmax(z), soft, qtype, mask)`, using laya's own `laya.common.proper_reward`: a strictly
proper score combining the log score, a spherical score, and (for ordinal "score"-type questions like
`urgency`/`severity`) a ranked-probability-score term that penalises being off by more levels more than
being off by one. This is fully differentiable -- no sampling -- so it is a drop-in loss swap, but it
shapes gradients differently from log-loss alone: the spherical term is bounded and scale-sensitive in a
way that log-loss is not, and the RPS term gives ordinal tasks a smoother incentive to move away from the
majority bucket one step at a time instead of jumping straight to right-vs-wrong.

**Novelty.** Proper scoring rules as *training* losses for classifiers are old (Brier, log-loss are both
proper; see "From Classification Accuracy to Proper Scoring Rules", JMLR 2024,
https://www.jmlr.org/papers/volume24/23-0106/23-0106.pdf, for a survey of why propriety alone doesn't
guarantee a good training objective). What's specific here -- using laya's exact composite reward
(log + spherical + RPS, with the RPS branch gated by question type) as a direct loss for a *tiny
frozen-trunk probe*, and measuring whether it reduces majority-class collapse relative to plain soft-CE
on the same features -- is not something we found prior work on. Closest: laya's own training presumably
already uses this reward (that's where the function lives), but applied to its own full ModernBERT-large
cross-encoder, not to a tiny branch reading a frozen, different trunk's intermediate layer.

**Cheap experiment.** `typed-decisions`, tasks that the brief reports as collapsed (e.g.
`agent_trace_observability.action`, macro-F1 0.13 in `results/tarski/sweep_typed.json`). Baseline: current
`ce` objective. Metric: accuracy, macro-F1, Brier-vs-soft, and two collapse diagnostics (fraction of test
predictions equal to the majority class; fraction of distinct labels predicted). Expected signal: `proper`
should reduce majority-fraction and raise macro-F1 versus `ce` at equal or better Brier, most visibly on
the ordinal ("score"-type) tasks where the RPS term is active. Runtime: seconds per task on a probe; full
sweep over ~8 tasks x 2 branch sizes in well under 30 min on GPU. **Implemented**, see
`explore/learning_proper_score_objectives.py` (`--objectives proper`).

**Wildness: 2** (mechanically simple, the open question is purely empirical).

---

## B. REINFORCE policy-gradient on proper scores, as distinct from A

**Mechanism.** Same reward as A, but used the way "policy-gradient-on-proper-scores" literally reads:
sample a hard decision `a ~ Categorical(softmax(z))` from the branch's own output, score that one-hot
report against the soft target with `proper_reward`, and update with the REINFORCE estimator
`-(reward - baseline) * log pi(a)` -- gradients flow only through the sampled action's log-probability,
never through the reward path itself. This is a strictly stochastic, higher-variance estimator of
(roughly) the same objective A computes exactly; the only reason to prefer it is if the extra noise it
injects acts as an implicit regulariser against the exact majority-class optimum that closed-form soft-CE
gradients walk straight into on a near-uniform target distribution.

**Novelty.** "Proper Scoring Rules for Agentic Uncertainty Quantification" (2026,
https://arxiv.org/pdf/2605.24756) makes the point explicitly: using a strictly proper scoring rule as a
direct calibration loss and using it as an RL/policy-optimisation reward are related but *not* the same
object, because propriety's guarantee is stated under a fixed evaluation law, not under a policy that is
itself being optimised by sampling from it. We could not find an empirical A-vs-B comparison of the two
readings on the same small classification head -- everyone we found picks one and uses it. TinyLoRA
("Learning to Reason in 13 Parameters", 2026,
https://www.alphaxiv.org/abs/2602.04118) is the closest spiritual relative: it argues RL succeeds with
1000x fewer trainable parameters than SFT specifically *because* its reward signal is sparser and cleaner
than a dense per-token loss, which is the same hypothesis being tested here at classification-head scale
instead of full-model LoRA scale.

**Cheap experiment.** Identical harness to A, `--objectives reinforce` vs `--objectives proper` vs
`--objectives ce`, same tasks and branch sizes. Expected signal: genuinely uncertain going in -- either
`reinforce` beats `proper` on macro-F1/collapse (support for the "sampling regularises" hypothesis) or it
just adds variance for no benefit (support for "direct is better when it's available", which would itself
be a useful negative result given how often people reach for RL for propriety's sake alone). Runtime:
same order as A, a few extra seconds per run for sampling. **Implemented**, see
`explore/learning_proper_score_objectives.py` (`--objectives reinforce`).

**Wildness: 3** (the mechanism is standard REINFORCE; the outcome on a routing head this small is
genuinely unclear).

---

## C. Bias-only "TinyLoRA-shaped" branches for routing (K+1 trainable parameters)

**Mechanism.** A branch with a *fixed* random projection `hidden -> K` (orthogonal-ish init, never
trained) after a parameter-free LayerNorm (`elementwise_affine=False`, i.e. it standardises but adds no
weights), plus a trainable per-class bias vector and a single trainable scale: `K+1` parameters total,
independent of hidden size. The brief's framing -- "a routing label carries only log2(K) bits" -- suggests
a full `ProbeBranch` (`(hidden+1)*K` parameters, thousands for a 4-8 way decision) may have far more
capacity than the label carries information, and that capacity is exactly what a near-uniform soft target
can turn into overfit majority-class shortcuts. A `K+1`-parameter head structurally cannot express such a
shortcut beyond "shift class c's logit up or down", which may be a feature here.

**Novelty.** Random-projection + tiny trainable vector is TinyLoRA's mechanism exactly (2026,
https://www.emergentmind.com/papers/2602.04118: "a single, tiny, low-dimensional trainable vector...
projected through a fixed, non-trainable random tensor, with an aggressive parameter-sharing strategy");
it was demonstrated for math-reasoning LoRA adapters on an 8B generative model trained with RL, not for a
classification head on a frozen encoder. Weight-imprinting-style tiny heads exist in few-shot vision
(Qi, Brown & Lowe, CVPR 2018, https://openaccess.thecvf.com/content_cvpr_2018/html/Qi_Low-Shot_Learning_With_CVPR_2018_paper.html)
but those still fit a full `hidden`-dim weight per class from few examples; freezing the projection itself
and training only a bias is a further reduction we did not find applied to text routing/typed-decision
heads.

**Cheap experiment.** Same harness as A/B, branch type `bias` vs `probe`, crossed with all five objectives
(`ce`, `cb_ce`, `logit_adj`, `proper`, `reinforce`). Metric: accuracy/macro-F1/collapse fraction *and*
parameter count, so the plot is accuracy-vs-params, not just accuracy. Expected signal: `bias` branches
should be worse in absolute accuracy (they cannot learn a good representation, only re-scale one) but the
interesting result is whether they collapse *less* (lower majority-fraction) per parameter than `probe`
branches trained on the same near-uniform soft targets -- i.e. whether less capacity is a partial fix
for this specific failure mode, not just a compression trick. Runtime: trivially fast (a `K+1`-parameter
head trains in well under a second per task). **Implemented**, see `explore/learning_proper_score_objectives.py`
(`--branch-types bias`, `BiasOnlyBranch`).

**Wildness: 3** (plausible mechanism, but "less capacity fixes collapse" could just as easily make
accuracy worse with no collapse benefit).

---

## D. Class-balanced and logit-adjusted soft-label losses

**Mechanism.** Two established imbalance fixes, adapted to soft (not hard) targets: (1) class-balanced
re-weighting (Cui, Jia, Lin, Song & Belongie, CVPR 2019,
https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)
weights each training example's soft-CE term by `1/effective_number(y)`, where the effective number uses
the *soft*-label mass per class (`soft_train.sum(0)`) rather than a hard count, since these targets are
themselves distributions; (2) logit adjustment (Menon, Jayasumana, Rawat, Jain, Veit & Kumar, ICLR 2021,
https://arxiv.org/abs/2007.07314) adds `tau*log(prior)` to the *training* logits only (evaluation logits
stay unadjusted), which stops soft-CE from rewarding the model for defaulting to the frequent class.

**Novelty.** Both techniques are established for hard-labelled long-tailed classification. Applying them
to *soft* teacher-distilled targets, and specifically to the "near-uniform soft label, imbalanced hard
label" combination the brief describes for typed-decisions, is a narrower gap -- most long-tail work
assumes the target is one-hot. This is the least novel idea in this file (wildness reflects that), but it
is the cheapest, most standard thing to try first, and the harness needed to test it is identical to A-C.

**Cheap experiment.** Same as A-C, `--objectives cb_ce,logit_adj`. Expected signal: should reduce
majority-fraction and raise macro-F1 relative to `ce`, likely by more than the proper-scoring objectives
on tasks where the failure is really "imbalance", and by less on tasks where the failure is really
"near-uniform target has almost no signal, no re-weighting fixes that". Comparing where each objective
wins is itself informative about *which* of the two stated causes (near-uniform soft labels vs. label
imbalance) actually explains the collapse task-by-task. **Implemented**, see
`explore/learning_proper_score_objectives.py` (`--objectives cb_ce,logit_adj`).

**Wildness: 1** (both techniques are standard; the soft-label adaptation is a small, low-risk change).

---

## E. Active distillation from a live local teacher: label-efficiency curves

**Mechanism.** `laya` is installed and its `typed-decisions` checkpoint is cached locally
(a real ModernBERT-large cross-encoder). It can be loaded as a live oracle during branch *training only*
(never at serving time) via `laya.load("convaiinnovations/laya", subfolder="typed-decisions")` and
queried with the exact question schema each typed-decisions row carries. This turns "how many labels does
a user need" into an active-learning label-BUDGET problem: seed a branch with a handful of teacher-queried
messages, then each round either (a) query the teacher on random unlabelled pool messages, or (b) query it
on the messages the *current* branch is most uncertain about (highest predictive entropy) -- uncertainty
sampling, but the "labels" being budgeted are teacher queries, not human ones. A third arm reuses the
random draw's message identities but substitutes the dataset's already-known gold one-hot label instead of
a teacher query, isolating how much of the benefit is "soft label" vs. "which messages got labelled".

**Novelty.** Every individual piece is established: uncertainty sampling for active learning (Lewis &
Gale, SIGIR 1994); knowledge distillation with soft labels (Hinton et al., 2015); a statistical argument
for why soft labels reduce sample complexity relative to hard ones (Menon et al., "A Statistical
Perspective on Distillation", ICML 2021, https://proceedings.mlr.press/v139/menon21a/menon21a.pdf: "the
teacher benefits student training by providing a regularization gain and reducing the sampling
complexity of hard labels"); "Diversity Enhanced Active Learning with Strictly Proper Scoring Rules" (Wu
et al., 2021, https://ar5iv.labs.arxiv.org/html/2110.14171) uses proper scores to choose which examples a
*human* should label. What we could not find is the combination specific to this system: an end user's
*own* frozen-trunk branch driving which of *their* unlabelled messages get spent as teacher queries against
a locally-resident question-in-input model, to minimise total teacher compute (not human labelling cost)
for reaching a target accuracy -- i.e. active learning as a distillation-budget optimiser rather than a
human-labelling-budget optimiser, in a setting where the "oracle" is a heavier local model rather than a
person or an API.

**Cheap experiment.** `typed-decisions`, `agent_trace_observability.action` and `invoice_processing.urgency`
(both flagged as collapsed). Baseline: teacher's own test accuracy (the ceiling), and a `random_hard`
control at equal label count. Metric: accuracy/macro-F1/Brier at each cumulative label count, for
`active_soft` vs `random_soft` vs `random_hard`. Expected signal: `random_soft` should beat `random_hard`
at equal label count (soft labels are worth more per example); `active_soft` should reach a given accuracy
with fewer teacher queries than `random_soft`, or should not -- either result is a real, quotable
label-efficiency curve for this exact system, something the brief explicitly asks for ("how many
labelled Slack messages a user needs"). Runtime: one trunk pass over ~250+40+120 messages, a handful of
batched teacher forward passes (tens of messages each), and ~7 rounds x 3 methods of probe training (each
sub-second); comfortably under 30 min on GPU, and the teacher (ModernBERT-large) is the only new compute
cost. **Implemented**, see `explore/learning_active_distillation.py`.

**Wildness: 2** (every piece is proven; the risk is only whether the active-vs-random gap is large enough
to matter at these label counts).

---

## F. Self-training on unlabelled traffic with dropout-consistency + proper-score agreement

**Mechanism.** For messages with no label at all (not even from a teacher), enforce agreement between two
stochastic forward passes of the *same* branch (dropout on, different masks) using `proper_reward` as the
agreement score instead of a symmetric KL/consistency loss: treat one pass's output as the "target" for
the other (stop-gradient), alternating which one is target vs. prediction, and only include a message in a
batch once its two passes agree above a confidence threshold (Noisy Student-style self-training, Xie et
al., 2020, https://arxiv.org/abs/1911.04252; FixMatch-style confidence gating, Sohn et al., 2020). This
would let a branch bootstrap from raw unlabelled Slack/ticket traffic beyond whatever teacher-query or
human-label budget it has, which matters because typed-decisions' ~1,080 messages are already a small
per-task sample once split across 4 workflows.

**Novelty.** Self-training and dropout consistency are both old; using a strictly proper score as the
consistency measure between two stochastic views of a *frozen-trunk branch* (rather than the usual
symmetric cross-entropy/KL, or the confidence threshold on raw softmax that FixMatch uses) is a small,
plausibly-useful variant we did not find written up, but it is close enough to existing semi-supervised
recipes that we rank it as the least architecturally novel of the ideas here beyond D.

**Cheap experiment.** Pool = typed-decisions train messages with their labels hidden entirely (no teacher,
no gold). Baseline: supervised-only branch on the labelled fraction. Metric: test accuracy/macro-F1 as a
function of how much of the unlabelled pool is admitted (confidence threshold sweep). Expected signal:
modest accuracy gains, likely smaller than E's teacher-distillation gains, since there is no external
signal here at all -- the interesting comparison is E vs. F at equal "extra compute" budget: is it better
to spend those cycles calling the real teacher, or self-training with no teacher? Runtime: a few dropout
forward passes per unlabelled batch, otherwise identical cost to standard branch training; well under the
budget. **Implemented** in `explore/learning_self_training.py` (smoke-tested, passes in ~6s; queued for
the full run in `explore/QUEUE_learning.txt`). The script also reports, purely as a diagnostic never used
for training (this is a benchmark with real gold labels), how often each round's admitted pseudo-labels
actually agree with the hidden gold label -- i.e. whether the agreement-based admission rule is
confidently right or confidently fooling itself.

**Wildness: 2**.

---

## G. Branch merging via task arithmetic / TIES: N branches to one shared body + N heads

**Mechanism.** Every `BlockBranch` at a given `(split, depth, init="next")` starts as an *identical* copy
of the trunk's own layers `[split, split+depth)` (`tarski/branches.py: BlockBranch.__init__`); only
training diverges them. That means each trained branch's weights minus that shared initialisation is a
"task vector" (Ilharco et al., ICLR 2023, https://arxiv.org/abs/2212.04089) in the *exact same* weight
space, not an approximation. Compute task vectors for N branches sharing a split/depth, merge them with
TIES (Yadav et al., NeurIPS 2023, https://arxiv.org/abs/2306.01708: reset small-magnitude deltas, resolve
sign conflicts, average what's left) into ONE shared body, and keep only the small linear output head
per task un-merged (it must stay separate; heads for different label sets aren't even the same shape).
If the merged body's shared layers still support every task's separate head at close to un-merged
accuracy, N decisions costing N x depth extra layers become 1 x depth extra layers plus N cheap linear
heads -- the efficiency win the brief describes directly ("cut N x depth cost to 1 x depth").

**Novelty.** Model merging is a hot area, but nearly all of it merges independent fine-tunes of the *whole*
network from one shared checkpoint (task arithmetic, TIES, model soups: Wortsman et al., ICML 2022) or
merges LoRA adapters. "Train Separately, Merge Together: Modular Post-Training with Mixture-of-Experts"
(2026, https://arxiv.org/html/2604.18473v1) and Branch-Train-Merge-family work (branches trained
independently, merged by weight averaging) are the closest systems-level analogues, but there every
"branch" is a full copy of the model, not a short forked stack of `depth` layers starting partway through
a much deeper frozen trunk and stopping early (truncate-and-fork, not full-depth fine-tune). We found no
merging work specifically on this shape: several tasks' fine-tuned copies of the SAME short slice of a
frozen encoder's own layers, at a SHARED depth, truncated above `split+depth`. Whether TIES-style merging
even works when the "model" being merged is only 1-3 transformer blocks (much less redundant/overparameterised
than a full LLM, where these methods were validated) is an open, cheap-to-answer question.

**Cheap experiment.** Construct k>=4 same-shape pseudo-tasks that share a frozen trunk and split/depth --
e.g. partition Banking77's 77 intents into k label-disjoint groups and train one `BlockBranch` per group
(`split=8, depth=2`) as in the existing sweep, or use k of typed-decisions' 2-or-4-way tasks at a common
split/depth. Baseline: N separately-trained branches (today's system) and mean-only weight averaging
(no TIES trimming). Metric: per-task test accuracy of (separate) vs (mean-merged body + own head) vs
(TIES-merged body + own head), plus latency/param count. Expected signal: TIES-merged should sit between
"separate" (upper bound) and "mean-merged" (likely worse, given no interference resolution); if it's
close to "separate" for k=4-8, that is a genuine systems result for local-first serving. Runtime: k branch
trainings (already cheap, minutes) plus the merge itself (a few tensor ops, seconds); comfortably under
30 min.

**Result: tested, negative.** Originally scoped out of this agent's top-1-2 build and ranked #2 below on
paper; the coordinator implemented it directly (`explore/merge_branches.py` -- not written by this agent)
and ran the exact ablation this section proposed. It failed: TIES-merged-and-refit lags independently
trained branches on both benchmarks tried (CLINC150 87.6% independent vs. 85.4-85.8% merged;
customer-service 77.6% independent vs. 72.2% merged), and a jointly trained shared body (no merging at
all, just train one body across tasks from the start) is worse still (CLINC 84.9%, customer-service 56%).
Read plainly: truncate-and-fork branches (1-3 layers) are apparently not redundant/overparameterised
enough for TIES's interference-resolution to have much to work with, unlike the full fine-tuned LLMs TIES
was validated on -- and the branches don't even prefer to share a body when trained jointly from scratch,
which rules out "TIES specifically is the wrong merge operator" as the sole explanation. This is a real,
useful negative result for local-first serving: the N-branches-to-1-shared-body compression this section
hoped for does not fall out of this architecture for free, and per-task branches earn their separate
weights.

**Wildness: 4** (correctly flagged pre-hoc as the likeliest idea in this file to fail outright, and it
did).

---

## H. Weight-imprinted route addition without forgetting

**Mechanism.** Adding a new label (a new team, a new category) to an already-trained branch today means
retraining its whole output head from scratch, risking regressing the existing routes. Instead, imprint
the new class's row directly from data (Qi, Brown & Lowe, CVPR 2018, weight imprinting: set the new row of
the output layer to the (normalised, appropriately scaled) mean pooled+normed feature vector of a handful
of examples of the new class), freeze the old rows, and only fine-tune the new row (plus perhaps a rank-1
correction shared across all rows) on a small mixed replay of old + new examples. Measure retention of the
old routes' accuracy immediately after imprinting and after a short fine-tune.

**Novelty.** Weight imprinting is established for few-shot class-incremental vision classifiers, including
follow-up work specifically on its interaction with feature-space geometry (e.g. "Robust Weight
Imprinting: Insights from Neural Collapse and Proxy-Based Aggregation", 2025,
https://arxiv.org/pdf/2503.14572). Applying it to a *frozen-trunk decision branch* used for message
routing, and framing the metric explicitly as "does the branch forget its old routes" for a
production-style incremental-labelling workflow, is a reasonable but incremental combination -- the
underlying technique is not new, only the setting.

**Cheap experiment.** CLINC150's `domain` task (11 labels including oos): hide one domain at branch-training
time, then imprint it in after the fact from k=5/10/20 examples, freezing the other rows. Baseline: full
retrain of the head on all 11 domains at once (the ceiling) and naive fine-tune-everything (the forgetting
baseline). Metric: accuracy on the new domain vs. accuracy drop on the other 10, at each k. Expected
signal: imprinting should get the new domain to reasonable accuracy immediately (zero-shot from a few
examples) with ~zero forgetting, worse new-domain accuracy than a full retrain, but much better
old-domain retention than naive fine-tuning. Runtime: trivial (a handful of forward passes to compute the
imprinted row, then a short fine-tune); well under budget. **Implemented** in
`explore/learning_route_imprinting.py` (smoke-tested, ~3s; queued for the full run). It also freezes the
shared LayerNorm during the surgical fine-tune (not just the other output rows), and adds a from-scratch
ceiling on all 11 domains for reference.

**Wildness: 2**.

---

## I. Probe-difficulty curriculum: order branch training by the autosplit layer-wise margin

**Mechanism.** `tarski/autosplit.py` already computes, for free, a linear-probe validation-accuracy curve
per task across every depth. The same machinery, run per-*example* rather than per-task (train one
early-depth linear probe, then look at each training example's margin: top-1 minus top-2 probability),
gives a natural difficulty score with no extra labels beyond what training already uses. Order branch
training batches easy (large margin at a shallow depth) to hard (still ambiguous at the branch's own
split depth), on the hypothesis that a near-uniform soft target's gradient is especially unstable early in
training and benefits from starting on the examples where there is a clear signal at all.

**Novelty.** Curriculum learning by difficulty is decades old (Bengio et al., 2009). Using a *frozen
trunk's own shallow-depth linear-probe margin* as the difficulty score for training a *later, deeper*
branch on the same trunk is the specific combination we did not find: most curriculum-for-transformers
work either uses external difficulty proxies (sentence length, perplexity) or difficulty from the same
model being trained (self-paced learning), not a cheaper, shallower probe on the same frozen backbone the
final branch will also read. A 2025 search turned up curriculum work that combines progressive layer
addition with curriculum ordering ("Curriculum-Guided Layer Scaling for Language Model Pretraining", 2025,
https://arxiv.org/pdf/2506.11389) and general findings that transferring shallow layers is easier than
deep ones, but nothing using probe accuracy itself as the per-example ordering signal for downstream
branch training.

**Cheap experiment.** typed-decisions collapsed tasks; baseline: random-order (current) batching. Metric:
validation accuracy after each epoch (does curriculum converge faster / to a better optimum, or does it
just reach the same majority-class collapse sooner). Expected signal: likely a wash or small win on
convergence *speed* rather than final accuracy, since the underlying capacity/objective issue (A-D) is
probably the dominant effect -- but cheap enough (one extra shallow-probe pass, already almost free) to be
worth ruling in or out. Runtime: one extra linear-probe-by-depth pass (seconds, already how
`autosplit.layer_curves` works) plus otherwise-identical branch training. **Implemented** in
`explore/learning_curriculum.py` (smoke-tested, ~5s; queued for the full run). Given the training-length
finding above, the expected signal is now sharpened to almost certainly a convergence-speed effect, not a
final-accuracy one; the script reports an explicit "epoch to reach 90% of that run's own best validation
accuracy" for both curriculum and random order so that is directly comparable.

**Wildness: 2**.

---

## J. Outlier-exposure-style cross-task negatives for majority-class / out-of-scope collapse

**Mechanism.** CLINC150's out-of-scope binary task is weak (85-87% vs. 81.8% for always answering
in-scope) and typed-decisions tasks collapse toward one class; both are, in a sense, "the model doesn't
know what it doesn't know" failures. Outlier Exposure (Hendrycks, Mazeika & Dietterich, ICLR 2019,
https://arxiv.org/abs/1812.04606: train against an auxiliary dataset of *known* outliers, pushed toward
uniform output, so the classifier learns better features for "this doesn't look like any in-scope class"
rather than just thresholding max-softmax) has an almost free auxiliary set here: since tarski already
runs one trunk pass across many tasks' messages, OTHER tasks' or OTHER workflows' messages are natural
"known outliers" for a given task's decision (a `security_incidents` message is definitely not a
`customer_service.urgency` decision the model has ever seen a real label for) -- add a small
uniform-target term on those, batched for free alongside the task's own labelled batch.

**Novelty.** Outlier exposure is established for OOD/OOS detection with a dedicated auxiliary corpus.
Using *other tasks in the same multi-task local-first trunk* as that auxiliary corpus -- something already
sitting in the training run, requiring no extra data collection -- for a decision-routing system that
serves many tasks from one trunk by construction, is the specific angle we did not find prior work on
(outlier exposure papers assume the auxiliary set is deliberately gathered, not "whatever other decisions
this deployment happens to have").

**Cheap experiment.** CLINC150 `oos` binary task and typed-decisions' worst-collapsing tasks. Baseline:
plain soft-CE (today). Treatment: add a uniform-target term on a batch of other-task/other-workflow
messages each step, weighted by a small `lambda`. Metric: oos accuracy/F1 (CLINC) and majority-fraction /
macro-F1 (typed-decisions). Expected signal: should help CLINC oos more directly (that's exactly outlier
exposure's designed use case); less clear for typed-decisions collapse, where the failure isn't really
"looks in-distribution but isn't" so much as "the objective doesn't push away from the majority class" --
worth running mainly on CLINC oos, with typed-decisions as a stretch check. Runtime: negligible extra
compute (reuses batches already being cached for other tasks); well under budget. **Implemented** in
`explore/learning_outlier_exposure.py`, scoped to the CLINC oos use case as planned (Banking77 messages
as the free auxiliary "known outlier" corpus; smoke-tested, ~4s; queued for the full run across several
lambda values including lambda=0 as the exact current-objective baseline).

**Wildness: 3** (well-matched to CLINC oos, speculative for typed-decisions -- not tested there, see the
script's docstring for why).

---

## K. Soft-label bit-value / sample-complexity leverage curves

**Mechanism.** A direct, quantitative version of "how many bits is a soft label worth": for a fixed branch
and task, plot test accuracy against N for both hard-labelled and teacher-soft-labelled training sets (no
active learning, just matched random draws at increasing N, i.e. the `random_soft` vs `random_hard` arms
of idea E without the active-learning axis), then read off, for a target accuracy, the ratio of hard-label
N to soft-label N needed to reach it -- literally quoting "one soft label is worth ~X hard labels for this
task" as a leverage multiplier.

**Novelty.** The qualitative claim (soft labels reduce sample complexity) is established (Menon et al.,
ICML 2021, cited under E). We did not find anyone reporting a leverage *multiplier* curve specific to a
routing/typed-decision branch, i.e. turning the theory into the exact number a user planning their
labelling effort would want. This is really a specific, sharper readout of idea E's data rather than a
separate mechanism -- listed separately because it is a distinct, very cheap deliverable (a curve, not a
training change) that directly answers the brief's explicit "label-efficiency curves" ask, and because it
does not require the active-learning machinery at all (so it is a fallback if E's active-vs-random gap
turns out to be noise).

**Cheap experiment.** Originally scoped as pure post-processing over E's `random_soft`/`random_hard`
curves (find the N at which each crosses a matched accuracy and report the ratio) -- but the coordinator's
follow-up specifically asked for "a label-efficiency curve for branches vs full fine-tune", which E does
not compute (E only trains cheap `ProbeBranch`es, never a full fine-tune). **Implemented** instead as its
own script, `explore/learning_label_efficiency.py`: at each of several N (20 up to the full ~270-message
pool), it trains `BlockBranch(split=14, depth=2)` (the coordinator's own `blocks@14+2`, 78.6% at full N)
on both hard and soft labels, reports the soft-vs-hard leverage ratio described above, AND (sparsely, at 2
of those N, since a full fine-tune is real per-step compute unlike a cached-feature probe) trains
`BlockBranch(split=0, depth=trunk.n_layers)` -- this codebase's own definition of "full fine-tune" per
`experiments/sweep.py`'s docstring -- at the same N, so the branch-vs-full-fine-tune gap (79.4% at full N)
can be read directly off the same curve at smaller N too. Smoke-tested (~21-35s; the full-fine-tune code
path is real 22-layer backprop, not a cache lookup, so its `min_steps` is overridden down for the smoke
run only -- see the script's comments). Queued for the full run.

**Wildness: 1**.

---

## Ranked top 3 (novelty x feasibility x relevance to local-first routing) -- as originally assessed

Kept as originally written, for the record, alongside what the first GPU pass actually found (see
"Status" above and each idea's updated result). G's empirical failure is the single biggest update to
this ranking: it looked like the best structural bet on paper and turned out not to work at all, which is
itself the most informative single result in this file so far -- a reminder that the "shared
initialisation implies mergeable" argument does not obviously survive contact with branches this short.

1. **E - Active distillation from a live local teacher (label-efficiency curves).** Directly answers the
   brief's own question ("how many labelled Slack messages does a user need"), uses a genuinely local
   asset (the cached `laya` checkpoint) nobody else's harness has sitting next to a frozen-trunk branch
   trainer, confirmed feasible end-to-end in the smoke test. Novelty is in the combination (live
   model-as-oracle driving a distillation label budget) rather than any single piece. First GPU run (2
   tasks) was inconclusive; scaled up and re-queued (see Status).
2. **G - Branch merging via task arithmetic/TIES (N branches to 1 shared body).** Looked like the largest
   potential win for local-first serving cost on paper. The coordinator built and ran it:
   **negative** -- TIES-merged bodies lag independent branches by 2-6 points on both benchmarks tried, and
   a jointly trained shared body is worse still. Demoted from #2-on-paper to "tested and disproved as
   stated"; see idea G above for the numbers.
3. **B - REINFORCE policy-gradient vs. direct proper-score loss.** Most precisely targeted at the brief's
   flagged failure (collapse under near-uniform soft labels), backed by a 2026 paper's explicit claim that
   the two readings of a proper scoring rule are not interchangeable -- and yet, as far as we found, nobody
   has actually run that A/B on a small classification head. Complicated by the training-length finding:
   the "collapse" this idea was aimed at fixing turned out to be mostly under-training, so the real
   question the full run now answers is narrower (does REINFORCE's extra variance help or hurt once both
   objectives get a fair, long-enough schedule) rather than "does it fix collapse".

Given G's result, if this ranking were redone today idea **K** (soft-label leverage / branch-vs-full-fine-tune
curves) would likely take its place at #2: it is the most directly useful to the user (a concrete
labelling-effort number, tied to the coordinator's own 78.6-vs-79.4 finding) and the cheapest to trust,
since it reuses `train_branch` unchanged rather than a custom loop.

---

## Implemented scripts

All seven scripts import only from `tarski` and `laya`, add a `sys.path` shim so they run as plain
scripts from the repo root (no `PYTHONPATH=.` needed), and have been smoke-tested on this CPU-only Mac
(all pass, all well under the ~2-minute budget individually). Every custom training loop (i.e. every
script that cannot just call `tarski.train.train_branch` unchanged) now reproduces its min-steps /
half-schedule-before-early-stop / last-epoch-if-tiny-val logic exactly, so none of them repeats the
under-training mistake the coordinator found in the first GPU pass. Full-run commands, GPU sizing and
time estimates are in `explore/QUEUE_learning.txt`.

- `explore/learning_proper_score_objectives.py` -- ideas A, B, C, D. Trains `ProbeBranch` and the new
  `BiasOnlyBranch` (K+1 params) on typed-decisions' collapsed tasks under five objectives (`ce`, `cb_ce`,
  `logit_adj`, `proper`, `reinforce`), reusing `tarski.train.{FeatureCache,predict_logits,evaluate,
  fit_temperature}` for apples-to-apples metrics against the existing sweep baselines.
  - Smoke: `.venv/bin/python explore/learning_proper_score_objectives.py --smoke` (CPU, 2 tasks, ~35-50s).
  - Full: see `explore/QUEUE_learning.txt` (8 collapsed/contrast tasks x 2 branch types x 5 objectives =
    80 trainings, each now correctly scheduled to >=300 steps; ~10 min on an A10).
- `explore/learning_active_distillation.py` -- idea E. Loads the real `laya` `typed-decisions` checkpoint
  as a live teacher and runs seeded active-vs-random-vs-hard-label label-efficiency rounds against a
  `ProbeBranch` student.
  - Smoke: `.venv/bin/python explore/learning_active_distillation.py --smoke` (CPU, 1 task, 1 round, ~12-14s,
    including loading the real ModernBERT-large teacher and querying it three times).
  - Full: scaled up per the coordinator's request to all 5 customer-service decisions, 10 rounds, 20
    messages/round, uncapped pool (~250-270 messages/task) -- see `explore/QUEUE_learning.txt`; ~15 min on
    an A10.
- `explore/learning_label_efficiency.py` -- idea K. Trains `BlockBranch(split=14, depth=2)` ("blocks@14+2")
  on both hard and soft labels at N=20..270, reports the soft-vs-hard leverage ratio, and (at N=50 and 270
  only, since it is real per-step full-model compute) trains `BlockBranch(split=0, depth=trunk.n_layers)`
  -- this codebase's own "full fine-tune" -- at the same N, for a direct branch-vs-full-fine-tune curve.
  - Smoke: `.venv/bin/python explore/learning_label_efficiency.py --smoke` (CPU, ~21-35s; `min_steps` is
    explicitly overridden down for the smoke run's full-fine-tune point only, since that path is real
    22-layer backprop rather than a cached-feature lookup and would otherwise ignore `--smoke`'s intent).
  - Full: see `explore/QUEUE_learning.txt`; ~25 min on an A10 (memory is not the constraint -- a
    ModernBERT-base full fine-tune fits easily in 24GB -- the estimate is dominated by wall-clock time for
    the two full-fine-tune points; drop to `--no-full-finetune` if the queue needs headroom).
- `explore/learning_self_training.py` -- idea F. No teacher, no extra labels: MC-dropout agreement
  (scored with `laya.common.proper_reward` used purely as a symmetric confidence/agreement measure between
  two stochastic views of the same branch) decides which unlabelled pool messages get pseudo-labelled each
  round.
  - Smoke: `.venv/bin/python explore/learning_self_training.py --smoke` (CPU, ~6-9s).
  - Full: see `explore/QUEUE_learning.txt`; ~8 min on an A10.
- `explore/learning_outlier_exposure.py` -- idea J. CLINC150's `oos` binary task, with Banking77 messages
  as a free auxiliary "known outlier" corpus (Hendrycks et al.'s outlier-exposure loss), swept over
  several lambda values including `lambda=0` (today's exact objective, as the baseline).
  - Smoke: `.venv/bin/python explore/learning_outlier_exposure.py --smoke` (CPU, ~4-6s).
  - Full: see `explore/QUEUE_learning.txt`; ~8 min on an A10.
- `explore/learning_route_imprinting.py` -- idea H. Hides one CLINC150 domain during training, imprints
  its output row from k examples with no fine-tuning, then compares a surgical (new-row-only, LayerNorm
  frozen) fine-tune against a naive (whole-head) fine-tune and a from-scratch ceiling, on the SAME small
  replay set, for old-domain retention vs. new-domain accuracy.
  - Smoke: `.venv/bin/python explore/learning_route_imprinting.py --smoke` (CPU, ~3-5s).
  - Full: see `explore/QUEUE_learning.txt`; ~6 min on an A10.
- `explore/learning_curriculum.py` -- idea I. A quick shallow-depth (depth 3) logistic probe's per-example
  margin orders training into an easy-to-hard "baby steps" curriculum for a deeper (`split=8`) `ProbeBranch`,
  compared against `train_branch`'s own random-order schedule at the identical total step budget.
  - Smoke: `.venv/bin/python explore/learning_curriculum.py --smoke` (CPU, ~5-7s).
  - Full: see `explore/QUEUE_learning.txt`; ~6 min on an A10.

Not implemented as a script: idea G (branch merging) was implemented and run directly by the coordinator
in `explore/merge_branches.py`, not by this agent -- see idea G's writeup above for the (negative) result.
