# Read once, decide many: forked task branches on a frozen trunk, for local decision models

*Working draft. Numbers marked TBD are filled from `results/tarski/` as runs finish.*

## Positioning

Routing and triage ask several small decisions about each message (team, category, urgency,
escalation). Today's open decision models pay for each decision with a full pass through the model:
fine-tuned classifiers need one model per decision, and question-in-input decision models (laya,
Verdict, JevK5, CLM) re-read the message once per question.

Sharing a frozen trunk across tasks is **not new**. Wei, Qi & He (ACL 2022) froze BERT's bottom
layers, fine-tuned each task's top layers independently at a per-task depth, distilled them and
merged the tasks into one graph, serving 21 tasks at Xiaomi on 5 GPUs instead of 42. Mainstream
(USENIX ATC 2018) did the same for video, with apps branching from a shared stem at different depths.
AdapterDrop, Embedding Recycling and Anthropic's representation re-use for classifiers (2025) are
close relatives (see `research_notes/novelty_check.md`).

This thesis re-examines that design for a different setting: **local-first decision models trained by
end users on their own labels, on a laptop, served next to question-in-input decision models**. What
is new is narrow and testable:

1. **A one-pass split selector.** Each task's split depth is read off a linear-probe curve computed
   from a single trunk pass over cached activations, instead of a fine-tuning sweep per depth (the
   main cost Wei et al. report).
2. **Truncate-and-fork without a teacher.** Branches are copies of the base's next layers trained
   directly on cached activations, with no distillation stage, and a controlled ablation of where
   branch layers come from (next layers, final layers, random) with the upper base layers dropped.
3. **Per-request trunk cut-off with hot-loaded branches.** Each request runs the trunk only as deep as
   its deepest requested branch; branches are loaded and evicted on demand; branches that share a
   split run as one batched computation.
4. **A three-way cost comparison on consumer hardware**: shared trunk vs N separate fine-tuned models
   vs a question-in-input decision model (laya), on Apple silicon GPU and CPU.
5. **Evidence on ModernBERT**, whose alternating local/global attention may split differently than BERT.

## Hypotheses

- **H1 (accuracy).** A branch of 1–2 layers copied from the base's next layers, reading a mid-depth
  trunk state, comes within a few points of a full fine-tune per decision.
- **H2 (cost).** The marginal cost of an extra decision is a small fraction of a full pass, so N
  decisions per message cost far less than N fine-tuned models or N question-in-input passes.
- **H3 (switching).** Changing which decisions are served is a branch-file load (milliseconds,
  megabytes), never a base-model reload (seconds, hundreds of megabytes).
- **H4 (depth per task).** Decisions differ in how deep they need to go, and the one-pass probe curve
  picks a split close to the best one found by a full sweep, at a small fraction of its cost.
- **H5 (initialisation).** Copying the base's next layers beats random layers and copies of the final
  layers when the trunk is frozen below the split and the upper layers are dropped.
- **H6 (isolation).** Because the trunk is frozen, adding or retraining a branch cannot change any
  other branch's outputs (true by construction; tested in `tests/`).

## Method

- **Trunk.** `answerdotai/ModernBERT-base` (22 layers, 768 wide, 149M parameters), frozen. The layer
  loop reproduces the stock forward pass (max |diff| < 1e-4, `tests/test_trunk.py`).
- **Branches.** `probe@k`: LayerNorm, mean pool, linear on the trunk state after layer k.
  `blocks@k+d`: copies of base layers [k, k+d) fine-tuned per task, final norm, mean pool, linear.
  `full`: copies of all 22 layers (embeddings frozen), the per-task fine-tuning reference.
- **Training.** Trunk states at depth k are computed once and cached (fp16); every branch trains on
  the cache (AdamW, one-cycle LR, early stopping on validation accuracy). One temperature per branch is
  fitted on validation NLL.
- **Data.** Banking77 (77 intents), CLINC150 plus (151 intents with out-of-scope; three decisions
  per message: intent, domain, out-of-scope) and typed-decisions (4 workflows x 5 decisions, soft
  teacher labels; laya's full fine-tune scores 0.766 on its test set).
- **Metrics.** Accuracy, macro-F1, ECE and NLL after temperature scaling; per-message latency at batch
  size 1 by N decisions; branch load time and bytes vs a full model copy.

## Results

*Single seed unless noted. Banking77 and CLINC150 inputs (<= 64 tokens) are shorter than ModernBERT's
128-token local window, so local/global layer effects only apply to typed-decisions. CLINC's
out-of-scope class is 1.6% of training, 3.2% of validation and 18.2% of test messages; its intent and
out-of-scope numbers carry that prior shift. typed-decisions results are being re-run after an
early-stopping bug (fixed in `tarski/train.py`) left most of its first probes untrained.*

### H1: accuracy by split depth and branch type

Banking77 (77 intents, test accuracy; full fine-tune 94.0):

| branch | base layers run | accuracy | vs full |
|---|---:|---:|---:|
| probe at any depth 4-22 | 4-22 | 87.3-88.7 | -5.3 to -6.7 |
| blocks@4+2 | 4 | 91.1 | -3.0 |
| blocks@8+2 | 8 | 92.0 | -2.0 |
| blocks@14+2 | 14 | 92.8 | -1.2 |
| blocks@18+1 | 18 | 93.1 | -0.9 |

CLINC150 (three decisions per message): intent full fine-tune 91.1 vs 87.1 for blocks@11+2 (-4.0);
best branch per decision: domain 88.6 (blocks@16+2), out-of-scope 87.7 (blocks@20+2).

**Reading:** a 1-2 layer branch lands 1-3 points under a full fine-tune on Banking77 and about 4 on
CLINC intent. The gap shrinks with split depth, so there is a real accuracy-for-compute trade-off rather
than a free lunch.

typed-decisions (JSON states, ~270 training messages per decision, test accuracy, mean of 3 seeds on the
Lambda GPUs, branches trained for at least 300 optimiser steps):

| | probe@22 | blocks@14+2 | full fine-tune (same architecture) |
|---|---:|---:|---:|
| customer service, 5 decisions | 71.6 | 78.5 | 79.6 |
| other 15 decisions (3 workflows; full fine-tune 1 seed) | 62.8 | 76.4 | 76.5 |
| all 20 decisions (4 workflows) | 65.0 | 77.0 | 77.3 |

For reference, laya's full fine-tune of a 2.8x larger model reports 76.6 on all 20, and Verdict
(ModernBERT-base, same size as ours) reports 77.1. The first typed-decisions runs looked like a failure
(branches at 53-59) because 5 epochs of ~270 messages is ~45 optimiser steps; the harness now trains
every branch for at least 300 steps. **H1 holds on JSON states too: 1.1 points under a same-size full
fine-tune on customer service, and level with it (76.4 vs 76.5) on the other 15 decisions, where
neither wins consistently: the full fine-tune is ahead by 3-8 points on five decisions and behind by 3-9 on four.**

### H2/H3: cost per decision and task switching

Median latency per message at batch 1, nothing else running on the machine
(`results/tarski/latency_{mps,cpu}_clean.json`), 512-token message:

| decisions per message | separate fine-tuned models | laya (question-in-input, 421M) | blocks@11+2 | blocks@11+1 | probe@11 |
|---:|---:|---:|---:|---:|---:|
| **Apple M6 GPU** | | | | | |
| 1 | 26 ms | 111 ms | 15 ms | 14 ms | 13 ms |
| 5 | 127 ms | 240 ms | 23 ms | 18 ms | 14 ms |
| 20 | 507 ms | 937 ms | 50 ms | 33 ms | 16 ms |
| **CPU** | | | | | |
| 1 | 124 ms | 342 ms | 73 ms | 68 ms | 62 ms |
| 5 | 622 ms | 1,557 ms | 118 ms | 92 ms | 63 ms |
| 20 | 2,518 ms | 6,411 ms | 289 ms | 183 ms | 66 ms |

Marginal cost of one more decision on CPU (512 tokens): 124 ms as a separate model, ~320 ms for laya
(which also uses a 2.8x larger encoder), 11 ms for a 2-layer branch, 6 ms for a 1-layer branch and
0.2 ms for a probe. **H2 holds**: at 20 decisions shared-trunk branches are 9-10x cheaper than
separate models on both the GPU and the CPU, and probes are ~38x cheaper on CPU.

**Apple Neural Engine trunk (local-first).** Converting the trunk to Core ML with static shapes
(`explore/lit_eff_ane.py`; matches `Trunk.taps` to ~1e-6) runs it to depth 8 in 0.81 ms (fp16) or
0.57 ms (int8 weights) per message at batch 1 on the M6's Neural Engine, vs 8.3 ms on PyTorch MPS
(5-14x faster across depths). Accuracy is unchanged in fp16 (Banking77 probe 89.04 vs 89.01, blocks:2
91.94 vs 91.97, 99.8-99.9% identical decisions); int8 weights hold at depth 8 but produce non-finite
states at depth 14 because of a few very large activation channels (the outlier-aware quantisation
experiment addresses this). Use the `CPU_AND_NE` compute unit: `ALL` sometimes lands on the GPU and is
~4x slower. Running the trunk on the Neural Engine alongside GPU branches raises throughput 1.2x at 20
decisions and 2.3x at one. Branches that share weight matrices (SVF) run 20 decisions in 4.9 ms of branch
time vs 11.3 ms for separate branches.

Fused multi-branch execution (`tarski/fused.py`) did not beat running branches in a loop on the M6
GPU; it gives a small gain on CPU for short messages only.

**H3 is weaker than claimed.** Loading a 20 MB branch took 18 ms (GPU) / 6 ms (CPU); loading a full
596 MB model copy took 84 ms / 14 ms, but from a warm page cache and with lazy weight loading, so this
understates a cold switch. The robust saving is memory: 20 MB vs 596 MB per resident task.

Cold-cache re-test on the A10 (page cache dropped before each trial; load + first decision, median of
5; other jobs shared the GPU, so absolute times are inflated): probe 191 ms, 2-layer branch 201 ms,
full model copy 314 ms, laya checkpoint 441 ms. Keeping 5 tasks resident: base + 5 branches 806 MB of
GPU memory vs base + 5 full copies 3,457 MB (4.3x, and the gap grows with every task). **H3, restated:
switching is cheap either way on fast storage; what the design saves is memory per resident task.**

### H4: automatic split depth

The one-pass probe curve costs 16-49 s per dataset. On Banking77 and CLINC150 the curve is flat
(Banking77 probe validation accuracy 90-91 from depth 1 to 22), so the selector picks depths 1-3.
Branches trained there cost almost no trunk compute but lose accuracy against deeper branches:
Banking77 blocks@1 90.9 vs 93.1 for the best sweep branch; CLINC150 blocks@1 86.5 intent, 85.4 domain,
85.8 out-of-scope vs 87.1-88.6. **Probe accuracy by depth does not predict branch accuracy by depth**
when the curve is flat: the selector optimises the wrong curve. On typed-decisions the chosen depths do
differ widely between decisions (1 to 20), which supports the premise that decisions need different
depths. **Revised selector (branch proxy).** Ranking depths by 120-step 1-layer branch runs at depths 2-20 picks
depth 18 on Banking77 (blocks@18+2 92.8 vs the best sweep config 92.9) and depth 18 for CLINC intent and
domain (88.1 and 88.1, level with the best configs); it misses on CLINC out-of-scope (picks 2: 85.6 vs
87.9), the task with the train/test prior shift. Selection takes 104 s (Banking77) and 381 s (CLINC, 3
tasks) on the A100, several times cheaper than sweeping full branches. On the typed customer-service
decisions (31 validation rows per task) the proxy picks depths averaging 73.8 against 78.5 for a fixed
blocks@14+2. **H4 is mixed: the branch proxy works when validation sets are large (Banking77, CLINC
intent/domain) and fails on small ones or under prior shift; probes do not work at all.**

### H5: where branch layers come from

2-layer branches with the trunk frozen below and the upper base layers dropped (test accuracy):

| dataset, split | copy next layers | copy final layers | random init |
|---|---:|---:|---:|
| Banking77 @4 | 91.5 | 89.9 | 92.0 |
| Banking77 @8 (2 seeds) | 92.0 / 91.8 | 91.0 / 90.4 | 91.9 / 92.2 |
| Banking77 @14 | 92.7 | 91.9 | 92.4 |
| CLINC150 intent @4 | 85.4 | 84.2 | 87.5 |
| CLINC150 intent @8 (2 seeds) | 86.2 / 86.8 | 85.0 / 85.4 | 87.6 / 87.7 |
| CLINC150 intent @14 | 88.5 | 86.8 | 88.4 |
| typed customer service @8 (3 seeds, 5 decisions) | 75.1 | 69.1 | 75.7 |
| typed customer service @14 (3 seeds) | 77.6 | 74.1 | 77.4 |
| typed customer service @20 (3 seeds; next = final layers here) | 74.4 | 74.5 | 77.1 |

**H5 is refuted, on short text and on JSON states.** Copying the base's next layers is no better than random layers, and worse at
shallow splits on CLINC150 (-1 to -2 points); copying the final layers is consistently worst. This
agrees with Zhang et al. (ICLR 2021) on re-initialising top layers and contrasts with MORES, where
copying helped a cross-attention module. Practical consequence: branches do not need the base's
upper-layer weights at all, which also frees branch architecture from the base's layer shape.

### Exploration

57 ideas from five exploration passes are ranked in `research_notes/explore/TRIAGE.md`; 14 are
queued. First result: making laya's attention one-way (the question reads the message, the message
never reads the question) without retraining scores 0.684 on typed-decisions vs 0.767 unsplit and
0.453 for the block split used in `readonce/`, while each extra question costs 4.5x fewer
token-layers. Retraining with the one-way mask (as MICE, EMNLP 2026, trains asymmetric attention) is
the obvious next step.

## Few labels (k-shot)

Branch accuracy against a full fine-tune when each route has only k labelled examples
(`explore_kshot_*.json`; out-of-scope held out of training, so a different protocol from the sweeps):
the gap grows as labels shrink, from -0.9 with all labels to -2.9 (k=25) and -3.7 (k=10) on Banking77,
and -5.9 / -11.8 on CLINC. With a single example per route, matching messages against the route names
beats every trained branch. **H1 holds with ~100 labels per route; with ~10 a full fine-tune is worth
4-12 points on the ModernBERT trunk.**

## Findings beyond the hypotheses

Full registry with verdicts, numbers and anomalies: `THEORIES.md` (91 theories: 31 supported, 26 mixed,
25 refuted, 3 known, 2 inconclusive, 4 not tested).

1. **The trunk matters more than the branch.** Swapping ModernBERT-base for gte-modernbert-base (same
   architecture, contrastively trained; one seed, same protocol) lifts frozen features enough that a
   plain probe nearly matches a ModernBERT full fine-tune: Banking77 probe 93.4 (full 93.8), CLINC intent
   probe 90.5 (full 90.7), typed customer-service blocks@14+2 80.8 (full 79.6), and zero-shot routing
   from route names alone reaches 64-71% on CLINC. It needs a 3-seed confirmation on all 20 typed tasks.
2. **Out-of-scope needs no out-of-scope labels.** Relative Mahalanobis on normalised trunk states
   reaches AUROC 0.957, equal to a supervised detector, and the share of traffic that fits no route can
   be estimated to within ~0.02 and used to set the threshold.
3. **Routes can be added and removed in closed form.** LDA/KLDA/APER heads on trunk (or branch)
   features add a route in seconds with exact unlearning; after 11 incremental sessions accuracy is 93.0
   vs 94.3 for retraining.
4. **laya can be made read-once.** Fine-tuning laya for two epochs with a one-way mask (the question
   reads the message, the message never reads the question) scores 0.784 vs 0.767 for stock laya, with
   the message encoded once and 4.5x less compute per extra question.
5. **The Mac's Neural Engine runs the trunk 5-14x faster than PyTorch on the Mac GPU** at unchanged
   accuracy (depth 8 in 0.57-0.81 ms).
6. **Refuted directions:** merging independently trained branches (TIES), joint multi-task bodies,
   passenger tokens, per-request pruning, zero-shot routes from descriptions on ModernBERT, LLM-written
   out-of-scope negatives, and copying specific base layers into branches (H5).

## Conclusion

A frozen encoder with small per-decision branches is a sound local-first design for routing and triage:
with ~100 labels per route it lands within about 1 point of a same-size full fine-tune on intents and
on JSON ticket states, makes each extra decision ~10x cheaper than a separate model, keeps tasks
isolated, and runs in milliseconds on a laptop (sub-millisecond trunk on the Neural Engine). The core
architecture is prior art (Wei et al. 2022); what this work adds is the evidence for the local,
user-trained setting, the finding that random branch layers do as well as copied ones, a cheap branch
proxy for split depth (with its limits), the few-label curve, the one-way retraining of laya, and a set
of label-free out-of-scope and route-addition tools. The biggest lever found is the choice of trunk.

## Prior work

Closest: Wei, Qi & He, "A Flexible Multi-Task Model for BERT Serving", ACL 2022
(https://aclanthology.org/2022.acl-short.89/); Jiang et al., "Mainstream", USENIX ATC 2018; Rücklé et
al., AdapterDrop, EMNLP 2021; Du et al., "General Purpose Text Embeddings from Pre-trained Language
Models for Scalable Inference", Findings of EMNLP 2020 (one frozen encoder's features shared across many
tasks, stored with binary quantisation; sharing beats distillation from about 7 tasks);
Saad-Falcon et al., Embedding Recycling, EACL Findings 2023; Cunningham
et al. (Anthropic), cost-effective classifiers via representation re-use, 2025; Gao et al., MORES,
EMNLP 2020 and de Jong et al., LUMEN, ICML 2023 (copy-initialised layers); Lee et al. 2019 and Sajjad
et al. 2020 (freezing and dropping BERT layers); Zhang et al., ICLR 2021 (re-initialising top layers
can help few-sample fine-tuning, a counterpoint for H5). Full notes, verdicts and queries:
`research_notes/novelty_check.md`.

## Limits

- Fixed label sets: a branch answers only the labels it was trained on. New routes need labelled
  examples (or a fallback to a question-in-input model such as laya/OpenJev, which the server supports).
- One base per store: a branch is bound to the exact base weights it was trained on and must be
  retrained when the base changes (retraining on cached states is minutes).
