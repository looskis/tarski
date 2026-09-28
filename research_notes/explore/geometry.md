# Geometry lens: what the frozen trunk's hidden states make easy

Exploration notes, 2026-09-27. Lens: representation geometry and interpretability of the frozen
ModernBERT-base trunk (22 layers; layers 0, 3, ..., 21 global, the rest 128-token windows).
Code: `explore/geometry_passenger.py`, `explore/geometry_oneway_laya.py`, `explore/geometry_depthscan.py`; follow-ups: `geometry_oneway_train.py`, `geometry_tapmix.py`, `geometry_subspaces.py`, `geometry_sufficient.py`, `geometry_descanchor.py`, `geometry_atlas.py`, `geometry_globalfork.py` (shared helpers in `geometry_common.py`). Queue: `explore/QUEUE_geometry.txt`.

## What I looked at first

- **The question never needs to be seen by the message.** In laya (and in every question-in-input model),
  the tokens that produce the answer are on the question side: the option `[MASK]` markers. The message
  only has to be *readable*. That points to attention masks where the message is read-only. The message's
  states then do not depend on the question, so one trunk pass can serve every question, branch and
  probe. Ideas 1–3 all use this.
- **The typed-decisions "probe collapse" was mostly an early-stopping artifact.** In the old log
  (`results/tarski/sweep_typed_buggy_earlystop.log`), most probes stopped at epoch 4 or 5 with their best
  epoch at 1: the majority class, on about 26 validation messages. This was fixed in
  `tarski/train.py` while I worked (no epoch selection below 50 validation rows). My scripts use the
  same rule and add a tuned L-BFGS/ridge probe baseline, so every comparison has a fair probe.
- **The existing sweep hints at a global-layer effect.** On Banking77, the only `blocks@k+1` whose copied
  layer is global (`blocks@18+1`, copying layer 18) is the best branch (93.1). Every `+1` branch that
  copies a local layer scores 90.2–91.5. On CLINC the pattern does not hold (`blocks@6+1`, global:
  86.0 vs `blocks@11+1`, local: 86.7). Banking77 and CLINC messages are at most 64 tokens, so there
  the windows never bind and the two layer types differ only in weights and RoPE theta. The skeptic lens
  (`explore/skeptic_receptive_field.py`) makes the same point.
- **An early CPU signal that did not hold.** On 60 pairs, one-way laya matched unsplit laya (0.700 vs
  0.700). On the full test set it scores 0.684 vs 0.767 unsplit (block split 0.453). See idea 2.

Overlaps with sibling lenses, so they can be deduplicated:
- Idea 1 is close to the systems lens's KV-query fork (`explore/systems_kvquery.py`). That fork starts
  at the split with fine-tuned copies of d layers. Idea 1 rides the frozen trunk from the embeddings up,
  with learned tokens and a LoRA on the passenger rows only.
- Idea 6's cross-task check is close to far-field's syndrome decoding (`explore/farfield_syndrome.py`).
- Idea 5 is close to skeptic's receptive-field study.

Ideas 2 and 3 (one-way questions over a trained or frozen MLM) are not covered by any other lens.

---

## 1. Passenger tokens: read-only per-decision tokens that ride the frozen trunk

**Status: tested (CLINC) · rerun queued (typed-decisions).**
- **CLINC150 (mean of intent, domain, oos):** P4r8 scores 86.7, level with `blocks@11+2` (87.0). The
  baselines are `probe@22` 85.2 and `logreg@22` 85.0; oos AUROC is 0.856 vs 0.854 for the probe.
  P4r0 (tokens only) 81.9 and P1r0 79.6 are too weak without the LoRA.
- **typed-decisions: under-trained.** P4r0/P4r8 scored ~55.6–55.8, well below `probe@22` 65.7 and `logreg@11` 68.0.
  Each task's parameters only step on their workflow's batches, which gave about 136 steps per task.
- **Fix and rerun.** The loop now enforces ≥ 300 steps per task (18 epochs on typed), and the rerun is
  queued as `passenger_typed_v2`.

**Mechanism.** Each decision owns P learned vectors (P = 1–4) that enter at the embedding layer and ride
through the frozen trunk's own layers, next to the message:
- at every layer, their queries attend to the message's keys/values (and to each other);
- the message rows never attend to passenger columns, so the message's hidden states are bit-for-bit the
  shared trunk pass (parity checked: relative difference below 2e-6), and probes, blocks branches and every
  other passenger read the same pass;
- a decision costs P tokens times the layers ridden, not d layers times all tokens: 88 token-layers
  for P=4 over 22 layers, against about 480 for `blocks@k+2` on a 240-token JSON state;
- because passengers ride from depth 0 to 22, one decision reads every depth. There is no split to choose,
  and the trunk's own attention does the pooling, which avoids mean-pooling collapse on long JSON states;
- optional capacity: a rank-r LoRA applied only to the passenger rows (an encoder analogue of IBM's
  activated LoRA). It adds about 100k parameters per rank and keeps the message pass shared.

A variant rides only to depth 11, which also truncates the trunk.

**Novelty.** Partially done.
- **Read-only prompts (the soft-token mechanism).** RPO (Lee et al., "Read-only Prompt Optimization for
  Vision-Language Few-shot Learning", ICCV 2023, https://arxiv.org/abs/2308.14960) uses masked
  attention so that prompts do not shift CLIP's features.
- **Per-layer query tokens.** VQT (Tu et al., CVPR 2023, https://arxiv.org/abs/2212.03220) adds query
  tokens per layer, but their outputs are not propagated.
- **Label embeddings over frozen features.** Y-Tuning (Liu et al. 2022,
  https://arxiv.org/abs/2202.09817) passes label embeddings through new cross-attention layers over
  frozen features.
- **Token-scoped LoRA.** Activated LoRA (Greenewald et al. 2025, https://arxiv.org/abs/2504.12397) applies
  LoRA only to tokens after an invocation, so the causal KV cache can be reused.
- **Sibling lens.** The systems lens's KV-query fork forks the CLS at the split into fine-tuned layer
  copies.

Not found: passengers as a many-decision branch type on one shared bidirectional encoder pass, riding
from the embeddings with the frozen weights, with passenger-only LoRA and per-decision cost accounting.

**Cheap experiment.** `explore/geometry_passenger.py` (implemented).
- **Data and configs.** typed-decisions (20 tasks) and CLINC150 (intent, domain, oos). Configs:
  `P4r0`, `P4r8`, `P4r8@11`, `P1r0`.
- **Baselines, in the same run:** tarski `probe@22`, and L-BFGS logistic regression on standardized
  mean pools at depths 11 and 22 with the L2 penalty tuned.
- **Metrics:** accuracy, macro-F1, ECE, Brier against soft labels, and AUROC on oos.
- **Expected signal.** On typed-decisions, `P4r8` should beat the tuned logistic regression by several
  points, because attention can find the one field that matters. On CLINC oos, AUROC should rise over
  the probe.
- **Kill criterion:** passengers no better than the tuned logistic regression at the same trunk pass.
- **Runtime:** about 25–30 minutes for everything, or two queue entries of about 15 minutes each.

**Wildness: 3.**

## 2. One-way questions on laya: the question reads the message, the message never reads the question

**Status: tested; retraining follow-up queued (`oneway_train`, A100).**
- **Full test set, 2,000 pairs, no retraining:**

  | configuration | accuracy |
  |---|---|
  | one-way (k=28 + one-way head) | 0.684 |
  | unsplit | 0.767 |
  | block split at k=28 | 0.453 |
  | one-way@7 | 0.716 |
  | one-way@14 | 0.694 |

- **What one-way keeps.** It recovers 73% of the block split's loss (0.231 of the 0.314 points lost by
  block@28 vs unsplit). Cost is 1,942 token-layers per question vs 8,736 unsplit (4.5× less).
- **What failed.** The 60-pair CPU parity (0.700 vs 0.700) did not hold at scale; the gap is 8.3 points.
  Score questions lose most: 0.608 vs 0.724.
- **Queued.** `explore/geometry_oneway_train.py` fine-tunes laya's typed-decisions checkpoint under the
  one-way mask. It has three variants:
  - full fine-tune;
  - a LoRA on question rows only, which keeps the state encoder bit-for-bit laya's (verified in smoke:
    state-row difference 0.0);
  - a joint-mask control with the same recipe.

**Mechanism.**
- **Split.** laya encodes `[CLS] question [SEP] [MASK] opt ... [SEP] | state [SEP]` once per question.
  readonce makes the state cacheable with a block-diagonal split below depth k (0.617 at k=14, 0.453 at
  k=28). One-way changes only which rows see which columns: state rows attend only to state rows, while
  question rows (including the option markers) attend to the question *and* the state at every layer.
- **Why the state is cacheable.** ModernBERT's RoPE and windows are relative, so the state's states are
  identical whatever question sits in front (checked: relative difference 1.9e-6 for questions of 65 and
  32 tokens). The state is encoded once per message.
- **Cost per question.** Each further question costs only its own ~40–60 tokens through the layers, plus
  attention reads of the cached state keys and values. That is the same cost as the block split, with
  strictly more information flow.
- **Head.** laya's 2 appended head layers can be made one-way too. The logits move by less than 0.01, so
  the head is not where the question needs the state to see it.
- **Retraining.** None. The script measures how much of laya's accuracy ever needed the state to see the
  question.
- **Payoff if it holds.** Laya's flexibility (any question, any options, no labels) at read-once cost,
  and on the same cached message pass the tarski branches use, provided the trunk is laya's encoder.

**Novelty.** Partially done.
- **Closest: MICE** (Vast et al., EMNLP 2026, https://arxiv.org/abs/2602.16299). Query and document are
  encoded independently up to about half the depth; above that the query attends to *fixed* document
  representations. It is trained with distillation, and the ModernBERT-family variants lose more.
- **Sparse cross-encoders** (Schlatt et al., ECIR 2024, https://arxiv.org/abs/2312.17649). They train
  with restricted attention directions and find the *query* need not attend to the document, which is the
  opposite direction.
- **cbjev** (tomek7667, Sep 2026,
  https://dev.to/_tomek7667/ask-ten-questions-read-the-text-once-making-a-decision-model-67x-faster-k0e).
  It packs up to 10 laya questions into one sequence. Questions never see each other, but the document
  attends to all of them: 0.749 with no training, 0.783 after fine-tuning. The document is read once
  per *call*, not cached across calls or question sets.
- **Decoder analogues.** Prefix caching and the UniLM prefix-LM mask (Dong et al. 2019,
  https://arxiv.org/abs/1905.03197) are the causal analogue.

Not found: converting an already-trained bidirectional decision cross-encoder to one-way attention *with
no retraining*, so that the text is encoded once for any future set of questions, and measuring what that
costs on typed decisions.

**Cheap experiment.** `explore/geometry_oneway_laya.py` (implemented).
- **Data:** typed-decisions test (2,000 pairs) with laya's typed-decisions checkpoint.
- **Grid:** block and one-way at k ∈ {0, 7, 14, 20, 24, 26, 28}, plus a fully one-way head.
- **Metrics:** accuracy by question type, Brier, ECE, and token-layers per question vs unsplit.
- **Baseline:** unsplit laya, 0.766.
- **Expected signal.** One-way at k=28 within a few points of 0.766 while block@28 sits at 0.45. The
  60-pair CPU sample gave 0.700, 0.333 and 0.700 respectively.
- **Next step if it holds:** fine-tune only the question-side rows with a LoRA (the aLoRA analogue) under
  the one-way mask, to close the remaining gap without touching the shared state encoder.
- **Runtime:** about 6 minutes.

**Wildness: 3.** The mask is simple; the claim that no retraining is needed is the surprising part.

## 3. One-way cloze routes: zero-shot decisions as read-only prompts over the shared trunk pass

**Status: tested; partial.**
- **CLINC150 domain (10-way, zero-shot):**
  - two-way PET: 0.51–0.61 raw, 0.62–0.64 prior-corrected;
  - one-way: 0.37–0.53 raw, 0.57–0.59 prior-corrected;
  - agreement between the two: 63–75%.

  The predicted ≤ 2-point gap with > 85% agreement failed: the gap is 4–6 points after prior correction.
- **Banking77 (77-way):** with the best template ("Topic:"), one-way matches two-way: 0.418 vs 0.417
  prior-corrected, with 59% agreement. Other templates lose 1–4 points.
- **Base MLM zero-shot is weak** in absolute terms.

**Mechanism.**
- **What rides.** A new route with no labelled data is a cloze template
  (" This message is about [MASK].") whose tokens ride as passengers behind the message:
  - positions continue after the message;
  - local windows are respected, exactly as if the template were appended;
  - the message never sees them.
- **Scoring.** The frozen MLM head reads the `[MASK]` passenger, and verbalizer words score the labels.
  An optional label-prior correction (unsupervised) is applied.
- **Cost.** The message's states are the same shared pass the tarski branches read, so every zero-shot
  question costs only its ~8 template tokens, not a re-read of the message.
- **What it bridges.** laya's "ask anything" and tarski's shared trunk. A route can exist as a template
  on day 0 and be replaced by a trained branch once labels accumulate.

**Novelty.** Partially done.
- **Cloze classification with an MLM head:** PET (Schick & Schütze, EACL 2021,
  https://arxiv.org/abs/2001.07676).
- **ModernBERT's MLM head as a zero-shot classifier:** ModernBERT-Large-Instruct (Clavié et al. 2025,
  https://arxiv.org/abs/2502.03793). It uses full attention over the prompt.
- **Read-only prompts:** RPO (above).
- **Decoder analogue:** prefix caching makes suffix prompts one-way for free.

Not found: cloze prompts as one-way passengers in a bidirectional encoder, so zero-shot questions reuse
a cached message encoding. Measuring the accuracy gap between one-way and two-way prompting is the new
part.

**Cheap experiment.** Implemented in `explore/geometry_passenger.py` (the `--cloze` stage).
- **Data:** CLINC150 domain (10 in-scope domains, 4,500 test messages, hand-written verbalizers) and
  Banking77 (label-name verbalizers).
- **Templates:** three.
- **Variants:** two-way PET (stock forward on message plus template), one-way with windows respected,
  and one-way global.
- **Metrics:** accuracy, prior-corrected accuracy, agreement with two-way, and token-layers per question.
- **Expected signal.** Absolute zero-shot accuracy from base ModernBERT will be modest; the Instruct
  checkpoint would be the real vehicle. The measurement is the gap: one-way within about 2 points of
  two-way, with agreement above 85%, at about 1/3–1/30 of the per-question compute.
- **Runtime:** about 3 minutes, inside the passenger run.

**Wildness: 4.**

## 4. Token-identity-centred pooling for long structured states

**Status: tested; refuted.**
Centred pools are *worse* than plain mean pools at every depth tested:

| dataset | change vs plain mean pool |
|---|---|
| typed-decisions | −2.4 to +0.6 points |
| CLINC150 | −0.7 to −1.6 points |
| Banking77 | −0.8 to −4.7 points |

Source: `explore_depthscan.json`.

**Mechanism.**
- **Why mean pooling fails here.** Mean-pooling a 240-token JSON state averages mostly schema: keys,
  braces, the same field names in the same order in every message.
- **The fix.** From each token state, subtract the mean state of its token id, computed once over
  training messages with no labels:
  `pool_c(x) = mean_t (h_t − μ[id_t]) = mean_t h_t − mean_t μ[id_t]`.
  The second term depends only on the token ids, so at inference it is one table lookup and one
  subtraction per message, with no second pass.
- **Rare tokens keep their signal.** μ is shrunk towards the global mean token state by ~20 pseudo-counts,
  so rare, content-bearing tokens are centred on the global mean, not on themselves.
- **Where it applies.** Any probe or passenger readout.

**Novelty.** Partially done.
- **SIF** (Arora, Liang & Ma, ICLR 2017, https://openreview.net/forum?id=SyK00v5xx) down-weights
  frequent words and removes the common component, for static vectors.
- **All-but-the-top** (Mu & Viswanath, ICLR 2018, https://arxiv.org/abs/1702.01417) removes the dominant
  directions.
- **PromptBERT** (Jiang et al., EMNLP 2022, https://arxiv.org/abs/2201.04337) subtracts a template-only
  embedding ("template denoising") to remove static token bias.

Token-id-conditional centring of *contextual* states, with shrinkage, used as a pooling for decision
probes on structured inputs: not found, but it is a small step from these.

**Cheap experiment.** `explore/geometry_depthscan.py` (implemented). It uses ridge probes, with the L2
penalty tuned on validation, on centred against plain mean pools at depths {3, 6, 9, 11, 12, 16, 22}, for
typed-decisions, CLINC and Banking77.
- **Expected signal:** +2–5 points on typed-decisions, and about 0 on short intents.
- **Kill criterion:** no gain on typed-decisions.
- **Runtime:** included in the depth scan, about 5–8 minutes in total.

**Wildness: 2.**

## 5. Global-layer-aligned splits and "global-only" forks

**Status: (a) and (b) tested, negative; (c) queued.**
- **(a) Probe sawtooth.** None for mean pools (±0.2 points). [CLS] pools are about 1 point better right
  after global layers on typed-decisions (0.640 vs 0.629) and Banking77 (0.754 vs 0.742).
- **(b) `blocks@15..20+1` on Banking77.** 91.5, 91.9, 92.6, 92.7, 91.7, 91.5. The global layers 15 and 18
  are not special: layer 15 is the worst.
- **(c) Global-only fork.** `explore/geometry_globalfork.py` is queued as `globalfork_typed` and
  `globalfork_banking77`.

**Mechanism.** ModernBERT mixes information across the whole message only in every third layer.
- **Hypothesis.** A branch whose copied layers include a global layer, or a split right after one, gets
  more for its parameters, because its extra layer can re-pool the whole message.
- **Architecture variant: the global-only fork.** The branch copies only the next one or two *global*
  layers above the split (skipping the local ones), for example `fork@11 → {12, 15}`. That gives two
  layers' cost with global mixing each time.
- **Where the test bites.** Windows bind only on long inputs (typed-decisions, 72% of states exceed 128
  tokens). On Banking77 and CLINC the difference is the layers' learned roles and RoPE theta (160k global,
  10k local).

**Novelty.** Not found.
- There is no layer-type-aware probing or splitting of ModernBERT. The ModernBERT paper
  (https://arxiv.org/abs/2412.13663) does not probe by layer type.
- Hybrid local/global attention exists elsewhere (Longformer, Gemma 2/3), but split-point selection by
  layer type does not.
- Overlaps with the skeptic lens's receptive-field script.

**Cheap experiment.**
- **(a) Probe sawtooth, done in the depth scan.** Mean ridge-probe accuracy after global vs after local
  layers, and the local bump `acc(d) − (acc(d−1)+acc(d+1))/2`, for mean and [CLS] pools on three datasets.
- **(b) Blocks ablation, no new code.** Run on Banking77 and CLINC150:
  ```
  python -m experiments.sweep --dataset banking77 --probe-splits --block-splits 15 16 17 18 19 20 --depths 1 \
      --out results/tarski/explore_globalsplit_banking77.json
  ```
  If the effect is real, `blocks@15+1` and `blocks@18+1` (global) beat 16, 17, 19 and 20 by at least 1
  point. Runtime about 6 minutes per dataset.
- **(c) The global-only fork** needs a `BlockBranch` that takes a non-contiguous layer list, a core
  change of about 10 lines in `tarski/branches.py`: `layers=[12, 15]` instead of `range(split, split + depth)`.

**Wildness: 2.**

## 6. Out-of-scope from trunk geometry, with no out-of-scope training data

**Status: tested; mixed.**
- **Cross-depth disagreement works.** It gives AUROC 0.914 and oos-binary accuracy 86.3% (majority is
  81.8%) with no out-of-scope training data. That beats the supervised ridge oos probe (AUROC 0.840, 84.5%).
- **Mahalanobis alone falls short of the prediction.** The best single depth gives AUROC 0.861, and the
  multi-depth average 0.844, below the predicted 0.90–0.95.
- **The nearest-class-mean Mahalanobis classifier** reaches 88.9% on in-scope intents.

**Mechanism.** OOS detection is weak today: 85–87% oos accuracy against 81.8% for always answering
in-scope. The trunk's geometry gives a detector without training a branch or seeing any OOS examples:
- **Per-depth Mahalanobis.** Class-conditional Mahalanobis distance to the 150 in-scope intent means,
  with a shared covariance on standardized mean pools, at every depth. The statistics come from the same
  cache the branches train on.
- **Multi-depth average.** The average of per-depth z-scored distances.
- **Cross-depth disagreement.** The fraction of depths whose nearest-class prediction differs from the
  modal one. Messages that match no intent keep changing their "nearest intent" across depths.
- **Serving.** A calibrated threshold turns any of these into an `oos` branch that needs no oos labels.
  That matters for user-defined routes, where nobody labels "none of the above".

**Novelty.** Mostly done.
- **Mahalanobis on CLINC150.** Podolskiy et al., "Revisiting Mahalanobis Distance for
  Transformer-Based Out-of-Domain Detection" (AAAI 2021, https://arxiv.org/abs/2101.03778), report
  AUROC 98.4 with *fine-tuned* RoBERTa.
- **Layer-wise aggregation:** Darrin et al., "Unsupervised Layer-wise Score Aggregation for Textual OOD
  Detection" (AAAI 2024, https://arxiv.org/abs/2302.09852).
- **Layer disagreement for uncertainty:** TULIP (2024, https://arxiv.org/abs/2405.17494).
- **Principal-subspace residuals:** ViM (Wang et al., CVPR 2022, https://arxiv.org/abs/2203.10807).
- **Cross-task inconsistency (domain vs intent):** the far-field lens.

What is specific here: a frozen, *not fine-tuned* trunk, statistics that are free by-products of the
branch cache, and cross-depth disagreement of nearest-class predictions as the score.

**Cheap experiment.** Implemented in the depth scan: CLINC150, AUROC, share of oos missed when 95% of
in-scope traffic is kept, and oos-binary accuracy at a threshold tuned on validation.
- **Baselines:** a supervised ridge oos probe, the tarski oos branch (85–87% acc) and majority (81.8%).
- **Expected signal.** Frozen-feature Mahalanobis AUROC around 0.90–0.95, below fine-tuned (0.98) but
  far above what the oos branch implies. Multi-depth ≥ best single depth.
- **Runtime:** about 1 minute inside the scan.

**Wildness: 2.**

## 7. Decision atlas: shared decision directions across schemas

**Status: zero-shot tested (negative); few-shot test queued (`atlas`).**
Urgency transfer, Spearman correlation with the target's expected urgency:
- within a workflow: 0.68;
- across workflows: 0.05–0.09, at depths 11 and 22, for both mean and centred pools.

`explore/geometry_atlas.py` adds severity/risk and needs-human groups, and asks whether the atlas
direction helps as a *prior* with 8–64 target labels.

**Mechanism.** typed-decisions has four workflows that each ask "urgency" (and three that ask a
disposition or severity) over *different* JSON schemas.
- **Hypothesis.** The trunk encodes urgency along a direction that survives the schema change. If so,
  a new workflow gets those decisions zero-shot: an "atlas" of decision directions learned once, with
  the user labelling only what is genuinely new.
- **Test.** Fit a ridge regression from one workflow's pooled states to its expected urgency (from the
  soft labels) and apply it unchanged to every other workflow's states, measuring rank correlation.

**Novelty.** Partially done.
- **Cross-dataset transfer of probe directions** is studied for truthfulness and similar concepts in
  LLMs. Some transfers work; "The Geometries of Truth Are Orthogonal Across Tasks" (2025,
  https://arxiv.org/abs/2506.08572) finds they often do not.
- **Across structured-state schemas** for operational decisions, with a frozen encoder: not found.

**Cheap experiment.** Implemented in the depth scan: typed-decisions, the 4×4 Spearman transfer matrix
at depths 11 and 22, for mean and centred pools.
- **Expected signal.** Within-workflow ρ around 0.4–0.6. Across-workflow ρ > 0.2 would support the
  atlas; around 0 kills it (the smoke sample shows around 0, but with 120 messages).
- **Runtime:** seconds.

**Wildness: 3.**

## 8. Cost-aware tap mixing: let each decision learn how deep the trunk must run

**Status: queued (`tapmix`).**
Known so far: the multi-tap probe (6/11/16/22) gains over the best single depth on three datasets:

| dataset | multi-tap | best single depth |
|---|---|---|
| Banking77 | 89.7 | 84.4 |
| CLINC150 | 83.6 | 80.1 |
| typed-decisions | 70.2 | 69.7 |

`explore/geometry_tapmix.py` measures accuracy against trunk cut: prefix mixers, a cost-weighted group
lasso with refit, and cost-aware selection.

**Mechanism.**
- **Mix.** Instead of one split depth per task, a branch reads a learned mixture of taps
  `Σ_d w_d · pool(h_d)`, and the depths it uses are chosen under a cost penalty. The mixture weights are
  parameterized as a cumulative "stop" distribution, so the penalty is the expected deepest depth used.
- **Serving.** The trunk runs only to the deepest depth any requested decision uses.
- **What it replaces.** The one-pass probe curve that picks the split (tarski's H4). The per-request
  truncation depth becomes a trained quantity with an explicit accuracy/latency knob λ.

**Novelty.** Parts are done:
- the ELMo scalar mix (Peters et al. 2018, https://arxiv.org/abs/1802.05365);
- per-task layer mixing in probing (Tenney et al., ACL 2019, https://arxiv.org/abs/1905.05950);
- Head2Toe feature selection across all layers with group lasso (Evci et al., ICML 2022,
  https://arxiv.org/abs/2201.03529);
- early exit.

A mixture whose penalty is the *deepest* tap used, so that it directly sets per-request trunk
truncation for a set of decisions: not found.

**Cheap experiment.**
- **Setup.** On the depth-scan features (mean pools at all 22 depths): a softmax mix plus linear head per
  task, with penalty λ·E[max depth], for λ in {0, 1e-3, 1e-2}.
- **Metric.** Accuracy against the mean deepest depth needed per message for the 20 typed-decisions tasks.
- **Baseline.** The autosplit choice.
- **Expected signal.** Same accuracy at 20–40% fewer trunk layers per request.
- **Runtime.** About 5 minutes, since features are cached.
- Not implemented.

**Wildness: 2.**

## 9. Shared decision subspaces: fuse branches whose decisions use the same directions

**Status: queued (`subspaces`).**
`explore/geometry_subspaces.py` runs three tests:
- principal-angle affinity with a shuffled-label same-messages null;
- rank-r shared bottlenecks for workflow, affinity-greedy and random groups;
- fused blocks@11+2 (one stack, 5 or 3 heads) against separate branches.

**Mechanism.**
- **Measure overlap.** Fit probes, then compute principal angles between the row spaces of their weight
  matrices at the split depth. Decisions whose subspaces overlap, such as urgency and severity, or domain
  and intent, can share one fused branch.
- **Share compute.** One blocks stack with several heads (tarski's `fused.py` already batches same-shape
  branches), or a shared rank-r projection of the trunk state that all their probes read.
- **Grouping criterion.** A cheap, training-free signal for grouping, computed from the probes the user
  already has.

**Novelty.** Largely done.
- **Shared predictive structure:** ASO (Ando & Zhang, JMLR 2005).
- **Task grouping:** Standley et al. (ICML 2020, https://arxiv.org/abs/1905.07553) and TAG (Fifty et al.,
  NeurIPS 2021, https://arxiv.org/abs/2109.04617).
- **Branch points from representation similarity:** Vandenhende et al. (BMVC 2020).
- **Projection kernel between attention-head weight subspaces** (2026,
  https://arxiv.org/abs/2601.10266).

Using frozen-trunk probe-weight subspaces to decide which user-trained branches to fuse is a small
application step.

**Cheap experiment.**
- **Setup.** typed-decisions ridge weights at depth 11, which the depth scan already computes, go into a
  principal-angle matrix and hierarchical clustering. Then train fused `blocks@11+2` per cluster vs one
  per task.
- **Metric.** Mean accuracy against total branch compute.
- **Expected signal.** Clusters within a workflow; fused branches at no accuracy cost for 2–3× less branch
  compute.
- **Runtime.** About 10 minutes.
- Not implemented.

**Wildness: 2.**

## 10. Sufficient-statistics branches: mergeable, exactly unlearnable, OOD for free

**Status: queued (`sufficient`).**
`explore/geometry_sufficient.py` measures:
- LDA vs NCM vs the logistic probe;
- merge exactness (smoke: relative difference ≤ 4e-12, agreement 1.0);
- exact unlearning (smoke: ≤ 4e-7);
- k-shot new labels without retraining, against retrained probes;
- OOD from the same statistics.

**Mechanism.**
- **What the branch stores.** Per-class sums, counts and one shared scatter matrix at depth k, that is,
  LDA/Gaussian heads.
- **What that buys:**
  - adding a label from five examples is adding a sum;
  - deleting a user's examples is exact subtraction;
  - two users' branches merge by adding their statistics;
  - the same statistics give the Mahalanobis OOD score of idea 6.
- **Fit with local-first.** Deterministic, auditable routes that are editable without gradient descent.

**Novelty.** Already done.
- **Streaming LDA on frozen features:** Deep SLDA (Hayes & Kanan, CVPRW 2020,
  https://arxiv.org/abs/1909.01520).
- **Exact unlearning of ridge heads on frozen foundation models** (2026,
  https://arxiv.org/abs/2603.12977).

Included because it is nearly free here: the depth scan already reports nearest-class-mean
(Mahalanobis) accuracy on CLINC intents at every depth.

**Cheap experiment.** Compare LDA with trained probes on Banking77 and CLINC at their best depths, for
accuracy and edit cost. Also a user-merge simulation: split the training set in two, merge the statistics,
and check the result is identical to training on everything. About 3 minutes.

**Wildness: 1.**

## 11. Branch transport across base upgrades

**Status: ruled out by the user; not tested.**

**Mechanism.** tarski branches refuse to load on a different base, and "retrain" is the only answer.
- **Transport.** Fit a map between the old trunk's depth-k space and the new trunk's depth-k′ space from
  *unlabelled* messages the user already has, either orthogonal Procrustes or ridge on mean pools.
- **Result.** Old probe branches keep working on the new base on day 0, while retraining runs in the
  background.
- **Blocks branches.** These need the map at the token level, applied before the branch's layers.

**Novelty.** Largely done as model stitching.
- **Latent space translation** (Maiorca et al., NeurIPS 2023, https://arxiv.org/abs/2311.00664).
- **Relative representations** (Moschella et al., ICLR 2023, https://arxiv.org/abs/2209.15430).

Applied to user-trained decision branches in a local store across base-model upgrades: not found. That
is an application step, but a practically important one.

**Cheap experiment.**
- **Setup.** ModernBERT-base → ModernBERT-large, both cached locally. Train probes on base at depth 22.
  Fit Procrustes and ridge maps on 2,000 unlabelled Banking77 or CLINC messages.
- **Metric.** Accuracy of the transported probes against probes retrained on large and against the
  un-upgraded base.
- **Expected signal.** Transported probes within 2–4 points of retrained ones.
- **Runtime.** About 5 minutes.
- Not implemented.

**Wildness: 3.**

## 12. Description-anchored routes: probe weights from label descriptions

**Status: queued (`descanchor`).**
`explore/geometry_descanchor.py` trains one bilinear scorer over option-description anchors. It is
tested in-distribution (all 20 tasks) and zero-shot, leaving one workflow out, against uniform, a
majority oracle prior, untrained cosine and per-task probes.

**Mechanism.**
- **Encode descriptions.** Each option's description (typed-decisions ships them:
  `"stop": "Halt the agent now."`) goes through the same trunk once, giving pooled states at depth k.
- **Shared map.** One map M, low-rank bilinear and trained across many decisions, turns a description
  state into probe weights: `logit_c = pool(x)ᵀ M pool(desc_c)`.
- **New routes.** A new route from descriptions alone then costs one dot product per label on the shared
  pass: laya's flexibility at probe cost, but only as good as M's transfer.

**Novelty.** Done in vision and partly in NLP.
- **Generating classifier weights from class descriptions:** DeViSE (Frome et al. 2013) and Text2Model
  (Amosy et al. 2022, https://arxiv.org/abs/2210.15182).
- **The bi-encoder zero-shot literature** in general.

The version in which message and description share one frozen trunk tap is an application step.
Ideas 2 and 3 are the stronger read-once routes to the same goal.

**Cheap experiment.**
- **Setup.** typed-decisions, leave-one-workflow-out: train M on 15 tasks and score the 5 held-out tasks
  zero-shot.
- **Metric.** Accuracy against majority and against one-way laya (idea 2) on the held-out workflow.
- **Expected signal.** Above majority on score and yes/no questions, weak on choice questions.
- **Runtime.** About 5 minutes on the depth-scan features plus description encodings.
- Not implemented.

**Wildness: 2.**

---

## Status board (after the first GPU runs)

| # | idea | status | result so far |
|---|---|---|---|
| 1 | passenger tokens | tested / rerun queued | CLINC P4r8 86.7 (≈ blocks@11+2 87.0); typed under-trained (55.8 vs probe 65.7), rerun `passenger_typed_v2` |
| 2 | one-way laya | tested / retrain queued | 0.684 vs 0.767 unsplit, 0.453 block, 4.5× less compute per question; `oneway_train` (A100) |
| 3 | one-way cloze routes | tested | gap 4–6 pts on CLINC domain (prior-corrected), ≈0 on Banking77 best template |
| 4 | token-id-centred pooling | tested | refuted (−5 to +0.6 pts vs plain mean pool, mostly negative) |
| 5 | global-layer splits / forks | (a), (b) tested; (c) queued | no sawtooth; blocks@15/18+1 not special; `globalfork_*` |
| 6 | OOD from geometry | tested | cross-depth disagreement AUROC 0.914, 86.3% vs 81.8% majority, no oos data |
| 7 | decision atlas | zero-shot tested; few-shot queued | across-workflow Spearman 0.05–0.09 vs 0.68 within; `atlas` |
| 8 | cost-aware tap mixing | queued | multi-tap +5.3 / +3.5 / +0.5 pts over single depth; `tapmix` |
| 9 | shared decision subspaces | queued | `subspaces` |
| 10 | sufficient-statistics branches | queued | `sufficient` (smoke: merge/unlearn exact) |
| 11 | branch transport | ruled out by the user | none |
| 12 | description-anchored routes | queued | `descanchor` |

Queue: `explore/QUEUE_geometry.txt`.

## Ranked top 3 (novelty × feasibility × relevance to local-first routing)

1. **One-way questions on laya (idea 2).** It has the highest relevance: laya's any-question
   flexibility with the message encoded once and cached. It needs no training and runs in about 6
   minutes. Novelty is partial: MICE and sparse cross-encoders train asymmetric attention, and cbjev
   packs questions but lets the document see them. A zero-retraining conversion of a trained decision
   cross-encoder was not found. The early 60-pair signal is 0.700 vs 0.700 unsplit, against 0.333 for the
   block split.
2. **Passenger tokens with passenger-only LoRA (idea 1).** It is a new branch type whose marginal cost is
   P tokens, reads every depth and needs no split choice. Novelty is partial (RPO, VQT, aLoRA, Y-Tuning
   are close), and it overlaps with the systems lens's KV-query fork. The script is implemented, with
   parity checks and fair baselines.
3. **One-way cloze routes (idea 3).** It gives zero-shot routes with zero training on the same shared
   pass; the prompt is a passenger. Novelty is partial (PET, ModernBERT-Instruct, RPO; the decoder
   analogue is standard). This is feasible now. Its relevance depends on the one-way vs two-way gap
   being small; absolute accuracy needs an instruction-tuned MLM.

Runner-up for effort: the depth scan (ideas 4–7) is one script and about 5–8 minutes. It answers three
geometry questions at once:
- does token-id centring fix JSON pooling;
- is there a global-layer sawtooth;
- how well does frozen-trunk Mahalanobis detect oos with no oos data.

## Implemented scripts and commands

All ten scripts pass their `--smoke` runs on CPU (6–70 s each). Run from the repo root; the follow-up commands (name | GPU | minutes | command) are in `explore/QUEUE_geometry.txt`. The one custom training loop (`geometry_passenger.py`) now enforces >= 300 optimiser steps per task; the other new loops do too.

| script | full command | est. M6 runtime | writes |
|---|---|---:|---|
| `explore/geometry_oneway_laya.py` | `.venv/bin/python explore/geometry_oneway_laya.py --out results/tarski/explore_oneway_laya.json` | ~6 min | per (mask, k, head) accuracy, Brier, ECE, token-layers |
| `explore/geometry_passenger.py` | `.venv/bin/python explore/geometry_passenger.py --out results/tarski/explore_passenger.json` (or split: `--datasets typed-decisions --cloze --out results/tarski/explore_passenger_typed.json` and `--datasets clinc150 --out results/tarski/explore_passenger_clinc.json`) | ~25–30 min (or 2 × ~15) | parity; baselines (`probe@22`, `logreg@11/22`); passenger configs; cloze gaps |
| `explore/geometry_depthscan.py` | `.venv/bin/python explore/geometry_depthscan.py --out results/tarski/explore_depthscan.json` | ~5–8 min | ridge curves at 22 depths × {mean, cls, centred, multi-tap}, sawtooth, OOD table, urgency transfer |
| `explore/geometry_oneway_train.py` | see `explore/QUEUE_geometry.txt` (`oneway_train`) | ~25 min, A100 | untrained vs joint_ft vs oneway_ft vs oneway_qlora, each under one-way and joint masks |
| `explore/geometry_tapmix.py` | `tapmix` | ~12 min, A10 | single / prefix / group-lasso / cost-aware selection: accuracy vs trunk cut |
| `explore/geometry_subspaces.py` | `subspaces` | ~20 min, A10 | affinity (with same-messages null), shared bottlenecks, fused vs separate blocks@11+2 |
| `explore/geometry_sufficient.py` | `sufficient` | ~6 min, A10 | LDA/NCM vs logreg, merge and unlearn exactness, k-shot new labels, OOD |
| `explore/geometry_descanchor.py` | `descanchor` | ~6 min, A10 | shared description-anchored scorer: in-distribution and leave-one-workflow-out |
| `explore/geometry_atlas.py` | `atlas` | ~3 min, A10 | zero-shot and few-shot (atlas prior) concept transfer across workflows |
| `explore/geometry_globalfork.py` | `globalfork_typed`, `globalfork_banking77` | ~15 + 10 min, A10 | next / global-only / local-only layer forks at equal cost |

Implementation notes:
- **`Rider` (`geometry_passenger.py`).** It reimplements the ModernBERT layer loop so the message path
  runs once, with no gradients and under autocast. Each passenger block attends to that layer's message
  keys/values, and autograd holds one shared reference to them per layer. Parity: the message path equals
  `Trunk.taps` exactly, and passengers match a full-sequence one-way forward (relative difference 1.4e-6).
  The full-sequence masked forward equals the stock model.
- **Independent training.** Passenger configs and tasks train in one batch loop, but every parameter
  tensor has a task axis, gradients are not clipped jointly, and AdamW is per-parameter. Training them
  together is therefore the same as training each alone with the same data order.
- **Epoch selection** follows the fixed `tarski.train` rule: best validation epoch only with at least 50
  validation rows, otherwise the last epoch.
- **`geometry_oneway_laya.py`** reuses `readonce.core` (read-only import) for items, batching, laya's
  head and metrics. It checks that its block mask reproduces `readonce.core.encode_split` exactly, and
  that one-way state encodings do not depend on the question in front of them.
