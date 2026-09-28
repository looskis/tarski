# Exploration: systems, compilers, databases and caching

Lens: treat "N decisions about a message" as a query workload over a neural network. The trunk is a
storage engine that materialises views (hidden states, keys and values). Branches are queries. The
engine is a query planner that decides what to compute, what to reuse, and what to skip.

Written 2026-09-27; updated the same day.

- **Implemented:** every idea with a testable prediction (1–9 and 11) has a script in
  `explore/systems_*.py`, and every smoke test passes.
- **Tested:** ideas 1 (Banking77) and 2 have GPU results.
- **Queued:** everything else. The full-run commands are in `explore/QUEUE_systems.txt`.
- **Not run:** idea 10, which is already done in the literature.

Each idea has a status line under its heading.

Prior-art checks were quick (one or two web searches per idea). Verdicts are provisional.

**One finding for the main novelty note.** Du et al., "General Purpose Text Embeddings from
Pre-trained Language Models for Scalable Inference" (Findings of EMNLP 2020,
https://arxiv.org/abs/2004.14287) is missing from `research_notes/novelty_check.md`. The paper
does three things tarski also does:

- It shares one frozen encoder's features across many tasks.
- It stores those features with binary quantisation (16x smaller).
- It reports that sharing beats distillation once about 7 tasks are served.

The paper should be cited next to Wei et al. 2022 and Embedding Recycling.

---

## 1. KV-query decision streams ("fork the CLS token") — implemented

**Status: tested on Banking77; typed-decisions rerun queued (`kvquery_typed_fixed`).**

- **Banking77, split 8 (A10):** kvq:2:4 91.3 and kvq:4:4 91.5, against blocks:2 92.2 and probe 88.3.
  The kvq variants use 0.08 and 0.16 GFLOP per decision, against 0.26 for blocks:2 (at 13 tokens).
  kvq:4:4:frozen reached 88.6. kvq:4:1:frozen, a bare CLS probe at depth 12, reached 33.2: the CLS
  state alone is a poor readout, and the attention-pooled starts are what make it work.
- **typed-decisions, first run: confounded.** The kvq variants scored 49.5–54.3, against blocks:2 75.3
  and probe 62.7. But `train_kvq` ran about 50 optimiser steps per task, while the baselines' updated
  `train_branch` enforces at least 300. `train_kvq` now follows the same rule: at least 300 steps,
  best-validation selection only with 50 or more validation rows, and no early stop before half the
  schedule.
- **Latency on A10, 299-token message, fused branches only:**
  - N=20: kvq:2:1 32 ms, blocks:2 44 ms, probe 21 ms.
  - N=1: kvq:2:1 is slower, 22 ms against 20 ms. It runs one more trunk layer and recomputes the
    shared K/V instead of capturing them from the trunk pass.

**Mechanism.** A branch today copies d whole layers and re-runs them over every token, once per
decision. That costs N x d x L token-layers.

A KV-query branch forks only the CLS token at split k:

- **The stream.** Each decision gets a residual stream of r tokens: the CLS state, plus r-1 starts
  made by attention pooling. The stream runs through fine-tuned copies of base layers [k, k+d).
- **Read-only access.** At each layer, the stream's queries attend to the keys and values that the
  frozen base layer computes for the message tokens. Message tokens never attend to the stream, so
  their K/V are identical for every decision. The trunk's own pass already produces them once it
  runs past layer k.
- **Batching.** N decisions become one small multi-query attention per layer against a shared
  prefix. This is the LLM "many sequences sharing one prompt cache" pattern, applied to a
  bidirectional encoder with per-decision weights.
- **Cost.** Per decision per layer, the cost is O(r·L·D + r·D²) instead of O(L·D² + L²·D). At
  typed-decisions' median of 240 tokens, kvq:2:1 is ~0.02 GFLOP per decision against ~5.2 GFLOP for
  blocks:2, about 250x less.
- **Initialisation.** With r=1 and the base's own window masks, the stream exactly reproduces the
  trunk's CLS path at initialisation. The smoke test measures a relative error of 1.4e-7. So each
  decision starts as "the base model, continued for one token" and fine-tunes from there.
- **Why it could also help accuracy.** It lets decisions read deep layers cheaply. Attention
  readout may also avoid the probe collapse on typed-decisions: mean pooling over 240 JSON tokens
  washes out the one field a decision needs.

**Novelty.** Partially done.

- *Read-only prompts* (RPO, Lee et al., ICCV 2023, https://arxiv.org/abs/2308.14960). Prompts
  initialised from special tokens read the frozen encoder's tokens through masked attention at every
  layer, and the image tokens never read them. RPO's weights are frozen, its only trainable parts are
  prompt embeddings, and it serves one task on CLIP.
- *Visual Query Tuning* (Tu et al., CVPR 2023, https://arxiv.org/abs/2212.03220). Per-layer
  learnable query tokens summarise frozen intermediate features. The queries do not carry a residual
  stream across layers.
- *Attention probes.* EleutherAI blog (https://blog.eleuther.ai/attention-probes/); "Attention,
  Please! Revisiting Attentive Probing" (2025, https://arxiv.org/abs/2506.10178); Meyoyan & Del Corro,
  token- and layer-selective probes (ACL 2026, https://arxiv.org/abs/2601.13288).
- *Shared-prefix attention* for many decoding sequences: Hydragen (ICML 2024,
  https://arxiv.org/abs/2402.05099).

Not found anywhere: a per-decision *fine-tuned* single-token fork that exactly continues the base
CLS path, served as N decisions over one shared frozen KV, positioned against blocks branches on a
cost-per-decision axis.

**Experiment.** `explore/systems_kvquery.py`.

- **Data:** typed-decisions at split 11, with 20 tasks and 2 seeds. banking77 at split 8 as a
  second dataset.
- **Configs:** `probe` and `blocks:2` are re-trained in-script as baselines. The KV-query configs
  are:
  - `kvq:2:1`
  - `kvq:4:4`
  - `kvq:4:4:frozen`, which is RPO-like and trains only 0.01M parameters
  - `kvq:2:4`
  - `kvq:4:4:global`, which reads all tokens at local layers
  - `kvq:4:1:frozen`, which is a CLS probe at depth 15
- **Metrics:** acc, macro-F1, ECE, NLL, analytic GFLOP per extra decision, trainable parameters,
  and fused latency at N = 1, 5 and 20 against FusedBlocks.
- **Expected signal:**
  - On typed-decisions, kvq beats the 51–55% probes clearly. It is within a few points of blocks:2,
    at about 1/250 of the per-decision FLOPs.
  - Latency at N=20 is flat for kvq and grows with N for blocks.
  - banking77 checks that it does not lose on short text (probe@8 88.6, blocks@8+2 92.0).
- **Runtime:** under ~25 min on typed-decisions (`--budget-min 27` guards it), ~10 min on banking77.

**Wildness:** 3.

---

## 2. Incremental view maintenance for Slack threads — implemented

**Status: tested (A10).** CLINC150, 10-message threads, split 11.

| mode | trunk FLOPs vs re-encode | agreement with full re-encode | acc |
|---|---|---|---|
| stale | 15% | 95.8% | 84.5 |
| band16 | 35% | 98.9% | 85.7 |
| band32 | 53% | 99.4% | 85.7 |
| select10 (CacheBlend-style) | 30% | 96.6% | 84.7 |
| full re-encode (reference) | 100% | 100% | 85.9 |

Agreement and accuracy are for blocks:2 branches trained on full encodings.

- **Band vs select.** At similar cost, the window band beats deviation-based token selection.
- **Training on stale states.** Branches trained on stale states and served stale reach 86.5 at 15%
  of the FLOPs, above the full-trained branches on full re-encoding (85.9). I read that as parity:
  one seed, and the stale-trained branch also scores 86.8 on full encodings.

**Mechanism.** A thread grows one message at a time, and each new message needs its decisions.

- **The cost problem.** Re-encoding the whole thread each time costs O(T²) tokens over a thread. In
  a causal LM the prefix KV cache makes appends exact. In a bidirectional encoder, old tokens should
  attend to the new ones, so cached prefix states go stale.
- **The idea.** Treat the per-layer prefix states as a materialised view and maintain it
  incrementally:
  - `stale`: compute only the new tokens, which read the stale prefix K/V.
  - `bandW`: also recompute the W prefix tokens nearest the append. ModernBERT's local layers, two
    of every three, have a ±64-token window, so the dirty region of a local layer is a band.
  - `selectP`: CacheBlend-style. Run layer 0 exactly for all tokens, then recompute only the P% of
    prefix tokens whose layer-0 output moved most.
- **Train on what you serve.** A second set of branches trains on stale-mode states, to see whether
  matching training to the serving distribution removes any gap.
- **Metric.** What matters is whether the *decisions* change, not the hidden states.

**Novelty.** Partially done. The mechanism is known:

- **Selective KV recompute for decoders:** CacheBlend (EuroSys 2025,
  https://arxiv.org/abs/2405.16444).
- **Approximate KV reuse in bidirectional diffusion LMs:** Fast-dLLM (2025,
  https://arxiv.org/abs/2505.22618) and dLLM-Cache (2025, https://arxiv.org/abs/2506.06295).
- **Incremental document edits with VQ-sparsified transformers:** Sharir & Anandkumar (2023,
  https://arxiv.org/abs/2307.14988).
- **Restart-incrementality, which re-encodes everything, for bidirectional encoders in incremental
  NLU:** Madureira & Schlangen (EMNLP 2020, https://arxiv.org/abs/2010.05330).

Not found: append-only stale-KV reuse for an *encoder classifier*, the local-window band as the
recompute policy, and decision agreement as the target.

**Experiment.** `explore/systems_threadinc.py`.

- **Data:** CLINC150 threads of ~10 messages from one domain, with out-of-scope messages inserted
  at random. Every test message is decided at the step it arrives.
- **Branches:** probe and blocks:2 at split 11, pooling over the new message. They are trained on
  full encodings, and again on stale encodings.
- **Modes:** stale, band 16/32/64 and select 10/25%, each against full re-encoding.
- **Metrics:** acc, agreement with the full-encoding decisions, cosine of the pooled new-message
  state, and trunk FLOPs relative to a re-encode.
- **Expected signal:** stale mode keeps ≥97% agreement at ~0.2x FLOPs. Band and select close most
  of the remaining gap. Training on stale states closes the rest.
- **Runtime:** ~15 min.
- **Follow-up if it works:** combine with idea 1. A KV-query stream per new message reads the
  maintained K/V, so the whole per-message cost is the new tokens plus N one-token streams.

**Wildness:** 3.

---

## 3. Decision-set dead-code elimination: a JIT-specialised trunk per decision set

**Status: implemented (`explore/systems_dce.py`), smoke passes, queued (`dce`).**

**Mechanism.** For a fixed set S of requested decisions, much of the frozen trunk below their splits
may be irrelevant to them. This is dead code for this "program".

1. **Score.** Compute an importance score for each trunk attention head and each block of 64 MLP
   channels below max(split). Following Michel et al., the score is the gradient of a gate at 1 on
   the loss Σ_{s∈S} KL(branch_s(full trunk) ‖ branch_s(gated trunk)). It needs only unlabelled
   calibration messages and no labels.
2. **Prune.** Remove the lowest-scoring x%.
3. **Compile.** Keep the pruned trunk in a plan cache keyed by the hash of S, as a query compiler
   caches plans.

Branches are not retrained. The pruned trunk is valid only for S, so a request for a different set
falls back to the full trunk or its own plan.

**Novelty.** Probably new as a combination. Related work:

- Head importance and pruning: Michel et al., NeurIPS 2019 (https://arxiv.org/abs/1905.10650).
- Task-specific pruning with retraining: CoFi (ACL 2022), ALPS (2025,
  https://arxiv.org/abs/2505.18799), SHRP (2025, https://arxiv.org/html/2512.20635v1). All of these
  are per task and fine-tune the pruned model.

Not found: pruning a *shared frozen* backbone for a *set* of independently trained heads, without
retraining, cached per request signature.

**Experiment.**

- **Data:** CLINC150 (3 tasks) and banking77.
- **Branches:** blocks@11+2 trained once.
- **Sweep:** importance on 512 unlabelled train messages, then accuracy and decision agreement at
  10/20/30/40% head+MLP sparsity.
- **Baselines:** magnitude pruning, and importance computed for a *different* decision set. The
  second shows whether specialisation matters.
- **Expected signal:** 20–30% of trunk FLOPs removed at <1 point loss for set-specific plans, with
  clearly worse results for the other-set plans.
- **Runtime:** ~10 min.

**Wildness:** 3.

---

## 4. Schema-constant KV templates for JSON states (columnar encoding)

**Status: implemented (`explore/systems_schema.py`), smoke passes, queued (`schema`).** 44% of typed-decisions tokens are structure (measured).

**Mechanism.** typed-decisions states are JSON with one fixed key set per workflow. I measured this:

- 1–2 distinct key sets per workflow.
- **44% of tokens are structure** (keys and punctuation; median 0.44, p10 0.30, p90 0.55).

Treat the schema as a table header:

1. Precompute template K/V for structure tokens once per (workflow, layer). They can be averaged
   over training states, with positions handled by re-rotating RoPE per offset.
2. At serving time, encode only value tokens. Their queries read template K/V for the keys and fresh
   K/V for the values.

Trunk cost scales with value tokens, about 1.8x fewer. A cheaper first version encodes each field
value as its own short sequence (Block-Attention style) and lets a thin fusion layer or KV-query
branch combine them.

**Novelty.** Partially done. Related work:

- Prompt Cache: schema-defined reusable prompt modules with position-aware KV reuse, for decoders
  (MLSys 2024, https://arxiv.org/abs/2311.04934).
- Block-Attention (2024, https://arxiv.org/abs/2409.15355).
- EPIC and TurboRAG (position-independent caching).

Not found: schema-derived templates inside a bidirectional encoder for structured-state
classification.

**Experiment.** typed-decisions at split 11.

1. **Feasibility.** Replace structure-token states at every layer with per-workflow template states,
   by computing a mixed forward that is masked like `systems_threadinc.run_chains`. Measure the
   cosine to full states and the accuracy of blocks:2 and probe branches trained on full states.
2. **Train on templated states** and re-measure.

- **Baseline:** full encoding.
- **Expected signal:** less than a 2 point loss if keys carry little message-specific information.
- **Runtime:** ~10 min.

**Wildness:** 4.

---

## 5. Semantic query optimisation for decisions: conjunctive early exit and functional-dependency short-circuit

**Status: implemented (`explore/systems_planner.py`), smoke passes, queued (`planner`).**

A label-only finding already exists, computed on the full typed-decisions training labels: no
dependency among co-labelled typed-decisions tasks reaches 0.95 purity. The strongest are:

- `agent_trace.action → needs_review`: 0.945, against a 0.515 majority baseline.
- `invoice.discrepancy_severity → matches_order`: 0.931, against 0.517.

So exact short-circuiting is available on CLINC, where intent → domain and oos have purity 1.0, but
only approximate there.

**Mechanism.** A planner orders a request's decisions by cost and skips work in three ways.

- **Conjunctive early exit.** Run the trunk to the shallowest checkpoint (probes at depth 4). If
  *every* requested decision is confidently answered there (calibrated p ≥ τ), stop. Otherwise
  resume the trunk (a generator that keeps h and ctx) to the next checkpoint.
- **Functional-dependency short-circuit.** Learn from co-labelled training data which decisions
  determine others, such as intent → domain → oos on CLINC, or severity → urgency on
  typed-decisions. When P(B | Â) is peaked for the predicted A, answer B from the dependency table
  without running B's branch.
- **Conditional decisions.** Decisions such as "if category = billing, ask refund_eligible" become a
  DAG that the resumable trunk walks lazily.

**Novelty.** Low. Related work:

- Probabilistic Predicates (Lu et al., SIGMOD 2018) and CORE, which is correlation-aware proxy
  ordering (VLDB 2022, https://arxiv.org/abs/2201.00309), for ML inference queries.
- Per-instance early exit: DeeBERT (https://arxiv.org/abs/2004.12993) and Apparate (SOSP 2024,
  https://arxiv.org/abs/2312.05385).

The multi-decision conjunction on one trunk is a small twist.

**Experiment.**

- **Data:** CLINC150 (3 tasks).
- **Branches:** probe@4, probe@11 and blocks@11+2.
- **Metrics:** mean trunk depth, branches run per message, and accuracy over a τ sweep.
- **Baseline:** everything at blocks@11+2.
- **Expected signal:** 30–50% fewer trunk layers and ~2/3 fewer branch runs at <0.5 point loss.
  Domain from intent should be nearly free.
- **Runtime:** ~5 min.

**Wildness:** 2.

---

## 6. Speculate from cache, verify at a shallow layer: a near-duplicate decision cache

**Status: implemented (`explore/systems_speccache.py`), smoke passes, queued (`speccache`).**

**Mechanism.**

1. **Lookup.** Key each message by SimHash of its tokens. On a candidate hit, run the trunk only to
   layer v (2–3) and compare the pooled state to the cached layer-v state.
2. **Hit.** If cos ≥ τ, return all N cached decisions. The deep trunk and every branch are skipped.
3. **Miss.** Continue the pass from layer v. The work is not wasted, because the pass resumes.

Slack traffic has bots, alerts and templated messages, so hits should be common there.

**Novelty.** Low to moderate. Related work:

- Learned caches at intermediate layers: Balasubramanian et al. 2021 (https://arxiv.org/abs/2101.07344).
- Freeze Inference (HotCloud 2019).
- Semantic caches for LLMs: GPTCache.

New here: verifying with the same trunk's shallow prefix, and caching a whole *decision vector* per
entry.

**Experiment.**

- **Data:** CLINC150 test streamed in random order, with the cache filled from previously seen test
  messages and from train.
- **Metrics:** accuracy against hit rate for τ ∈ {0.95…0.999} and v ∈ {2, 4, 6}.
- **Baseline:** no cache.
- **Expected signal:** a knee where a 20–40% hit rate costs <0.3 points. CLINC understates real
  duplication.
- **Runtime:** ~3 min.

**Wildness:** 2.

---

## 7. Projection pushdown at the fork: shared token compaction before all branches

**Status: implemented (`explore/systems_compaction.py`), smoke passes, queued (`compaction`).** The uncompacted compact-cache path reproduces the plain branch exactly (abs diff 0).

**Mechanism.** At split k, compact the L tokens to r tokens once per message. Options:

- ToMe-style bipartite merge with size-weighted attention.
- A top-r scorer trained on the union of tasks.

All N block branches then run on r tokens. The compaction is paid once and amortised over N
decisions, so branch cost drops by L/r for every decision. On JSON states, punctuation and key
tokens are obvious merge candidates (see idea 4).

**Novelty.** Low to moderate. Related work:

- ToMe (ICLR 2023, https://arxiv.org/abs/2210.09461), with a BERT text-classification port.
- PoWER-BERT (ICML 2020).
- Learned token pruning (LTP, KDD 2022).
- Meyoyan & Del Corro 2026, token-selective probes.

The "one compaction amortised over N branches" framing is the only new part.

**Experiment.**

- **Data:** typed-decisions, blocks@11+2 with r ∈ {240 (none), 128, 64, 32}.
- **Metrics:** mean acc and branch GFLOPs.
- **Baseline:** uncompacted blocks.
- **Expected signal:** r=64 within 1 point, at ~4x lower branch cost.
- **Runtime:** ~10 min.

**Wildness:** 2.

---

## 8. Workload-aware joint split planning ("free depth promotion")

**Status: implemented (`explore/systems_jointsplit.py`), smoke passes, queued (`jointsplit`).** It uses one-pass probe curves, so it plans for probe branches.

**Mechanism.** The trunk runs to the deepest split any requested decision needs, so per-task split
choice is the wrong optimisation. The planner should instead optimise the workload.

- **Free promotion.** Given co-request statistics (typed-decisions requests all 5 tasks of a
  workflow together), each task can move up to the request's maximum depth for free, picking its best
  split ≤ that depth.
- **Pull down outliers.** If one task forces depth 18, test pulling it to 11: the saving is 7 trunk
  layers for the whole request.
- **Solve jointly.** Choose all splits together, minimising E_request[max split] + Σ branch cost
  subject to an accuracy-loss budget. This is a small DP over the autosplit curves.

**Novelty.** Low. Mainstream's scheduler jointly picks branch points under a shared-stem compute
budget (USENIX ATC 2018, https://www.usenix.org/conference/atc18/presentation/jiang). Wei et al.
2022 threshold marginal gains. The per-request max-depth objective is a detail.

**Experiment.**

- **Inputs:** no training beyond `autosplit` curves; can use existing `sweep_*.json` for
  typed-decisions and CLINC.
- **Metric:** expected trunk layers per request against mean accuracy, comparing per-task autosplit
  with the joint plan.
- **Expected signal:** the same accuracy with 20–40% fewer trunk layers when co-requested tasks have
  mismatched curves.
- **Runtime:** ~5 min, most of it the probe curves.

**Wildness:** 1.

---

## 9. Base-product sharing: delta-coded branches ride on the trunk's own matmuls

**Status: implemented (`explore/systems_deltabranch.py`), smoke passes, queued (`deltabranch`).** The shared-product fused path matches the looped LoRA branches (max abs diff 1.9e-6).

**Mechanism.** A branch forked at k with init "next" is base layer k plus a delta.

- **Shared products.** When the trunk continues past k for a deeper decision, it already computes
  LN_k(h_k)·Wqkv_k and the MLP input product. If branch deltas are low-rank (LoRA) or 1-bit (BitDelta)
  and the branch LayerNorms stay frozen, each branch's first-layer input-side products become
  "trunk product + small correction". That removes roughly the QKV and Wi FLOPs of every branch's
  first layer.
- **Storage.** Branch files shrink ~10x as deltas.

**Novelty.** Low. Related work:

- S-LoRA (https://arxiv.org/abs/2311.03285) and Punica share base computation across adapters.
- BitDelta (NeurIPS 2024, https://arxiv.org/abs/2402.10193), Delta-CoMe and DARE compress deltas.

The observation that the trunk's continuation *is* the base computation for every branch at k is a
cute special case.

**Experiment.**

- **Data:** banking77 blocks@8+2.
- **Branches:** full-copy branches against LoRA r=16 with frozen norms, plus post-hoc BitDelta of
  the full-copy branches.
- **Metrics:** acc, file bytes, and fused latency at N=20 with shared products.
- **Expected signal:** ≤0.5 point loss, ~10x smaller files, 20–30% lower branch latency when the
  trunk runs deeper anyway.
- **Runtime:** ~10 min.

**Wildness:** 2.

---

## 10. Compressed trunk-state views for backfill (already done)

**Status: not run. Already done in the literature:** Du et al. 2020 (binary quantisation of stored multi-task features, 16x smaller); ColBERTv2 residual compression; Embedding Recycling. Measuring it for block branches would be a replication, so it has lower priority than the rest.

**Mechanism.** Store token-level trunk states at a canonical depth for historical messages, in
compressed form (int4, 1-bit, or ColBERTv2-style centroid + residual codes). A decision added weeks
later can then be trained and backfilled over history without re-running the trunk. On a laptop,
storage is the constraint: fp16 is 1.5 KB per token per depth.

**Novelty.** Already done:

- Du et al. 2020 (https://arxiv.org/abs/2004.14287): binary quantisation of stored multi-task
  features, 16x smaller.
- Embedding Recycling (EACL Findings 2023, https://arxiv.org/abs/2207.04993).
- ColBERTv2 residual compression, 1–2 bits per dimension (https://arxiv.org/abs/2112.01488).

It is still worth measuring for *block* branches, which have to run transformer layers on
decompressed states.

**Experiment.**

- **Data:** banking77 and typed-decisions.
- **Setup:** a FeatureCache subclass that quantises on insert (int8, int4, 1-bit with scale, PQ
  with 256 centroids plus 2-bit residual). Train probe@k and blocks@k+2 on each.
- **Metric:** acc against bytes per token.
- **Expected signal:** int4 is free; 1-bit costs <1 point for probes and more for blocks.
- **Runtime:** ~10 min.

**Wildness:** 1.

---

## 11. Layer-granular continuous batching for heterogeneous-depth requests

**Status: implemented (`explore/systems_layerbatch.py`), smoke passes, queued (`layerbatch`).** It executes real trunk layers on the device with a virtual clock and simulated Poisson arrivals.

**Mechanism.** Concurrent requests need different trunk depths.

- Instead of batching whole requests, the scheduler batches *layers*. A message that only needs
  depth 8 leaves the batch at layer 8.
- New arrivals join at layer 0 in the slot it freed. This is Orca's iteration-level scheduling, with
  "iteration" meaning a trunk layer.
- Branch work for messages that finish joins a fused branch batch.

**Novelty.** Low. Related work:

- Orca (OSDI 2022).
- DVABatch: multi-entry multi-exit batching (ATC 2022).
- Apparate (SOSP 2024).

It is also of limited value on a single-user laptop. It matters for a workspace-level router that
handles bursts.

**Experiment.**

- **Setup:** simulate Poisson arrivals with depth requests drawn from the typed-decisions and CLINC
  mix, and measure real per-layer latency on MPS.
- **Metric:** p50 and p99 latency against throughput.
- **Baseline:** static batching to the maximum depth.
- **Runtime:** ~5 min.

**Wildness:** 2.

---

## Ranking (novelty x feasibility x relevance to local-first routing)

1. **KV-query decision streams (idea 1).**
   - Novelty: partial. Read-only queries exist in vision (RPO, VQT) and as attention probes. The
     fine-tuned single-token fork with exact CLS continuation, served as N decisions over one shared
     frozen KV, was not found.
   - Feasibility: high (implemented).
   - Relevance: highest. It makes the marginal decision nearly free even on 240-token JSON, and may
     fix the typed-decisions probe collapse.
2. **Thread-incremental encoding (idea 2).**
   - Novelty: partial. The mechanism is CacheBlend or Fast-dLLM applied to an encoder classifier,
     with a window-derived band and decision agreement as the target.
   - Feasibility: high (implemented).
   - Relevance: high for Slack threads, and it composes with idea 1.
3. **Decision-set dead-code elimination (idea 3).**
   - Novelty: likely new as a combination.
   - Feasibility: medium (needs trained branches and gate gradients).
   - Relevance: medium to high for fixed routing workloads.

Runner-up: schema templates (idea 4). The 44% structure-token measurement makes it concrete, and it
is the most novel. It is also the most work.

## Scripts and queue

Every script lives in `explore/` and imports shared helpers from `explore/systems_common.py`.

- **Smoke tests.** Each has a `--smoke` CPU mode on tiny subsets, and all ten pass in 2–25 s.
  Accuracies in smoke mode are meaningless (tiny data, a few steps).
- **Full runs.** Commands are in `explore/QUEUE_systems.txt`, in the form
  `name | GPU | est. minutes | command`. Each writes `results/tarski/explore_<name>.json` plus a `.log`.
  All fit an A10 (24 GB).
- **Training rule.** Every branch trains through `tarski.train.train_branch` (at least 300 steps) or
  through `train_kvq`, which uses the same rule.

| idea | script | built-in check | queue name (A10 est. min) |
|---|---|---|---|
| 1 | `systems_kvquery.py` | stream = trunk CLS path at init (rel err 1.4e-7); fused = loop (1e-7) | `kvquery_typed_fixed` (25) |
| 2 | `systems_threadinc.py` | masked incremental path with all tokens active = plain encode (0) | done (lambda run) |
| 3 | `systems_dce.py` | unpruned plan reproduces branch accuracy | `dce` (8) |
| 4 | `systems_schema.py` | train-only signatures; templated fraction reported | `schema` (20) |
| 5 | `systems_planner.py` | offline over calibrated test probabilities | `planner` (10) |
| 6 | `systems_speccache.py` | cache seeded with train decisions, test streamed in random order | `speccache` (8) |
| 7 | `systems_compaction.py` | uncompacted compact path = plain branch (0) | `compaction` (25) |
| 8 | `systems_jointsplit.py` | val for choices, test for reporting | `jointsplit` (6) |
| 9 | `systems_deltabranch.py` | shared-product fused = looped LoRA (1.9e-6) | `deltabranch` (6) |
| 10 | not implemented | already done in the literature | not run |
| 11 | `systems_layerbatch.py` | real layer execution, virtual clock | `layerbatch` (8) |
