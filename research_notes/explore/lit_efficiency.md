# Literature scan: efficiency and adaptation techniques for the tarski harness

Written 2026-09-27. Lens: recent (2025–2026) work on caching, token reduction, adaptive depth,
quantisation, classifier generation, parameter-efficient branches, multi-task serving and test-time
adaptation. The scan looked for ideas that plug into "one frozen ModernBERT-base trunk, many small
per-decision branches". It prefers ideas that cut laptop/CPU latency or memory, or that let a user add a
decision with fewer labels.

**Already covered elsewhere, so not re-proposed:**
- ToMe at the fork (systems 7, `systems_compaction.py`).
- Delta/LoRA/BitDelta branches (systems 9, `systems_deltabranch.py`).
- Compressed trunk-state views (systems 10), thread KV reuse including dLLM-Cache and Fast-dLLM
  (systems 2), and near-duplicate decision caches (systems 6).
- Dead-code pruning (systems 3), anytime bundles (farfield 5) and per-instance depth gating (skeptic 6).
- LDA/sufficient-statistics heads (geometry 10), description-anchored routes (geometry 12) and weight
  imprinting (learning H).
- Text-to-LoRA, Doc-to-LoRA, SHINE, Drag-and-Drop, TinyLoRA, VeRA, LoRA-XS, aLoRA and UpgradeBench
  (`Appending layers to language models/frontier_methods.md`).

Where an entry below touches one of these, it says what is different.

**Method.**
- **Search.** Web search plus arXiv, ACL Anthology and GitHub pages. Every URL below was opened or
  returned by search during this scan.
- **Reading depth.** Abstract-level for most papers. FastFit, MotherNet, TabICL's README, the TabPFN text
  adapter and ModernBERT-AppleNeuralEngine were read in more detail.
- **CPU checks.** Two sanity checks on the real trunk support entries 1–3. They are scratch scripts, not
  added to the repo; the recipes are given inline so they can be re-run. They used `device="cpu"` and
  ran under 2 minutes each, and no GPU jobs were run.

## Ranked summary

| # | Idea | Main source (year) | Helps | Evidence so far | Novelty |
|---|---|---|---|---|---|
| 1 | **Shadow trunk:** static token vectors fitted to reproduce the split-depth pooled state, giving every probe a zero-trunk tier, with risk-controlled exits | Tokenlearn/Model2Vec (2024–25), Jazbec et al. NeurIPS 2024, UAT NeurIPS 2025 | CPU latency | **CPU check:** Banking77 probe@4 scores 88.8 on the shadow vs 89.0 on the real trunk; 76% of messages exit at tier 0 with 0.8% flips | Partial |
| 2 | **Split off ModernBERT's massive channels, rotate, then quantise** states, K/V and the trunk | Massive Activations (2024), QuaRot/SpinQuant (2024–25), TurboQuant (2025), ModernBERT-ANE repo | Memory, CPU/ANE | **CPU check:** channel 251 reaches 35,087 at depth 18. Plain per-token int8 has 42–49% error on the other channels at depths 14–18; with 6 fp16 channels plus rotation it is 0.7–0.9% | Low (application) |
| 3 | **Cold-start selection** of which messages to label for a new decision | TypiClust ICML 2022, ProbCover NeurIPS 2022, Rauch et al. 2025 | Fewer labels | **CPU check:** Banking77 at 1/2/5 labels per class: random 20.1/29.4/50.2 vs typicality 28.0/41.7/60.6 | Low (not in notes) |
| 4 | **In-context tabular foundation models as zero-gradient branches** (TabICLv2, TabFlex, MotherNet) | TabICLv2 ICML 2026, MotherNet ICLR 2025, TabFlex ICML 2025 | Fewer labels, no training loop, calibration | None yet; one related negative (text adapter < PCA) | Partial |
| 5 | **Late-interaction (MaxSim) branches seeded from label names** | FastFit NAACL 2024 | Fewer labels, zero-shot init | None yet | Partial |
| 6 | **Trunk on the Neural Engine, branches on GPU/CPU** | ModernBERT-ANE repo (2025), ANE measurement papers (2026), HeteroInfer (2025) | Laptop latency and energy | **Measured on the Mac:** trunk to depth 8 in 0.81 ms (fp16) or 0.57 ms (int8 weights) on the ANE vs 8.3 ms on PyTorch MPS; Banking77 decisions 99.8–99.9% identical in fp16; int8 weights break at depth 14 | Engineering |
| 7 | **Singular-value-only (SVF) branches:** ~6k params per 2-layer branch, shared-weight batching across N | Transformer² ICLR 2025 | Branch memory, N-way batching | None yet | Partial |
| 8 | **Unary-score token reduction instead of pairwise ToMe**, at the fork and inside the trunk | CATIS (2026) | Branch/trunk FLOPs on JSON | None yet | Low (an arm on systems 7) |
| 9 | **Backprop-free closed-form test-time refinement** of LDA branches from unlabelled traffic | ADAPT NeurIPS 2025 | Fewer labels | None yet | Low |

Entries 1–3 already have a measured signal on this trunk and cost minutes, so they should be queued
first. Entry 4 is the most interesting untested bet for "add a decision with few labels".

## Implementation status (2026-09-27)

**Code.**
- Scripts are `explore/lit_eff_<idea>.py`, with shared helpers in `explore/lit_eff_common.py`.
- Every script has a `--smoke` mode. All eight pass on CPU, each in under 70 s.
- Training uses `tarski.train.train_branch`, which enforces ≥300 steps, no stopping before half the
  schedule, and the last epoch when there are fewer than 50 validation rows. Few-label runs use no
  validation rows at all.
- GPU runs are queued in `explore/QUEUE_lit_eff.txt` (16 jobs, each ≤25 min on an A10; the many-class
  TabICL job is marked A100).

**Dependencies.** Added with `uv add`, so `pyproject.toml` and `uv.lock` changed:
- `tabicl` (BSD-3-Clause, code and checkpoint);
- `coremltools` (Mac only in practice).

The Lambda instance needs `uv sync` (or `experiments/lambda.sh setup`) before the `lit_icl_*` jobs. Those
jobs run with `HF_HUB_OFFLINE=0` so they can fetch the 110 MB TabICL checkpoint once.

| # | Script | Queue entries | Status |
|---|---|---|---|
| 1 | `lit_eff_shadow.py` | `lit_shadow_{banking77,clinc150,typed}` | Queued. Mac CPU latency measured (below) |
| 2 | `lit_eff_quant.py` | `lit_quant_{banking77,clinc150,typed}` | Queued. Mac CPU int8 latency measured (below) |
| 3 | `lit_eff_coldstart.py` | `lit_coldstart_{banking77,clinc150,typed}` | Queued. Covers CLINC bundles, typed workflows and blocks:2 |
| 4 | `lit_eff_icl.py` | `lit_icl_{typed,clinc150,manyclass}` | Queued. TabICLv2 only (see entry 4). Mac CPU latency measured |
| 5 | none | none | Skipped: the decision-models agent is implementing FastFit-style late interaction |
| 6 | `lit_eff_ane.py` | Mac only | **Done on the Mac** (results in entry 6) |
| 7 | `lit_eff_svf.py` | `lit_svf_banking77`, `lit_svf_clinc_typed` | Queued. Mac MPS latency measured (entries 6 and 7) |
| 8 | `lit_eff_unary.py` | `lit_unary` | Queued. A separate arm; `systems_compaction.py` is not modified |
| 9 | `lit_eff_tta.py` | `lit_tta` | Queued |

---

## 1. Shadow trunk: a bag-of-tokens tier that reuses every existing probe, with risk-controlled exits

**Status.** Implemented as `explore/lit_eff_shadow.py`; smoke passes; queued as three per-dataset jobs.
- **Shadows.** Two are fitted on unlabelled training messages:
  - `tokmean`, the per-token mean state. The served ProbeBranch runs on it unchanged.
  - `poolfit`, a ridge fit to the pooled state.
- **Tier-0 candidates.** The same probe on the shadow, a logistic head retrained on the shadow, and a
  TF-IDF control.
- **Served branches.** probe, logreg and blocks:2 (blocks on Banking77 and CLINC only).
- **Thresholds.** Learn-then-Test on validation, per bundle.
- **Validation-size limit.** typed-decisions has only ~108 validation messages, too few to certify
  δ = 1% (that needs n ≳ 300 with zero flips). Expect "not certifiable" there; the fixed-τ rows still
  apply.
- **Mac CPU latency** (batch 1, 8 threads, load average 3–4 from another agent's jobs, so trunk times are
  inflated):

| Dataset, depth | Trunk | Tier 0 |
|---|---:|---:|
| Banking77, 4 | 4.1 ms | 0.009 ms |
| Banking77, 8 | 12.9 ms | 0.012 ms |
| CLINC, 6 | 8.4 ms | 0.013 ms |
| CLINC, 11 | 15.0 ms | 0.012 ms |
| typed-decisions, 11 (108 tokens) | 37.8 ms | 0.019 ms |

Tier 0 is roughly three orders of magnitude cheaper than running the trunk. Result:
`results/tarski/explore_lit_shadow_latency_mac.json`.

**Sources.**
- Minish Lab, Model2Vec and Tokenlearn/POTION.
  - Code: https://github.com/MinishLab/model2vec and https://github.com/MinishLab/tokenlearn.
  - Method post: https://minishlab.github.io/tokenlearn_blogpost/.
  - Tokenlearn trains static token vectors so that their pooled output matches a sentence transformer's
    mean output on ~1M C4 documents. potion-base-8M scores 64.44 on MTEB classification.
  - These are not peer reviewed.
- Aarsen, "Train 400x faster Static Embedding Models with Sentence Transformers", Hugging Face blog,
  15 Jan 2025, https://huggingface.co/blog/static-embeddings. It reports ~397x faster CPU inference and
  87.4% of all-mpnet-base-v2's retrieval quality.
- Jazbec et al., "Fast yet Safe: Early-Exiting with Risk Control", NeurIPS 2024,
  https://arxiv.org/abs/2405.20915. Exit thresholds are tuned post hoc so that a risk (for example,
  disagreement with the full model) is bounded with high probability; the method is distribution-free.
- Bajpai et al., "Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability"
  (UAT), NeurIPS 2025, https://arxiv.org/abs/2509.23666. It adapts exit thresholds online with a bandit
  when traffic drifts.

**Idea to borrow.** Tokenlearn's objective, with tarski's own trunk as the teacher. Fit one static vector
per token so that the mean of a message's token vectors reproduces the trunk's mean-pooled state at split
depth k. This is a sparse least-squares problem on unlabelled messages; no labels and no new heads are
needed.

- **Tier 0.** Every probe already trained at depth k runs unchanged on the shadow state, so the shadow
  becomes a zero-trunk tier for all probe decisions.
- **Exits.** A bundle exits at tier 0 only if every requested decision is confident. Thresholds are chosen
  with risk control, so the flip rate against the served branch is at most δ with probability 1−ε.
- **Cost.** A sparse gather of ~15 vectors plus a dot product per decision, i.e. microseconds on CPU.

**CPU check (done, Banking77, depth 4 and 8).**
- **Recipe.**
  1. Mean-pooled trunk states for train and test.
  2. Standardise with train statistics.
  3. Fit an sklearn logistic probe on the real train states.
  4. Fit static vectors E (3,005 seen tokens x 768) with `scipy.sparse.linalg.lsqr` (damp 0.05) on the
     length-normalised count matrix of the train messages. The fit uses unlabelled messages only.
  5. Evaluate the same probe on `C_test @ E`.
- **Result.** The whole run took 94 s, 40 s of it encoding.

| depth | probe on real states | same probe on shadow | argmax agreement | tier-0 exit rate at τ = 0.9 | flips vs real probe | cascade acc |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 89.0 | 88.8 | 91.8% | 76% | 0.8% | 89.1 |
| 8 | 88.8 | 88.7 | 90.2% | 76% | 1.3% | 89.4 |

A probe trained directly on shadow features scores 88.1 at depth 4. That is close to the same probe
applied to the shadow (88.8), so the shadow keeps almost everything the probe uses.

**Implication for the skeptic lens.** On Banking77, a probe on the frozen trunk is effectively a
bag-of-words classifier. Banking77 probe numbers therefore say little about contextual value; the gap to
blocks (92–93) is what measures it.

**Experiment.** `explore/eff_shadow.py`.
- **Shadow fits.** One per depth, on the unlabelled train pool, for Banking77, CLINC150 (3 decisions) and
  typed-decisions.
- **Branches served.**
  - Probes at their autosplit depth.
  - Blocks branches, with the shadow tier using a probe as a proxy and exits controlled against the
    blocks output.
- **Thresholds.** Learn-then-Test on validation for δ ∈ {0.5, 1, 2}% and ε = 0.05.
- **Metrics.**
  - Bundle exit rate, accuracy and flips vs the served branch.
  - Measured Mac CPU latency per message: tier 0 vs trunk to depth k.
- **Baselines.**
  - Always run the trunk.
  - A TF-IDF + logistic tier 0, which tests whether fitting to the trunk beats plain bag-of-words.
  - `skeptic_depth_gating.py` (probe@4 → deeper) as the competing cheap tier.
- **Expected signal.**
  - Banking77: 60–75% bundle exits at ≤1% flips vs the probe, and fewer vs blocks.
  - CLINC: lower, because the oos decision holds the bundle back.
  - typed-decisions: low (<20%), because JSON meaning depends on structure.
- **Speed.** A 4–10x average CPU latency cut for Banking77-like traffic.
- **Cost.** Encoding ~2 min on an A10 (or reuse `FeatureCache`); fitting under 1 min CPU per depth;
  latency runs 5 min on the Mac CPU.

**Novelty.** Partial.
- Known: static distillation (Model2Vec, Tokenlearn), cascades and risk-controlled exits.
- Not found in a quick search: a static shadow of a *shared trunk's intermediate state*, so that all
  existing probes gain a zero-trunk tier with no retraining, gated per bundle with a flip-rate guarantee.
- Related in the notes:
  - `systems_speccache.py` has a lexical nearest-neighbour cache that reuses whole decision vectors, not
    a regression into trunk space.
  - `systems_schema.py` templates JSON structure tokens inside the trunk.

**Solidity.** The Model2Vec/Tokenlearn claims are READMEs and blogs, but our check reproduces the core
effect directly. Jazbec et al. is peer reviewed and the risk-control machinery is standard.

**Risks.**
- Out-of-vocabulary tokens in new traffic. They get zero vectors, which lowers confidence, so they should
  fall through to the trunk, but this needs checking.
- The shadow must be refitted if the split depth changes.

---

## 2. Split off the massive channels, rotate, then quantise: cached states, K/V, and the trunk itself

**Status.** Implemented as `explore/lit_eff_quant.py`; smoke passes; queued per dataset.
- **Datasets and splits.** Banking77 at 8 and 14, CLINC at 11, typed-decisions customer_service at 14.
- **Part B, cached-state schemes.** fp16, int8 plain / split / split+rotation, int4 plain / split /
  split+rotation, and int3 split+rotation.
  - fp16-trained probe and blocks:2 are scored on every scheme.
  - They are also retrained on int8_plain, int4_split_rot and int3_split_rot (only the first two on
    CLINC).
- **Part C, quantised trunk end to end** on all three datasets:
  - w8, w4g64, w8a8, and w8a8 with LLM.int8-style fp outlier inputs;
  - K/V quantisation: KIVI at 4 and 2 bits, per-head rotation at 4 and 3 bits.
- **Smoke check** (tiny subsets, so ignore accuracy): outliers are split off automatically (channels
  251, 254, 67, 142, 689, …). State error on normal channels: int8_plain 27–45%, int8_split_rot 0.7%,
  int4_split_rot 13–14%.
- **Mac CPU, part D** (`results/tarski/explore_lit_quant_cpu_mac.json`). PyTorch dynamic int8 (qnnpack)
  is only 1.16–1.31x faster than fp32 at 16 tokens, and *slower* at 64 tokens (0.76–0.83x) and 240
  tokens (0.61–0.78x), because fp32 runs on Accelerate. On this Mac, int8 pays off through Core ML on the
  ANE (entry 6), not PyTorch CPU kernels.
- **Found in entry 6.** int8 *weights* on the ANE break the trunk at depth 14: non-finite states in 2,868
  of 3,080 messages. The fp16-weight model at the same depth is fine. That is the massive-channel problem
  this entry is about.

**Sources.**
- Sun, Chen, Kolter, Liu, "Massive Activations in Large Language Models", COLM 2024,
  https://arxiv.org/abs/2402.17762. A few fixed channels carry input-independent huge values that act as
  bias terms.
- Bondarenko, Nagel, Blankevoort, "Quantizable Transformers: Removing Outliers by Helping Attention Heads
  Do Nothing", NeurIPS 2023, https://arxiv.org/abs/2306.12929. BERT outliers come from no-op attention
  heads.
- Ashkboos et al., "QuaRot", NeurIPS 2024, https://arxiv.org/abs/2404.00456. Liu et al., "SpinQuant",
  ICLR 2025, https://arxiv.org/abs/2405.16406. Orthogonal rotations remove outliers, allowing 4-bit
  weights, activations and KV.
- Zandieh, Daliri, Hadian, Mirrokni, "TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate", arXiv Apr 2025, https://arxiv.org/abs/2504.19874.
  - Data-oblivious: a random rotation followed by per-coordinate optimal scalar quantisers, plus a
    1-bit QJL residual for unbiased inner products.
  - Reports KV "quality neutrality" at 3.5 bits per channel. No calibration is needed.
- Liu et al., "KIVI", ICML 2024, https://arxiv.org/abs/2402.02750: keys per channel, values per token,
  at 2 bits. "RotateKV", 2025, https://arxiv.org/abs/2501.16383: outlier-aware rotations for 2-bit KV.
- smpanaro, **ModernBERT-AppleNeuralEngine** (GitHub), https://github.com/smpanaro/ModernBERT-AppleNeuralEngine.
  ModernBERT's 20–30k outlier activations break fp16 on the ANE, and QuaRot/SpinQuant-style rotations fix
  fidelity.
- Apple, "Apple Intelligence Foundation Language Models Tech Report 2025",
  https://arxiv.org/abs/2507.13575. After 2-bit QAT of the base, task adapters also serve as
  "accuracy-recovery" adapters that absorb the quantisation error.
- Xia, Cui, Cao, "Efficient INT8 Inference of Small NLP Models on Server CPUs with PyTorch Native Stack",
  arXiv Aug 2026, https://arxiv.org/abs/2608.18182. SmoothQuant in TorchAO gives up to 5.8x throughput
  for BERT, DistilBERT and XLM-R on Xeon with negligible loss. It does not cover ModernBERT.

**CPU check (done: 64 Banking77 test messages, and 32 messages from each of Banking77 and
typed-decisions).**
- **Where the outliers are.** ModernBERT-base's residual stream has fixed massive channels, the same on
  both datasets:
  - Channel 251 is the largest at every depth except the last. It reaches 1,010 at depth 8, 13,311 at
    depth 14 and 35,087 at depth 18, against a 99.9th percentile of 20–485.
  - Channels 254, 67, 142, 689 and 606 follow.
- **Quantisation error.** Relative error on the *other* channels, per-token absmax scalar quantisation:

| depth | plain int8 | 6 channels kept fp16 + int8 | + random rotation, int8 | + rotation, 4-bit | + rotation, 3-bit |
|---|---:|---:|---:|---:|---:|
| 4 | 8.0% | 1.2% | 0.8% | 13.9% | 32% |
| 14 | 49.1% | 1.8% | 0.9% | 16.0% | 38% |
| 18 | 42.1% | 3.0% | 0.9% | 16.2% | 37% |
| 22 | 6.5% | 1.2% | 0.8% | 13.9% | 32% |

Rotation alone, without the split, does not rescue 4-bit at depths 14–18: the relative error exceeds
100%, because rotation smears a 35k value into every coordinate. The split is what matters. The 4-bit
column uses uniform levels; TurboQuant's Beta-optimal levels should do better.

**Idea to borrow.** Three steps, each almost free here:
1. Keep the handful of fixed massive channels in fp16, which is 1.6% of the width.
2. Rotate the rest (random orthogonal or Hadamard, as in TurboQuant or QuaRot), then quantise per token.
3. Train branches on the dequantised states they will be served. Because branches already train on
   cached states, this makes every branch an accuracy-recovery adapter for free (AFM's framing).

**Where it applies.**
- `FeatureCache`: fp16 costs 1.5 KB per token per depth; int8 halves that and 4-bit cuts it ~3.8x.
- The Slack-thread K/V cache and KV-query streams: 67.6 KB per token across 22 layers in fp16.
- Backfill views (systems 10).
- A W8A8 or weight-only int8 trunk on CPU. Short messages (Banking77 and CLINC are ~15 tokens) are
  weight-bandwidth-bound at batch 1, so weight-only int8/int4 should speed the CPU up even without int8
  activations.
- The ANE (entry 6).

**Experiment.** `explore/eff_quantstates.py`.
- **Cache arms.** For k ∈ {8, 11, 14}: plain per-token, split, and split+rotation, at 8, 4 and 3 bits,
  plus a TurboQuant-style Beta-optimal 3.5-bit arm.
- **Training.** Train probe and blocks:2 on each arm. Also train on fp16 and serve quantised, to size the
  mismatch.
- **K/V arm.** In `systems_kvquery.py`-style streams, quantise the trunk K/V: KIVI per-channel keys vs
  rotation + per-token.
- **Datasets.** Banking77 and typed-decisions.
- **Baseline.** The fp16 cache.
- **Expected signal.**
  - Plain int8 costs points at splits 14–18 and is fine at 4–11.
  - Split+rotation int8 is indistinguishable (<0.2 points).
  - Split+rotation 4-bit is within ~0.5 points for probes and ~1 point for blocks, mostly recovered by
    training on quantised states.
  - K/V at 3.5 bits is decision-neutral.
- **CPU latency arm.** On the Mac CPU, weight-only int8 trunk (TorchAO, or ONNX Runtime if ARM kernels
  are missing) at 16, 64 and 240 tokens.
- **Cost.** ~20 min on an A10 for training; 10 min on the Mac CPU for latency.

**Novelty.** Low. Every technique is known. What looks unreported in a quick search:
- the ModernBERT-specific measurement (fixed channels, magnitudes by depth);
- quantising *cached encoder states that branches are trained on*;
- the split-then-rotate recipe for this trunk.

**Solidity.**
- QuaRot, SpinQuant and KIVI are peer reviewed and widely reproduced.
- TurboQuant is arXiv-only; its KV claims are on LLM long-context tasks.
- The ModernBERT-ANE outlier claim is from a hobby repo, but it matches our measurement.

---

## 3. Cold-start selection: which messages should the user label first for a new decision?

**Status.** Implemented as `explore/lit_eff_coldstart.py`; smoke passes; queued per dataset.
- **Selectors.** random (3 seeds), TypiClust, k-means medoid (2 seeds each), and ProbCover (a greedy
  sparse-graph version; δ = twice the median nearest-neighbour distance, a stand-in for the paper's
  purity rule).
- **Learners.** logreg, LDA, probe and blocks:2. None uses validation data.
- **Budgets.**
  - Banking77: 77, 154, 385, 770.
  - CLINC: 150, 300, 750 labelled messages, each carrying all three decisions (the bundle case).
  - typed-decisions: 10 and 40 messages per workflow, each carrying 5 questions.
- **Where blocks:2 runs.** Banking77 {77, 385}, CLINC {150, 750} and typed customer_service {40}, with
  random, TypiClust and ProbCover.
- **Speed.** Selection on the full 15k-message CLINC pool takes 2–6 s per call.

**Sources.**
- Hacohen, Dekel, Weinshall, "Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets"
  (TypiClust), ICML 2022, https://arxiv.org/abs/2202.02794. Code:
  https://github.com/avihu111/TypiClust. It also hosts ProbCover.
- Yehuda, Dekel, Hacohen, Weinshall, "Active Learning Through a Covering Lens" (ProbCover), NeurIPS 2022,
  https://arxiv.org/abs/2205.11320.
- Rauch et al., "No Free Lunch in Active Learning: LLM Embedding Quality Dictates Query Strategy
  Success", arXiv May 2025, https://arxiv.org/abs/2506.01992.
  - Setting: frozen MTEB embeddings on 10 text-classification tasks.
  - A diversity-based initial pool helps.
  - The best strategy depends on embedding quality.
- Zhu et al., "One Knob to Rule Them All: A Unified Optimal Transport View of Cold-Start Active
  Learning", arXiv Aug 2026, https://arxiv.org/abs/2608.03249. Entropic OT unifies typicality, coverage
  and diversity. It was evaluated on vision only.

**Idea to borrow.** Before any label exists, cluster the cached trunk states of the unlabelled pool into
B clusters (B = the label budget) and ask the user to label the most typical message of each cluster.
Switch to uncertainty sampling once there are ~5 labels per class. TypiClust's phase-transition result
says typical examples win at low budget and atypical ones at high budget.

**CPU check (done).**
- **Setup.** Banking77; standardised mean-pooled depth-8 states; logistic probe; 3 random seeds and 2
  clustering seeds; 73 s total.

| budget | random | TypiClust | k-means medoid | classes covered (random / TypiClust) |
|---|---:|---:|---:|---|
| 77 (1 per class) | 20.1 | 28.0 | 26.0 | 49 / 52 |
| 154 (2 per class) | 29.4 | 41.7 | 42.1 | 63 / 71 |
| 385 (5 per class) | 50.2 | 60.6 | 61.4 | 75 / 77 |

That is +8 to +12 points from choosing *which* messages to label. The absolute few-shot numbers on mean
pools are low (FastFit reports 85% at 5 per class with a full fine-tune), which motivates entries 4 and 5.

**Experiment.** `explore/eff_coldstart.py`.
- **Datasets and budgets.**
  - Banking77 and CLINC intent at 1/2/5/10 per class.
  - typed-decisions per workflow pool (~270 messages) at 10/20/40 labels.
- **Selectors.** Random, TypiClust, ProbCover, k-means medoid, and clustering on multi-depth states
  (depth 4 + 14 concatenated).
- **Learners.** Probe, LDA (`geometry_sufficient.py`), and entry 4's in-context learner.
- **Bundle twist.** On CLINC, one selection pass is labelled for all three decisions (intent, domain,
  oos). Does selecting for one decision help the co-requested ones? On typed-decisions, one selection
  serves the workflow's 5 questions.
- **Baseline.** Random selection.
- **Expected signal.**
  - +8–12 points at ≤5 per class on intents.
  - Smaller on typed-decisions. Watch macro-F1: typicality can over-sample the majority class, and
    ProbCover's coverage should help there.
- **Cost.** CPU minutes once features are cached (~5 min on an A10 including encoding).

**Novelty.** Low: the methods are known, and they are simply missing from the notes. New parts:
- one selection for a *bundle* of decisions;
- choosing the clustering depth.

This also sharpens learning E (active distillation): select typical messages first, then ask laya only
about the uncertain ones.

**Solidity.** TypiClust and ProbCover are well replicated in vision. The 2025 text evidence warns that
results depend on the embedding; our check shows it works on ModernBERT mid-depth states for Banking77.

---

## 4. In-context tabular foundation models as zero-gradient branches ("add a decision = one forward pass")

**Status.** Implemented as `explore/lit_eff_icl.py` with **TabICLv2 only**; smoke passes; queued in three
jobs.
- **Licences and installability.**
  - `tabicl` 2.2.0 installed cleanly with `uv add`. Code and checkpoint (`jingang/TabICL`) are
    BSD-3-Clause.
  - MotherNet was skipped: it is not on PyPI (GitHub `microsoft/ticl` only).
  - TabPFN installs, but the TabPFN-2.5/3 weights are non-commercial, so it was not added.
- **Setup.**
  - PCA-64 of mean-pooled states at the split, with 8 estimators.
  - Compared with logreg (on PCA and on the full 768 dimensions), LDA, 5-NN and the tarski probe.
  - No validation data; random labelled subsets (2–3 seeds) plus one TypiClust subset.
  - More than 10 classes (CLINC domain, and the intents in the many-class job) use TabICL's
    hierarchical mode.
- **Mac CPU latency** (`results/tarski/explore_lit_icl_latency_mac.json`):
  - Adding a decision (fit): 1.2 s with 240 context rows, 4.2 s with 1,024.
  - Prediction: 20–24 ms per message at batch 1, 2.5–2.7 ms per message at batch 100.
  - At batch 1 a TabICL branch therefore costs more than the trunk itself (0.8 ms to depth 8 on the ANE,
    ~7 ms on CPU).
  - For serving, use 1–2 estimators, or distil the TabICL predictions into a probe after the decision
    is added.

**Sources.**
- Qu, Holzmüller, Varoquaux, Le Morvan, "TabICLv2: A better, faster, scalable, and open tabular
  foundation model", ICML 2026, https://arxiv.org/abs/2602.11139. Code:
  https://github.com/soda-inria/tabicl. According to the README:
  - permissive licence;
  - more than 10 classes via `support_many_classes` (hierarchical);
  - a KV cache of the training context, so the model is fitted once and predicts many times;
  - CPU/MPS/CUDA support;
  - quality degrades well beyond ~100 features.
  - The predecessor is TabICL, ICML 2025, https://arxiv.org/abs/2502.05564.
- Müller, Curino, Ramakrishnan, "MotherNet: Fast Training and Inference via Hyper-Network Transformers",
  ICLR 2025, https://arxiv.org/abs/2312.08598. Code: https://github.com/microsoft/ticl (Apache-2.0).
  - A transformer reads a training set and emits the weights of a child MLP (2x512, rank-32 factors) in
    one forward pass.
  - It is designed for ≤100 features and ≤10 classes.
  - It matches TabPFN on CC-18 (0.889 vs 0.893 ROC-AUC) with ~50x faster inference.
- Zeng et al., "TabFlex: Scaling Tabular Learning to Millions with Linear Attention", ICML 2025
  (spotlight), https://arxiv.org/abs/2506.05584. It handles hundreds of classes and thousands of
  features, which covers Banking77's and CLINC's intent tasks.
- TabPFN-2.5 (Prior Labs, Nov 2025), https://arxiv.org/abs/2511.08667. The weights are
  **non-commercial**, so it is fine for a comparison and not for shipping. The TabPFN-3 tech report
  (May 2026) is at https://arxiv.org/abs/2605.13986.
- Tajjar, Pfefferle, Purucker, Hutter, "Towards Pretraining Text Encoders for TabPFN", arXiv Jun 2026,
  https://arxiv.org/abs/2606.04876. A learned text-to-TabPFN adapter *lost* to PCA-30 on classification
  (84.5 vs 86.4 ROC-AUC). Use PCA.

**Idea to borrow.** Treat "add a decision" as an in-context learning call instead of a training run:
- **Context.** The labelled messages' PCA-reduced pooled trunk states at the decision's split depth.
- **Prediction.** One forward pass, with no hyperparameters, no gradient loop and no early-stopping
  pitfalls (the typed-decisions bug was exactly that).
- **Calibration.** TabPFN-family models are known for calibrated probabilities.
- **MotherNet variant.** It emits a small MLP, so the served branch costs about as much as a probe.

This is the "classifier from a few examples" counterpart of the "classifier from a description"
hypernetwork that the report lists as open problem 2.

**Experiment.** `explore/eff_icl_branches.py`. Needs `pip install tabicl` and the MotherNet package; not in
the venv today.
- **Features.** PCA-64, fitted on the unlabelled pool, of mean-pooled and CLS states at the autosplit
  depth, plus a two-depth concatenation.
- **Budgets.** 8/16/32/64/all labels per task, selected randomly and by entry 3.
- **Tasks.**
  - typed-decisions: 20 tasks, ≤10 options each, ~270 messages per workflow. The whole pool fits in
    context.
  - CLINC domain (11 labels) and oos (2).
  - Stretch: Banking77 and CLINC intent with TabICL's many-class mode or TabFlex.
- **Baselines.** Probe (`train_branch`), logistic regression on the same PCA features, LDA
  (`geometry_sufficient.py`) and kNN.
- **Metrics.**
  - acc, macro-F1, ECE, NLL.
  - Fit and predict latency per message on the Mac CPU, with the cached context.
- **Expected signal.**
  - At ≤64 labels, +2–5 points over logistic/probe on typed-decisions and CLINC domain/oos, with
    clearly lower ECE.
  - Parity or slightly worse at full data.
  - Probably worse on 77/151-way intents.
- **Cost.** Seconds per task. About 15 min on the Mac CPU for the whole grid, or ~5 min on an A10.

**Novelty.** Partial.
- Tabular foundation models on text embeddings are being benchmarked: Mráz et al., ICML 2025 workshop,
  https://arxiv.org/abs/2507.07829, and the 2026 adapter paper.
- Not found: their use as a branch type on an *intermediate layer of a frozen encoder* for multi-decision
  routing, with label-budget curves against trained probes.

**Solidity.**
- TabICL, TabPFN and MotherNet are peer reviewed and strong on tabular benchmarks.
- Transfer to dense, correlated PCA'd text features is unproven, and the only direct evidence (the
  adapter paper) is mixed.
- The biggest risk is that trunk states are far from the synthetic priors these models were trained on.

---

## 5. Late-interaction branches seeded from label names (FastFit's MaxSim on frozen trunk tokens)

**Status.** Not implemented here: the decision-models agent is implementing FastFit-style late interaction.

**Sources.**
- Yehudai and Bandel, "FastFit: Fast and Effective Few-Shot Text Classification with a Multitude of
  Classes", NAACL 2024 (demo). https://aclanthology.org/2024.naacl-demo.18/, arXiv
  https://arxiv.org/abs/2404.12365, code https://github.com/IBM/fastfit.
  - **Similarity.** The sum over tokens of the maximum cosine to the other text's tokens.
  - **Training.** Batch-contrastive, with class names added to the batch as extra examples.
  - **Inference.** Pick the most similar class name.
  - **Scope.** The whole encoder is fine-tuned.
  - **Results.** Banking77 5/10-shot: FastFit-L 85.2/88.8 vs SetFit-L 81.7/86.4. CLINC150 5/10-shot:
    93.4/95.3.
- mlx-raclate (https://github.com/pappitti/mlx-raclate) runs ModernBERT classifiers and late-interaction
  models natively in MLX. It is useful if this branch type goes to the Mac.

**Idea to borrow.** Score a message against each class by MaxSim between the message's trunk tokens at
depth k and the class name's (or option description's) trunk tokens at the same depth.
- **Encoding.** Class names are encoded once.
- **Trainable part.** A small projection (768→128), trained with FastFit's batch-contrastive loss.
- **Zero-shot at initialisation.** With an identity or PCA projection, it is a classifier from label
  names before any label arrives.

It avoids the mean-pool washout blamed for the typed-decisions probe failure: one JSON field can dominate
a MaxSim.

**Experiment.** `explore/eff_maxsim.py`.
- **Data.**
  - Banking77 and CLINC intent at 5/10 per class and at full data.
  - typed-decisions, with option descriptions as the "names".
- **Baselines.**
  - Probe@k and logistic on pooled states at the same budget. Our check gives ~50–61% at 5 per class.
  - KV-query streams (`systems_kvquery.py`).
  - Description anchors (`geometry_descanchor.py`) for zero-shot.
  - FastFit's published numbers as the full-fine-tune ceiling.
- **Expected signal.**
  - +5–15 points over pooled logistic at 5 per class, still below FastFit's 85.
  - Zero-shot from names well above uniform on Banking77, where names are informative.
  - Near chance on typed-decisions unless descriptions are used.
- **Serving cost.** L × K × m dot products per decision. For Banking77 that is about 16 tokens × 77
  classes × ~4 name tokens × 128 dims ≈ 0.6 MFLOP.
- **Cost.** ~10 min on an A10.

**Novelty.** Partial.
- Known: MaxSim classification (FastFit, ColBERT-style).
- Not found in a quick search: a frozen trunk's intermediate tokens scored against label tokens from the
  same pass, as a per-decision branch.
- Overlaps: attention probes and KV-query streams (a learned readout over tokens), and description
  anchors (label-side encodings, but pooled).

**Solidity.** FastFit is a demo paper, but its numbers are consistent with the SetFit literature and the
code is public.

---

## 6. Trunk on the Neural Engine, branches on the GPU or CPU

**Status: done on the Mac** (`explore/lit_eff_ane.py`, result in `results/tarski/explore_lit_ane_mac.json`,
about 6 minutes).

**Conversion.**
- HF's ModernBERT attention module does not convert with coremltools 9.0 on torch 2.14: an `int` cast
  on a traced shape fails.
- The script therefore re-implements embeddings plus layers [0, k) with static shapes (the math of
  `tarski/fused.py`, with masks and RoPE tables baked in per length bucket). It matches `Trunk.taps` to a
  relative error of 7e-7 (depth 8) and 1e-6 (depth 14).
- Each pair of buckets (32 and 64 tokens) converts in 8–13 s with fp16 weights and 19–28 s with int8
  weights.
- Energy was not measured: `powermetrics` needs sudo.
- Another agent's CPU smoke tests ran concurrently (load average 3–5), so the PyTorch CPU/MPS timings are
  noisy and slightly inflated.

**Latency, batch 1, trunk to depth k** (ms):

| depth, bucket (tokens) | PyTorch CPU | PyTorch MPS | Core ML CPU | Core ML GPU | **Core ML ANE fp16** | **ANE int8 weights** | Core ML ALL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8, L32 (18) | 7.42 | 8.27 | 1.53 | 3.65 | **0.81** | **0.57** | 3.80 |
| 8, L64 (39) | 9.79 | 8.20 | 3.06 | 3.90 | **0.97** | **0.75** | 0.96 |
| 14, L32 (18) | 14.42 | 9.91 | 2.69 | 5.90 | **2.14** | **1.22** | 6.03 |
| 14, L64 (39) | 16.57 | 10.43 | 4.93 | 6.30 | **1.66** | 2.08 | 1.57 |

The ANE trunk is 5–14x faster than the PyTorch MPS trunk at batch 1. Request `CPU_AND_NE` explicitly:
`ALL` sometimes places the model on the GPU and is then 4x slower.

**Parity on the Banking77 test set** (3,080 messages, ANE; branches trained on PyTorch MPS states):

| depth, weights | state rel. error | probe acc (Core ML vs PyTorch) | probe agreement | blocks:2 acc | blocks:2 agreement |
|---|---:|---|---:|---|---:|
| 8, fp16 | 0.3% | 89.04 vs 89.01 | 99.84% | 91.94 vs 91.97 | 99.87% |
| 8, int8 | 1.8% | 89.11 vs 89.01 | 98.57% | 91.84 vs 91.97 | 99.41% |
| 14, fp16 | 0.4% | 87.71 vs 87.74 | 99.84% | n/a | n/a |
| 14, int8 | **non-finite in 2,868 msgs** | 6.96 vs 87.74 | 7.8% | n/a | n/a |

- **fp16 is safe.** The fp16 trunk on the ANE is decision-equivalent at both depths, even with the
  13k-magnitude channels at depth 14. No rotation was needed.
- **int8 weights past the massive-channel layers are not.** The int8-weight model breaks at depth 14.
  Either keep layers from about 10 onwards in fp16, or apply entry 2's split/rotation before quantising.

**Does it change "fused branches do not help on the Mac GPU"?**
- **Setup.** Per message, depth 8, 20-label branches (ms). "SharedSVF" is entry 7's shared-weight
  serving.

| N | MPS trunk: loop / FusedBlocks / SharedSVF | ANE trunk: loop / FusedBlocks / SharedSVF | pipelined ANE+Fused vs sequential MPS+Fused (msgs/s) |
|---|---|---|---|
| 1 | 9.06 / 9.83 / 13.21 | 3.98 / 4.98 / 10.38 | 247 vs 107 |
| 5 | 12.59 / 11.48 / 12.51 | 10.01 / 7.83 / 9.46 | 119 vs 78 |
| 20 | 21.96 / 22.89 / 11.44 | 23.81 / 19.63 / **8.82** | 52 vs 44 |

- **Branch-only time on MPS** (`lit_eff_svf.py --latency-only`) at N = 20: loop 11.3 ms, FusedBlocks
  13.1 ms, SharedSVF 4.9 ms.
- **Answer: mostly no.** With the GPU freed from the trunk, FusedBlocks goes from about equal to the loop
  to 18% faster at N = 20; the difference is noisy.
- **What changes the result is sharing weights.** N SVF branches that share U and V are 2.3x faster than
  the loop at N = 20 (they are slower at N = 1).
- **What the ANE buys.** Trunk latency drops 5–14x. Pipelining the ANE trunk with GPU branches raises
  throughput 1.2x at N = 20 and 2.3x at N = 1.

**Sources.**
- Apple ML Research, "Deploying Transformers on the Apple Neural Engine" (2022),
  https://machinelearning.apple.com/research/neural-engine-transformers. DistilBERT ran up to 10x faster
  with 14x less memory. This is foundational.
- smpanaro, ModernBERT-AppleNeuralEngine (2025), https://github.com/smpanaro/ModernBERT-AppleNeuralEngine.
  - ModernBERT-base runs on the ANE through Core ML at ~3.0 TFLOP/s and 2.1 W for 1024-token inputs.
  - Rotations handle the outliers (entry 2).
  - Base model only, up to 1,024 tokens.
- Shahir M A, "What actually runs: a measurement study of language model placement and decode speed on
  the Apple Neural Engine", arXiv Aug 2026, https://arxiv.org/abs/2608.22110.
  - Small fp16 models can be placed on the CPU despite being ANE-compatible.
  - An int8 export triggered the ANE and cut warm latency 1.9x on an M1.
- Bryngelson, "Apple Neural Engine: Architecture, Programming, and Performance", arXiv Jun 2026,
  https://arxiv.org/abs/2606.22283. Rooflines for the M1 through the M5.
- Chen et al., "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference" (HeteroInfer),
  arXiv Jan 2025, https://arxiv.org/abs/2501.14794. The NPU is the primary unit and the GPU a secondary
  one on unified memory, for 1.34–6.02x speedups on Snapdragon.

**Idea to borrow.** Put the static-shape part every decision shares, the trunk to the deepest split, on
the ANE. Keep the many small, dynamic branches on the GPU or CPU. Pipeline them: message i+1's trunk runs
on the ANE while message i's branches run on the GPU. The unified memory makes the hand-off cheap.

This may also change the "fused branches did not help on the Mac GPU" result, because the GPU would no
longer be busy with the trunk.

**Experiment.** Mac only.
1. Start from the ModernBERT-ANE repo. Export trunks truncated at taps {8, 11, 14}, with shape buckets
   {32, 64, 128, 256} and int8 weights, with and without rotation.
2. Measure batch-1 latency and energy (`powermetrics`) against the MPS trunk (`latency_mps_clean.json`).
3. Measure decision agreement of existing branches, trained on MPS states, when fed ANE states. Also train
   on ANE states.

- **Expected signal.**
  - Equal or lower latency at ≥64 tokens and 2–5x lower energy.
  - ≥99% decision agreement with rotation, visibly lower without.
  - Bucket padding may cost at very short lengths.
- **Cost.** ~1–2 h of conversion work and ~20 min of measurement. No A10 needed.

**Novelty.** Engineering. No paper found on multi-decision serving split across ANE and GPU.

**Solidity.** The ANE sources are single-author arXiv papers and a hobby repo. They are directionally
consistent with Apple's own guidance and with the outlier measurement in entry 2.

---

## 7. Singular-value-only branches (SVF): probe-sized files, shared-weight batching across decisions

**Status.** Implemented as `explore/lit_eff_svf.py`; smoke passes; queued as two jobs.
- **Arms.** probe, full, svf (z only), svf_norm (z plus LayerNorms) and lora8.
- **Sizes.** Measured trainable parameters on Banking77: svf ~66k (0.13 MB) vs full 10.1M (20.2 MB).
- **Checks pass.** An SVF linear with z = 1 matches the base linear (relative error 3e-6), and
  SharedSVF matches the looped branches (absolute error 1e-7).
- **MPS latency** (`results/tarski/explore_lit_svf_latency_mps.json`, branches only, N = 20): loop
  11.3 ms, FusedBlocks 13.1 ms, SharedSVF 4.9 ms. See entry 6 for the trunk-included numbers.
- **Open question.** Accuracy awaits the queued runs.

**Source.** Sun, Cetin, Tang (Sakana AI), "Transformer²: Self-adaptive LLMs", ICLR 2025,
https://arxiv.org/abs/2501.06252.
- **SVF.** Train only a vector z that rescales the singular values of each frozen weight,
  W' = U diag(σ ⊙ z) Vᵀ.
- **Reported result.** It beats LoRA with orders of magnitude fewer parameters on LLM tasks (trained with
  RL).
- **Related.** SVFit, 2024, https://arxiv.org/abs/2409.05926.

**Idea to borrow.** A blocks@k+d branch whose layers are the base layers [k, k+d) with SVF scalings. Only
z, the norms and the head are trained.

- **Size.** The four matrices per ModernBERT layer give ~768 singular values each, so ~3.1k per layer and
  ~6k for d = 2 (plus the head). That is ~130 KB instead of ~20 MB per branch file (blocks:2 has 10.1M
  parameters).
- **Serving.** All SVF branches at the same split share U and V.
  - The first layer's Vᵀx is identical for every branch, so it is computed once.
  - Every later product is the *same weight matrix* applied to N activation streams, with a per-branch
    diagonal in between. That is a plain GEMM over N×L rows, which suits the Mac GPU far better than the
    per-branch-weight `bmm` that did not help.
  - 100 resident decisions cost one shared layer copy plus 100 small vectors, instead of 2 GB.

**Experiment.** `explore/eff_svf.py`.
- **Configs.**
  - Banking77 blocks@8+2 and @14+2, CLINC 11+2, typed-decisions 14+2.
  - Arms: SVF (z only), SVF + LayerNorm + head, LoRA r=8 (from `systems_deltabranch.py`), full copy,
    probe.
- **Metrics.** acc, trainable params, file bytes, and latency at N = 1/5/20 with a shared-weight GEMM
  against `FusedBlocks`.
- **Expected signal.**
  - Between probe and full copy, e.g. ~90.5–91.5 vs 92.0 on Banking77.
  - ~150x smaller files.
  - Lower N = 20 latency than `FusedBlocks` on MPS.
- **Cost.** ~15 min on an A10, plus 5 min of Mac latency.

**Novelty.** Partial.
- Known: SVF itself.
- Not found: SVF on forked branch layers, exploited for shared-weight N-way batching.
- `systems_deltabranch.py` covers post-hoc SVD of *deltas* and LoRA with a shared first product. It does
  not cover diagonal-in-SVD-basis branches.

**Solidity.** Transformer²'s evidence is generative and RL-trained. Classification with a trained head is
untested. SVF can only rescale existing directions, so capacity may be too low for hard tasks.

---

## 8. Unary-score token reduction instead of pairwise ToMe (an arm for systems 7, plus an in-trunk variant)

**Status.** Implemented as `explore/lit_eff_unary.py`, a separate script: ToMe and CompactCache are
copied, and `systems_compaction.py` is untouched. Smoke passes; queued (`lit_unary`).
- **Setup.** typed-decisions, the customer_service and security_incidents workflows, split 11,
  blocks:2.
- **Configs.** full, tome64/32, stride32, cls64/32 (attention from the next layer's CLS query), norm32
  (massive channels excluded), idf32, and two in-trunk cuts: trunk7:cls50 and trunk4:idf50.
- **Checks.** The compact path with nothing removed matches the plain branch. The in-trunk path with
  nothing removed matches the trunk's depth-11 states (relative error 4e-4 at full length).

**Sources.**
- Yang Shanglin, "Why Training-Free Token Reduction Collapses: The Inherent Instability of Pairwise
  Scoring Signals" (CATIS), arXiv Apr 2026, https://arxiv.org/abs/2604.16745.
  - Tested on ViT-L / ImageNet only.
  - ToMe, ToFu, PiToMe and MCTF collapse "cliff-like" at high compression.
  - Pairwise ranking consistency falls from 0.88 to 0.27 in deep layers.
  - The unary-signal method keeps 96.9% of accuracy at 63% fewer FLOPs, where the baselines drop to
    43–65%.
- Foundational for text: Kim et al., "Learned Token Pruning for Transformers" (LTP), KDD 2022,
  https://arxiv.org/abs/2107.00910. It thresholds attention-received scores.

**Idea to borrow.** Rank tokens by a per-token (unary) score rather than pairwise similarity.
- **Candidate scores.** Attention received from CLS at the split, token norm, or a token-id prior (JSON
  keys and punctuation).
- **Where to apply.**
  - At the fork (systems 7).
  - Also *inside* the trunk just after a global-attention layer, so trunk cost drops for every decision,
    not only branch cost.

**Experiment.** A new script next to `systems_compaction.py`, not an edit of it.
- **Configs.** typed-decisions blocks@11+2 at R ∈ {128, 64, 32}. Compare `unary<R>` with the existing
  `tome<R>`, `values` and `stride<R>`.
- **In-trunk variant.** Drop 50% of tokens after an early global layer and retrain probes.
- **Expected signal.** Unary ≥ ToMe at R = 32 by several points. The in-trunk drop stays within ~1 point
  for pooled probes but costs more for blocks.
- **Cost.** ~15 min on an A10.

**Novelty.** Low: it is one arm of an existing idea.

**Solidity.** A single-author, vision-only preprint. The diagnosis is plausible; treat it as a hypothesis
for text.

---

## 9. Backprop-free closed-form test-time refinement of Gaussian (LDA) branches

**Status.** Implemented as `explore/lit_eff_tta.py`; smoke passes; queued (`lit_tta`, about 10 min).
- **Setup.** An LDA branch from 5 or 10 labels per class (random or TypiClust), with no validation data.
- **Methods compared.**
  - none;
  - one-round self-training on the most confident 50%;
  - ADAPT-style online banks: the 16 most confident per class, refitted every 50 messages, at
    τ ∈ {0.9, 0.99};
  - a two-pass transductive variant.
- **Tasks.** Banking77 intent, and CLINC intent (with oos F1) and domain.

**Source.** Zhang et al., "Backpropagation-Free Test-Time Adaptation via Probabilistic Gaussian Alignment"
(ADAPT), NeurIPS 2025, https://arxiv.org/abs/2508.15568. Code: https://github.com/AIM-SKKU/ADAPT.
- Class means and a shared covariance are updated in closed form from confident test samples, using
  bounded per-class knowledge banks.
- No source data and no gradients are needed.
- It was evaluated on CLIP.

**Idea to borrow.** After a few-label LDA branch is deployed (`geometry_sufficient.py`'s
sufficient-statistics design), fold confident unlabelled traffic into bounded per-class banks.
- **Cost.** CPU only and O(d²) per update.
- **Reversible.** Removing a bank is an exact subtraction, consistent with that design's unlearning
  property.

**Experiment.**
- **Setup.** Banking77 and CLINC at 5 and 10 labels per class, selected by entry 3. Stream the test set
  once (online), and also run it transductively.
- **Compare.** No adaptation, and learning F's self-training.
- **Expected signal.** +2–5 points at 5 per class, ~0 at full data. Risk of confirmation bias on oos:
  report oos-F1.
- **Cost.** CPU minutes.

**Novelty.** Low. Test-time adaptation and semi-supervised prototypes are known, and learning F covers
self-training. The closed-form, reversible version is the only part that fits tarski specifically.

**Solidity.** Peer reviewed, but vision-language only.

---

## Checked and not proposed

- **Ettin encoders** (Weller et al., "Seq vs Seq: An Open Suite of Paired Encoders and Decoders", 2025,
  https://arxiv.org/abs/2507.11412; 17M–1B encoders on the ModernBERT recipe that beat ModernBERT at
  matched size).
  - As a small "scout trunk": truncating ModernBERT-base at depth 4 already costs about as much as a
    17–32M model and keeps one cache and one tokenizer.
  - Ettin-150M is a candidate trunk upgrade, but that runs into the branch-transport problem
    (geometry 11).
- **Multi-task edge serving systems.**
  - S2M3, ICDCS 2025, https://arxiv.org/abs/2508.04271.
  - Visual Perception Engine, https://arxiv.org/abs/2508.11584.
  - EdgeServing, https://arxiv.org/abs/2605.05527.
  - All are shared backbones plus heads with placement or scheduling. Nothing beyond Mainstream and Wei
    et al. matters for a single-user laptop.
- **New early-exit rules.** ADEPT, https://arxiv.org/abs/2601.03700; PAC-Bayes theory of early exits,
  https://arxiv.org/abs/2604.15764. These are covered by the queued anytime and depth-gating scripts;
  only the risk-control piece is used, in entry 1.
- **Hypernetwork successors.** Code2LoRA (https://arxiv.org/abs/2606.06492), Doc-to-LoRA and T2L all
  target generative LLMs and are already in `frontier_methods.md`. MotherNet and TabICL (entry 4) are the
  classification-native version.
- **MLX backends.** "Benchmarking On-Device ML on Apple Silicon with MLX" (https://arxiv.org/abs/2510.18921)
  is a non-archival poster on BERT-family models. An MLX port of the branches might fix the MPS
  per-op-dispatch cost that likely sank fused branches, but that is engineering. Revisit alongside
  entry 6.
