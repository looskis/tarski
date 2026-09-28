# Theory registry

Every hypothesis and exploration idea, the run that tested it, the key numbers and a verdict. Final synthesis
after all queued runs finished (`python experiments/theory_status.py`: 76/76 have results). Details:
`THESIS.md`, `research_notes/explore/*.md`; raw results in `results/tarski/{lambda (A10), lambda_a100 (A100), . (Mac)}`.

**Verdicts:** **supported**, **refuted**, **mixed**, **inconclusive**, **known** (replicates prior work),
**not tested** (reason).

**How to read the numbers**
- Test accuracy in %, unless stated. "full" = same-architecture full fine-tune (all 22 layers trainable).
  "blocks@k+d" = d base layers copied at split k; "probe@k" = LayerNorm + mean pool + linear.
- Test sizes: Banking77 3,076; CLINC150 5,500 (18.2% oos; always-in-scope = 81.8%); typed-decisions **100 per
  task** (per-task SE ~4-5 points; 5-task mean SE ~2; 20-task mean SE ~1). Differences under ~2 points on
  typed are noise unless seeds agree.
- CLINC protocol differs by script: the sweeps score 151-way intent (oos as a class, with the train/test
  prior shift); k-shot, DROID, KLDA, conformal and lit-OOS scripts hold oos out of training and report
  in-scope accuracy (4,500 messages) plus an oos AUROC. Numbers across the two protocols do not compare.
- Duplicate runs: the newest file by modification time is used. Superseded: `lambda/explore_kvquery_typed`,
  `lambda_a100/explore_kvquery_typed_long` and the Mac `explore_kvquery_typed` (under-trained) ->
  `lambda_a100/explore_kvquery_typed_fixed`; `explore_passenger_typed` -> `explore_passenger_typed_v2`;
  `lambda/explore_active_distillation` (2 tasks) -> `lambda_a100/explore_active_distillation`; the 15:25
  proper-score run (`run_proper_score.out`) was overwritten by the 17:37 run (`run_proper_score_objectives.out`);
  Mac `sweep_*.json` (single seed, typed ones pre-fix) -> `lambda/sweep_*_s{0,1,2}.json`. Smoke files are not used,
  except where noted for free-rider (a post-hoc computation over the real sweeps, re-derived here).

## Hypotheses

| # | Claim | Runs | Key numbers | Verdict |
|---|---|---|---|---|
| H1 | A 1-2 layer branch is within a few points of a full fine-tune | `sweep_{banking77,clinc150}_s0-2`, `sweep_typed_all_s0-2`, `sweep_typed_cs_full_s0-2`, `sweep_typed_other_full_s0`, `explore_kshot_*` | Banking77 blocks@18+2 92.9±0.1 vs full 93.8±0.6 (-0.9); probe 88.5. CLINC intent best branch 87.8 (blocks@16+2) vs full 90.8 (-3.0); domain 88.6, oos 87.9 (no full ref). Typed cs blocks@14+2 78.5 vs full 79.6±0.4; other 15 tasks 76.4 vs 76.5 (full 1 seed); all 20: 77.0±0.3 vs 77.3 (probe@22 65.0). **Few labels widen the gap** (k-shot, oos held out): blocks-full -2.9/-3.7 at 25/10 per class (Banking77), -5.9/-11.8 (CLINC) | **Supported** with full training data; weakens sharply below ~25 labels per class |
| H2 | An extra decision costs a small fraction of a pass | `latency_{mps,cpu}_clean` (Mac) | 512 tokens, 20 decisions, blocks@11+2: 50 ms vs 507 separate models vs 937 laya (M6 GPU); 289 vs 2,518 vs 6,410 ms (CPU). Marginal CPU cost: 11 ms (2-layer), 6 ms (1-layer), 0.2 ms (probe), 124 ms (separate model), ~320 ms (laya) | **Supported** (9-10x vs separate models, 19-22x vs laya; probes 32-38x) |
| H3 | Switching decisions is a file load, not a reload | `latency_*_clean` (warm), `switch_cold_{a10,cpu}` | Warm: 20 MB branch 6.5 ms (CPU) / 18 ms (MPS) vs 596 MB copy 14 / 84 ms. Cold, load+first decision: A10 201 vs 314 ms (GPU shared with other jobs), CPU 124 vs 180 ms. Resident, 5 tasks: 806 MB vs 3,457 MB | **Mixed**: switch time is cheap either way; the real saving is memory (4.3x at 5 tasks) |
| H4 | A one-pass curve picks a good per-task split | `autosplit_{banking77,clinc150}` (probe curve), `autosplit_branch_{banking77,clinc150,typed_cs}` (1-layer, 120-step branch proxy) | Probe curve is flat, picks depth 1-3: Banking77 blocks@1-3 90.6-90.9 vs 92.9; CLINC 86.5/85.4/85.8 vs 87.8/88.6/87.9. Branch proxy: Banking77 picks 18 -> 92.8 (best 92.9); CLINC intent/domain 18 -> 88.1/88.1 (at best); oos picks 2 -> 85.6 vs 87.9; **typed cs (31 val rows) picks 2-18 -> mean 73.8 vs 78.5 for a fixed blocks@14+2**. Cost 104 / 381 / 572 s (A100) | **Mixed**: probe selector refuted; branch proxy works on short text with large val sets, fails on CLINC oos and on typed |
| H5 | Copying the next base layers beats random or final-layer copies | `ablation_init` (Banking77/CLINC, Mac), `lambda_a100/ablation_init_typed` (cs, 3 seeds) | Banking77 @4/8/14, next vs random vs top: 91.5/91.9/92.7 vs 92.0/92.1/92.4 vs 89.9/90.7/91.9. CLINC intent: 85.4/86.5/88.5 vs 87.5/87.7/88.4 vs 84.2/85.2/86.8. Typed cs @8/14/20: 75.1/77.6/74.4 vs 75.7/77.4/77.1 vs 69.1/74.1/74.5 (at @20 next = top = layers 20-21) | **Refuted** on all three datasets: random >= next; final-layer copies worst |
| H6 | Adding/retraining a branch cannot change another | `tests/test_isolation.py` | Bit-identical outputs of other branches | **Supported** (by construction) |

## Exploration: systems (`research_notes/explore/systems.md`)

| # | Idea | Runs | Key numbers vs baseline | Verdict |
|---|---|---|---|---|
| 1 | KV-query decision streams | `lambda_a100/explore_kvquery_typed_fixed` (newest, 1 seed); `lambda/explore_kvquery_banking77`; blocks:2 baseline from `_typed_long` | Typed (split 11, 20 tasks): kvq:2:4 73.6, kvq:4:4 73.7, kvq:4:4:frozen 71.2 (10k params) vs blocks:2 75.3, probe 62.7, at 0.09-0.17 vs 5.1 GFLOP/decision. A100 fused latency N=20: 25.0 vs 30.0 ms (probe 22.0); N=1 22.4 vs 19.9 (slower). Banking77: 91.3-91.5 vs 92.2 at 3x fewer FLOPs | **Mixed**: -1 to -2 points for 30-60x fewer FLOPs per decision; wall-clock gain only at N>=20 on GPU; CPU not measured |
| 2 | Incremental re-encoding for threads | `lambda/explore_threadinc_clinc150` | Blocks:2: stale 15% of trunk FLOPs, 95.8% same decisions, 84.5 vs 85.9; band32 53%, 99.4%, 85.7; stale-trained branch served stale 86.5 | **Supported** |
| 3 | Per-decision-set trunk pruning | `lambda/explore_dce` | Agreement with unpruned: 83-90% at 10% sparsity, 41-64% at 20%; set-specific plan no better than another set's | **Refuted** |
| 4 | Schema-constant templates for JSON | `lambda/explore_schema` | 43% of tokens templated, ~57% trunk FLOPs (analytic). Blocks:2 (10 tasks) 74.5 -> 74.6 served templated (75.1 if trained templated); probes 62.6 -> 59.0 | **Supported** for blocks (1 seed; FLOPs saving not yet measured as latency) |
| 5 | Decision planner (early exit + dependencies) | `lambda/explore_planner` | CLINC exit+FD tau 0.5: 89.7 vs all-blocks 87.4 at 4.9 vs 17 layer-equivalents (-71%). Typed: no dependency reaches 0.95 purity | **Supported** (CLINC only) |
| 6 | Near-duplicate speculative cache | `lambda/explore_speccache` | CLINC 25% hits, 21% compute saved at -0.2; Banking77 8% hits, 6% saved at -0.1 | **Mixed** (small on these datasets; real Slack duplication untested) |
| 7 | Token compaction before branches | `lambda_a100/explore_compaction` | ToMe to 128 tokens 72.0 vs 74.6 (-2.6, half the branch FLOPs); to 64: 65.5 (-9.1). Value tokens only: 75.0 (+0.4) at 62% branch FLOPs | **Mixed**: generic compaction refuted; schema-aware (values only) is free |
| 8 | Joint split planning | `lambda/explore_jointsplit` (probe curves) | Typed: E[trunk depth] 16.8 -> 13.9 at -0.25; CLINC 22 -> 4 at -0.3 | **Supported** (modest; probes only) |
| 9 | Delta-coded branches / shared products | `lambda/explore_deltabranch` | LoRA-16 90.7 vs 91.8; BitDelta 88.2-90.1 at 14x smaller; SVD-128 91.6 at 4x smaller; shared products 9% faster only at N=20, 273 tokens | **Refuted** (predicted <=0.5 loss, 20-30% faster) |
| 10 | Compressed trunk-state views | none (literature); see efficiency #2 | Int4 split+rotate cached states cost <=0.4 points (`explore_lit_quant_*`) | **Known** (Du et al. 2020, ColBERTv2); replicated |
| 11 | Layer-granular continuous batching | `lambda/explore_layerbatch` (simulated arrivals) | Short 4-depth mix: p95 60 vs 203 ms (load 0.3), 84 vs 277 ms (0.6) vs static+early-exit; equal at load 0.9 and on the 2-depth long mix | **Mixed** (tail-latency win at moderate load only; low value on a single-user laptop) |

## Exploration: representation geometry (`geometry.md`)

| # | Idea | Runs | Key numbers vs baseline | Verdict |
|---|---|---|---|---|
| 1 | Passenger tokens | `explore_passenger_clinc`, `explore_passenger_typed_v2` (newest, >=300 steps) | CLINC P4r8 86.7 vs blocks@11+2 87.1, probe 85.2. Typed P4r8 63.4 vs probe@22 65.7, logreg@11 68.0, blocks ~75-77 | **Mixed**: parity on short text, refuted on JSON |
| 2 | One-way questions on laya | `explore_oneway_laya` (Mac), `lambda_a100/explore_oneway_train` | No retraining: 68.4 vs 76.7 unsplit (block split 45.3). Retrained with the one-way mask (2 epochs, 1 seed): 78.4 vs 79.3 for the same fine-tune with the joint mask; 4.5x fewer token-layers per extra question | **Supported** after retraining |
| 3 | One-way cloze zero-shot routes | `explore_passenger_clinc` (cloze arm) | CLINC domain prior-corrected: one-way 57-59 vs two-way 62-64; Banking77 best template 41.8 vs 41.7; absolute accuracy weak | **Refuted** |
| 4 | Token-id-centred pooling | `lambda/explore_depthscan` | vs plain mean pool: -0.8 to -4.7 (Banking77), -0.7 to -1.6 (CLINC), -2.4 to +0.6 (typed) | **Refuted** |
| 5 | Global-layer-aligned splits / global-only forks | `explore_depthscan`, `explore_globalsplit_banking77`, `explore_globalfork_{banking77,typed}` | No probe sawtooth (+-0.2); blocks@15..20+1 91.5/91.9/92.6/92.7/91.7/91.5 (global 18 best, global 15 near worst); global-only fork -0.3/-0.5 (Banking77); typed next/global/local 75.2/74.6/75.4 | **Refuted** |
| 6 | Out-of-scope from trunk geometry, no oos data | `explore_depthscan` | Cross-depth disagreement AUROC 0.914 (86.3% binary vs 81.8% majority) vs supervised ridge 0.840; plain Mahalanobis only 0.861 (fixed by Mahalanobis++, OOS #3) | **Supported** |
| 7 | Decision atlas across workflows | `explore_depthscan`, `explore_atlas` | Zero-shot Spearman -0.06 to 0.21 (within-workflow 0.68); atlas prior below target-only at every n=8-64 (urgency n=8: 0.12 vs 0.18) | **Refuted** |
| 8 | Cost-aware tap mixing | `lambda/explore_tapmix` (trained, 300 steps) | Banking77 prefix mixer @12 91.0 vs best single 89.9 (+1.1, 100% of the gain at cut 12); CLINC @22 86.0 vs 84.1 (+1.9, 56% at cut 12); typed mixers 59-68 vs single 69.8 (overfit). The earlier +5.3/+3.5/+0.5 were ridge vs weak ridge baselines | **Mixed**: helps short text, hurts typed |
| 9 | Shared decision subspaces / fused branches | `explore_subspaces` | Affinity real (0.32 vs null 0.02) but grouped bottlenecks = random/singleton (66-69); fused blocks -1.9 (typed 73.7 vs 75.6), -0.5 (CLINC) | **Refuted** |
| 10 | Sufficient-statistics (LDA) branches | `explore_sufficient` | LDA vs logreg: CLINC +1.7/+2.0, Banking77 -0.1/-0.8, typed -5.6/-6.1. Merge/unlearn exact (<=1e-11). New label, CLINC k=5/10 recall 0.53-0.64/0.69-0.82 vs retrained 0.22-0.24/0.51-0.57. OOD AUROC 0.84 | **Mixed**: strong on many-class short text, weak on typed |
| 11 | Branch transport across base upgrades | none | - | **Not tested** (ruled out by the user: retrain instead) |
| 12 | Description-anchored routes | `explore_descanchor` | Leave-one-workflow-out 31.5-34.1 vs uniform 31.8, majority prior 48.8; in-distribution 64-67 vs per-task logreg 67-69 | **Refuted** |

## Exploration: learning (`learning.md`)

| # | Idea | Runs | Key numbers vs baseline | Verdict |
|---|---|---|---|---|
| A | Direct proper-score loss | `lambda/explore_proper_score_objectives` (17:37 rerun, >=300 steps; 8 tasks) | Probe 59.9 vs CE 60.3 (majority 50.4) | **Refuted** (no gain) |
| B | REINFORCE on proper scores | same | 53.8, majority-class share 0.91 (collapses) | **Refuted** |
| C | Bias-only branches (K+1 params) | same | 36.7-46.5 under every objective vs majority 50.4 | **Refuted** |
| D | Class-balanced / logit-adjusted loss | same | cb_ce 55.7 (one diverged arm); logit-adjusted 43.4, macro-F1 0.346 vs 0.342 | **Refuted** |
| E | Active distillation from local laya | `lambda_a100/explore_active_distillation` (newest; 5 cs tasks, probe@6, 1 seed) | 224 teacher labels: active soft 61.6 vs random soft 59.8 vs random hard 55.0; teacher 76.4; student ceiling ~probe@6 64.6 | **Inconclusive** (+1.8 is within noise; soft > hard by 4.8) |
| F | Self-training on unlabelled traffic | `explore_self_training` (2 typed tasks, 60 labels) | 50 -> 52 and 54 -> 50 (oracle 58, 55); pseudo-label precision 0.23-0.65 | **Refuted** on typed (but see efficiency #9 on short text) |
| G | Branch merging (TIES) | `explore_merge_{clinc150,typed_cs}` | TIES+refit 85.4-85.8 vs independent 87.6 (CLINC); 72.2 vs 77.6 (cs) | **Refuted** |
| H | Weight-imprinted route addition | `explore_route_imprinting` | After imprinting: old domains 0.0%, new 100% (imprint dominates); surgical fine-tune old 60.3 (-2.3), new 15.8 vs ceiling 86.0 | **Refuted** as implemented (see anomalies; LDA/KLDA/APER answer the question) |
| I | Probe-difficulty curriculum | `explore_curriculum` (2 tasks) | 56 vs 58, 50 vs 51 | **Refuted** |
| J | Outlier exposure with Banking77 negatives | `explore_outlier_exposure` | Any lambda>0 collapses to always-in-scope 81.1 vs 82.3 at lambda=0 | **Refuted** (protocol doubt, see anomalies) |
| K | Label-efficiency curve, branch vs full | `lambda_a100/explore_label_efficiency` | One task only (cs.action): blocks@14+2 hard 55/60/50/58, soft 51/52/53/59 at n=20/50/100/160; full fine-tune n=50: 55. Leverage summary invalid (n bug) | **Inconclusive** (bug + 1 task; short-text answer in decision-models #5) |

## Exploration: far-field (`farfield.md`)

| # | Idea | Runs | Key numbers vs baseline | Verdict |
|---|---|---|---|---|
| 1 | Syndrome decoding of decision bundles | `lambda/explore_syndrome` | Label-free AUROC: joint 0.929, agree 0.919, combo 0.946 vs supervised stack 0.957; joint decoding +0.1-0.4 in-scope | **Mixed** (competitive, not above supervised) |
| 2 | JSON field gates / coarse-graining | `lambda/explore_coarsegrain` | 20 typed tasks: field gate/attn 71.4 @16 vs mean-pool probe 66.1; blocks 75-77; readable gates (severity <- asset_criticality) | **Supported** vs probes; below blocks |
| 3 | Negative selection | `explore_negsel` | AUROC 0.50-0.53 vs 1-NN 0.72-0.73 | **Refuted** |
| 4 | Prediction error across depth | `explore_predcode` | Error AUROC 0.53-0.74 vs MSP 0.91-0.92; error+MSP 0.87-0.91 (< MSP) | **Refuted** |
| 5 | Anytime decision bundles | `explore_anytime` | Banking77 patience-1 exit, mean depth 2.1: 90.4 vs 88.8 fixed@22; CLINC bundle depth 3.9: 85.4 vs 84.1 (probes) | **Supported** |
| 6 | Free-rider depth | `explore_freerider_smoke` (ran on the 3-seed lambda sweeps; full run empty), re-derived from sweeps | CLINC all-at-6 87.0 -> ride to 16 88.1 (+1.1); typed cs 8 -> 14 +2.5 (3-seed init ablation; +4.4 was 1 seed); 14 -> 20 -3.2 (urgency -6.3) | **Supported** (serving policy: ride up to ~16, not to 20) |
| 7 | Inverse-variance fusion over depth | `explore_anytime` | Fused >= single at 22/22 depths (Banking77, CLINC intent/domain), 16/22 (oos); Banking77 fused@8 90.8 vs 89.1 | **Known** (ZTW 2021); replicated |
| 8 | Alternative splicing | `explore_splice` | vs next layers: Banking77 probe-picked +0.9, random +0.4, final -0.7; CLINC domain final -1.9, mid -1.1, random -0.4 | **Mixed**: final-layer copies worse as predicted; next not special (supports H5) |
| 9 | Shared attention, private MLP | `explore_sharedattn` | -0.3 to -1.2 points; 1.30x (bs 1) / 1.36x (bs 16) faster than fused branches at N=20 (A10) | **Mixed** |
| 10 | Sleep-like consolidation | `explore_consolidate_{clinc150,typed_cs}` | CLINC 87.1 vs 87.4 independent (joint 86.8); typed cs 74.0 vs 77.6 (joint 72.0); forgetting <=0.2 / <=3 | **Mixed**: works on CLINC, -3.6 on typed |
| 11 | Template cancellation | `explore_template` (split 14, 20 tasks) | Probes 57.6 -> 65.0 (field) / 63.2 (global); blocks 76.3 -> 74.1 / 75.4 | **Supported** as predicted (probes only) |
| 12 | Population search over topology | `explore_evolve` (3 cs tasks) | Evolved 79.0 vs fixed 14+2 81.3 vs autosplit 75.0, at 11-13 trainings per task | **Refuted** |

## Exploration: skeptic (`skeptic.md`)

| Item | Runs | Key numbers | Verdict |
|---|---|---|---|
| Bug 1: CLINC oos prior shift (1.6/3.2/18.2%) | `lambda/explore_oos_priorshift` | Saerens-EM: oos F1 0.29 -> 0.57, macro-F1 0.60 -> 0.74 (oracle 0.57 / 0.74) | **Supported** (confirmed; label-free fix) |
| Bug 2: temperature fitted to soft labels, ECE on hard | `lambda/explore_dual_calibration` | Mean ECE 0.138 -> 0.118 fitting to hard labels (0.139 -> 0.124 without the NaN task); only ~9/20 tasks have >=30 val rows to fit T | **Supported** (small) |
| Bug 3: no same-architecture full fine-tune | typed/Banking77/CLINC `full` sweeps | 3 seeds everywhere except typed other-15 (1 seed); see H1 | **Supported**, resolved |
| Bug 4: early stopping on tiny validation sets | `sweep_typed_buggy_earlystop`, `sweep_typed`, `lambda/sweep_typed_s0` | Typed probes 50-55 -> 58-62 (fix) -> 62-66 (>=300 steps) | **Supported**, fixed |
| Bug 5: probe autosplit picks badly | `autosplit_*` | See H4 | **Supported** |
| Bug 6: local windows are a no-op on short inputs | structural (Banking77/CLINC <= 64 tokens < 128 window) | - | **Supported** |
| Bug 7: copy-next init not better than random | `ablation_init*` | See H5 | **Supported** (= H5 refuted) |
| Idea 1: prior-shift-corrected oos gate | as Bug 1 | as Bug 1 | **Known** (Saerens 2002) |
| Idea 2: receptive-field-aware split depth | `lambda/explore_receptive_field` | Synthetic field lookups solved at depth 1-4 (layer 0 is global) for gaps <=400; only gap 400 + 24 decoys fails at every depth | **Refuted** |
| Idea 3: separate correctness vs soft-label calibration | as Bug 2 | as Bug 2 | **Supported** (small) |
| Idea 4: confirmatory re-rank for autosplit | - | - | **Not tested** (folded into the H4 branch proxy) |
| Idea 5: attention-pooling probe | - | Nearest: coarsegrain token-attention head 66.7 vs mean pool 64.2 @22 | **Not tested** (skipped as duplicate) |
| Idea 6: per-instance depth gating | `lambda/explore_depth_gating` | probe@4 -> blocks@18+1 escalation: thresholds 0.72-0.96 give 92.6-93.0 at mean depth 6.7-10.1 (test-picked); the val-picked threshold (0.999) gives 92.7 at depth 15.6 = always-18 | **Mixed**: mechanism works; threshold selection fails |

## Literature: decision models (`lit_decision_models.md`)

| # | Idea | Runs | Key numbers vs baseline | Verdict |
|---|---|---|---|---|
| 1 | Conformal decision bundles (joint guarantee) | - | Per-decision pieces tested in OOS #2 and #6 | **Not tested** as a joint bundle (left to the OOS agent) |
| 2 | Out-of-fold confidence head | `explore_confhead_clinc150`, `lambda_a100/explore_confhead_{typed,typed_cs_oof}` | CLINC: MLP head AUROC 0.923 vs MSP 0.853, AURC 0.021 vs 0.032. Typed: 0.705 vs MSP 0.759 (val->test); OOF cs 0.806 vs 0.799; leave-workflow-out worse than MSP on 3/4 | **Mixed**: helps with 9k head-training rows, not with ~600 |
| 3 | Drop-in trunk with zero-shot readout | `explore_trunkswap_modernbert`, `lambda_a100/explore_trunkswap_{gte,gliclass}` (1 seed each) | **gte-modernbert-base**: probe@22 93.4 vs 88.1 (Banking77), 90.5 vs 85.1 (CLINC 151-way; full FT is 90.8); typed cs probe@14 77.0 vs 65.2, blocks 80.8 vs 78.4; label-name zero-shot 64-71% (oos AUROC 0.91) vs 12-24%. GLiClass: probes +0 to +10, blocks -1.4 to +1.7, native zero-shot 38-48% | **Supported** (gte); GLiClass mixed |
| 4 | Label-free "none of the above" (DROID) | `lambda/explore_droid` | CLINC AUROC probe@22 0.908 -> 0.952, blocks 0.939 -> 0.950 (mixup alone 0.954); oos F1 0.61 -> 0.78; supervised 0.96-0.97. Banking77-open (unseen intents): +0.00-0.01 | **Mixed**: far-OOS yes, near-OOS no |
| 5 | Late-interaction anchors + k-shot curve | `lambda_a100/explore_kshot_{banking77,clinc150}` (oos held out; 3 seeds for k<=10, 2 for 25, 1 for all) | Banking77 k=1/5/10/25/all: full 26/69/83/89/94, blocks 16/68/80/87/93, probe 23/54/68/80/88, LI 32/57/67/78/87. CLINC: full 40/84/92/95/97, blocks 23/68/80/89/94, probe 32/68/80/88/93, LI 44/70/78/86/91 | **Mixed**: LI wins only at 1 shot; full FT dominates at 5-25 per class |
| 6 | Day-0 routes from a local LLM (Qwen3-8B) | `lambda_a100/explore_synthroutes` | CLINC in-scope, synthetic only: 54.0 (probe) / 50.0 (blocks) vs real 92.5 / 94.5; refine loop no gain; synthetic + 5 real/class 76.0 / 78.7 vs 5 real only 69.5 / 66.9 | **Mixed**: useless alone, +6.5 to +11.8 as augmentation |
| 7 | Encoder vs decoder trunk (Ettin) | `explore_ettin_{enc,dec_bibranch}`, `lambda_a100/explore_ettin_dec_causal` | blocks@k+2, enc vs causal dec vs dec+bidirectional branch: Banking77 90.6 / 71.4 / 75.7; CLINC 87.7 / 71.0 / 74.1; typed cs 73.0 / 65.2 / 63.6. Ettin enc < ModernBERT (93.0 / 87.1 / 78.4) | **Supported**: the trunk must be an encoder |

## Literature: efficiency (`lit_efficiency.md`)

| # | Idea | Runs | Key numbers vs baseline | Verdict |
|---|---|---|---|---|
| 1 | Shadow trunk tier-0 with risk-controlled exits | `explore_lit_shadow_{banking77,typed}`, `lambda_a100/..._clinc150`, Mac latency | Banking77: 44-68% exit at tier 0, <1.7% flips, accuracy -0.2 to +0.9 vs the served branch; CLINC 23-48% exit, <1% flips; typed 0% exit. Plain TF-IDF tier 0 (88.9) >= best shadow tier (88.4). Tier 0 0.01-0.03 ms vs trunk 4-38 ms (CPU) | **Mixed**: cascade works on short text; the shadow part adds nothing over TF-IDF |
| 2 | Split massive channels, rotate, quantise | `lambda_a100/explore_lit_quant_{banking77,clinc150}`, `explore_lit_quant_typed`, Mac CPU | Cached states int4 split+rot within 0.4 (plain int4 collapses to 11-38% on intent/domain); trunk int8 weights -0.4 to +0.2; w4 -1.6 to -20; KV4 -0.4 to -3.8. Mac CPU dynamic int8: 1.2-1.3x faster at 16 tokens, 0.6-0.8x (slower) at >=64 | **Supported** for memory; not a CPU speed-up |
| 3 | Cold-start selection of what to label | `explore_lit_coldstart_{banking77,clinc150,typed}` | Banking77 5/class: LDA typicality 64.8 vs random 53.4, blocks 62.0 vs 53.1; CLINC +2-4 at 1-2/class, ~0 at 5; typed: typicality below random on 3/4 workflows | **Mixed**: many-class short text only |
| 4 | Tabular in-context models (TabICLv2) as branches | `lambda_a100/explore_lit_icl_{typed,manyclass}`, `explore_lit_icl_clinc150` | Typed full pool 68.4 vs in-script probe 62.6, logreg-PCA 67.3; at 16-64 labels 50-59 vs probe 54-61; many-class below LDA (Banking77 67 vs 69, CLINC 60 vs 76); 20-40 ms per message on CPU | **Refuted** (no few-label edge, 100x a probe's cost, far below blocks) |
| 5 | Late-interaction branches from label names | same run as decision models #5 | See decision models #5 | **Mixed** (wins only at 1 shot) |
| 6 | Trunk on the Apple Neural Engine | `explore_lit_ane_mac` | Depth 8: 0.81 ms fp16 / 0.57 ms int8 vs 8.3 ms MPS; fp16 parity (99.8-99.9% same decisions); int8 non-finite at depth 14; ANE trunk + shared SVF: 8.8 ms at N=20 vs 22 ms all-MPS | **Supported** (Mac) |
| 7 | Singular-value-only (SVF) branches | `explore_lit_svf_banking77`, `lambda_a100/..._clinc_typed`, Mac MPS | vs full branch: Banking77 @14 91.4 vs 93.0, CLINC 85.5 vs 87.1, typed cs 77.8 vs 79.0; @8 below a probe (88.4 vs 89.1). 0.03-0.14 MB vs 20 MB; N=20 shared: 1.8 vs 4.8 ms (A10), 4.9 vs 13.1 ms (M6) | **Mixed**: -1.2 to -1.7 for 150x smaller, 2.7x faster |
| 8 | Unary-score token reduction | `lambda_a100/explore_lit_unary` (10 typed tasks, 1 seed) | At the fork, 32-64 tokens: 59.5-65.4 vs 74.8; in-trunk IDF keep-50% at layer 4: 75.1 at 67% trunk FLOPs | **Mixed**: fork reduction refuted; in-trunk IDF drop is free |
| 9 | Closed-form test-time refinement (LDA) | `explore_lit_tta` | Banking77 5/class random 53.4 -> 59.8, typicality 63.7 -> 65.6; CLINC intent 5/class 62.5 -> 68.5 (plain self-training 69.4) | **Supported** (small; similar to self-training) |

## Literature: incremental routes and out-of-scope (`lit_incremental_oos.md`)

| # | Idea | Runs | Key numbers vs baseline | Verdict |
|---|---|---|---|---|
| 1 | KLDA / RanPAC closed-form routes | `lambda_a100/explore_lit_klda` | KLDA ensemble 90.7 vs probe 89.9 (Banking77), 92.7 vs 93.2 (CLINC in-scope); incremental final 89.9 / 92.4 vs joint probe 89.9 / 93.2 vs new-rows-only 52; new-route recall k=5: LDA 0.31/0.44 > KLDA 0.11/0.28 | **Supported** (probe parity in closed form) |
| 2 | Conformal selection for auto-routing | `explore_lit_conformal_select` | Banking77 FDR 0.044-0.049 at q=0.05, 86-88% auto-routed; CLINC (oos shift) FDR 0.09-0.12 at q=0.05; EM weighting 0.069-0.147; only oracle weights hold | **Mixed**: guarantee breaks under oos shift |
| 3 | Mahalanobis++ | `explore_lit_mahapp` | Raw 0.857 -> centred-L2 0.914 -> relative-Mahalanobis 0.957 (= supervised stack); oos F1 (cal+EM) 0.78 vs MSP 0.63 | **Supported** |
| 4 | Open-set label-shift estimation | `lambda_a100/explore_lit_openset_shift` | OOS share at 5-50% estimated within 0.001-0.022 using a Banking77 reference (count-and-classify errs up to 0.20); noise reference fails; F1 at the estimate = oracle | **Supported** (needs a real reference corpus) |
| 5 | Harmful-drift alarm + label-efficient check | `explore_lit_drift` | MSP-CUSUM: 0 false alarms; detects new routes (delay 258 msgs) and typos (10-18); proxy-mean misses new routes; MMD never fires. Active labels: CI width 0.070 vs 0.108 at 100 labels; certifies drops with 25-50 | **Supported** |
| 6 | Conformal clarification sets (CICC) | `explore_lit_cicc` | Marginal coverage 0.946-0.957 at alpha 0.05, worst-10 routes 0.80-0.88; classwise at ~13-20 cal rows/route -> 68-label sets; laya on the clarify band 65.4 vs top-1 56.4 (CLINC) but 58.9 vs 63.5 (Banking77); open-set gate catches 56% oos, 24% unseen routes | **Mixed** |
| 7 | BCTS + online prior tracking | `explore_lit_bcts` | BCTS ~ TS (+-0.4); windowed MAP-EM +2.2 (Banking77 stream 88.8 -> 91.0), +1.3 (CLINC); plain windowed EM -6 to -15; real CLINC shift oos F1 0.63 -> 0.74 | **Mixed**: BCTS no gain; MAP-EM tracking helps |
| 8 | LLM-written near-OOS negatives | `lambda_a100/explore_lit_synth_neg` | AUROC 0.82-0.85 vs 0.90-0.93 with no negatives, 0.94 with 250 real | **Refuted** |
| 9 | APER on forked branches | `lambda_a100/explore_lit_aper_branch` | 11 sessions to 150 routes: LDA on [trunk@22, branch] 93.0 vs joint retrain 94.3; sequential new rows 33.9 | **Supported** |
| 10 | OOS-inbox route discovery | `explore_lit_discovery` | 0-3 of 15 held-out routes recovered, 1-3 junk clusters; 13% of unseen-route messages flagged | **Refuted** |

Not experiments: the novelty flags in `lit_decision_models.md` and the "checked and not proposed" lists.

**Verdict counts (91 rows):** supported 31, mixed 26, refuted 25, known 3, inconclusive 2, not tested 4.

## Anomalies and follow-ups

| # | Issue | Evidence | Effect on conclusions | GPU rerun? |
|---|---|---|---|---|
| 1 | **Label-efficiency bug**: `explore/learning_label_efficiency.py` builds `{"n": n, ..., **eval_branch(...)}` and `evaluate()` also returns `n` (=100 test rows), so every curve point reads n=100 and the "leverage 1.00x" summary is meaningless. Only the default task ran; n=270 > pool 269, so no full-data full-FT point | `lambda_a100/explore_label_efficiency.json` vs its log (n=20/50/100/160) | K is inconclusive; typed has no few-label curve | **Yes, would change K** (command below the table) |
| 2 | **NaN temperature**: `tarski/train.py:fit_temperature` (LBFGS, no line search) can return NaN; `clamp` does not catch NaN, so probabilities become NaN and argmax is garbage | `sweep_typed_all_s2` probe@22 invoice.matches_order 46% vs 77% in s0/s1 (ECE 0.000, NLL NaN); `explore_dual_calibration` one task; confhead/trunkswap probe logs `T=nan` | Typed probe@22 mean 65.0 -> ~65.5; dual-calibration gain 0.020 -> 0.015 | No (no verdict changes); add a NaN guard (fall back to T=1) |
| 3 | **Free-rider full run empty**: script looks for `results/tarski/sweep_<name>.json`, the GPU box only has per-seed files | `lambda/explore_freerider.json` has no datasets | Numbers come from the smoke run over the pulled 3-seed sweeps; re-derived here from the sweeps | No (CPU post-hoc) |
| 4 | **typed-cs sweep seeds 1-2 OOM'd** (A10 shared with other jobs) | `run_typed_cs_s{0,1,2}.out` tracebacks; `sweep_typed_cs_s1/s2` lack blocks@8+2/14+2 | cs blocks@14+2 3-seed number comes from `sweep_typed_all_s*`; "+4.4" free-rider was 1 seed (3-seed: +2.5) | No |
| 5 | **Branch-proxy autosplit fails on typed**; the THESIS claim "typed decisions pick depths 1-20" rests on `autosplit_typed.json`, a Mac run with pre-fix, under-trained probes (e.g. outcome probe 32%) | `lambda_a100/autosplit_branch_typed_cs` (73.8 vs 78.5); `autosplit_typed.json` | H4 is mixed, not supported; per-task depth variety is still visible in fixed probe curves (`explore_jointsplit`: best splits 2-22) | No (31 validation rows is the limit, not training) |
| 6 | **Depth-gating headline wrong**: log says "trunk only 17% of the way"; JSON says depth saved 17% (mean depth 15.6 of 18). THEORIES' old "93.0 at depth 8.3" picked the threshold on test | `lambda/explore_depth_gating.json` (`picked_by_val` 0.999) | Skeptic idea 6 is mixed | No (fix the selection rule offline) |
| 7 | **Layer-batching simulator**: static batching p50 415 ms at load 0.6 vs 75 ms at 0.9 (long mix 1,362 vs 90 ms): latency falls as load rises | `lambda/explore_layerbatch.log` | Comparisons against "static" are invalid; against static+exit they stand | No |
| 8 | **Joint-training contradiction**: merge script's jointly trained body 54.0 vs consolidation's 72.0 on the same 5 cs tasks | `explore_merge_typed_cs` (15:08) vs `explore_consolidate_typed_cs` | Merge's joint arm looks under-trained; G stays refuted on TIES alone (72.2 vs 77.6) | No |
| 9 | **Route imprinting**: imprinting sends 100% of old-domain messages to the new route | `explore_route_imprinting.json` | Imprint norm/bias not matched to existing rows; H is refuted only "as implemented" | No (LDA/KLDA/APER cover route addition) |
| 10 | **Diverged arms in the objectives run**: probe cb_ce on needs_human (33%, Brier NaN), bias-only CE on cs.urgency (0.0%, Brier NaN) | `run_proper_score_objectives.out` | Means for cb_ce and bias-CE dragged down; A-D verdicts unchanged | No |
| 11 | **Outlier-exposure protocol**: uniform OE target on a binary in/out head pushes Banking77 outliers to 50/50 rather than to "oos"; all lambda>0 runs stop at epoch 4/8 at the trivial 81.1 | `run_outlier_exposure.out` | J tests a muddled objective; DROID (K+1 class) is the better test | No |
| 12 | **Ridge vs trained tap mixing disagree** (+5.3 ridge vs +1.1 trained on Banking77; typed turns negative) | `explore_depthscan` vs `explore_tapmix` | Old THEORIES quoted the ridge numbers; corrected | No |
| 13 | **Passenger parity on GPU** is 0.8% (message states) and 2.1% (passengers), vs <2e-6 on CPU | `explore_passenger_*` logs | "Bit-identical shared pass" holds only in fp32; verdict unchanged | No |
| 14 | **k-shot CLINC run resumed mid-way**; k=25 has 2 seeds and k=all 1 (same for Banking77); the untrained late-interaction arm collapses to 0.7% at ~42k bag tokens; summary mislabels proto columns | `explore_kshot_clinc150.log` | Trained arms unaffected | No |
| 15 | **KV-query typed**: 1 seed; blocks:2 baseline taken from the under-trained `_long` run (blocks there used the fixed `train_branch`) | `explore_kvquery_typed_fixed` has no blocks:2 arm | Gap of -1.7 has SE ~1 | No (would not change "mixed"; CPU latency is the open question) |
| 16 | **DCE mismatch**: "unpruned acc" (intent 92.2, oos 99.8) is not the probe test accuracy (86.1, 84.9), and the script prunes for probes@11, not the blocks@11+2 the note describes | `lambda/explore_dce.log` | Agreement collapses at 10-20% sparsity either way | No |
| 17 | **Drift MMD never fires**, even when accuracy falls to 26% | `explore_lit_drift.log` | Likely a mis-set threshold; MSP-CUSUM is the recommended alarm anyway | No |
| 18 | **H5 at split 20**: "next" and "top" copy the same layers (20-21) | `ablation_init_typed` | The @20 row is next/top vs random only | No |
| 19 | **Ettin decoder**: probe accuracy falls with depth (77.9 -> 57.5) and last-token probes are 43-50%, a sign the attention-sink token dominates pooling | `explore_ettin_dec_causal.log` | Blocks with mean and last pooling agree (71.4 / 71.6), so "encoder needed" stands | No |
| 20 | **Single-seed trunk swap** drives the top recommendation; typed evidence is 5 tasks x 100 messages | `explore_trunkswap_gte` | Short-text probe gains (+5) are far outside noise; typed blocks +2.4 is ~1 SE | Not needed for the verdict; before switching the default trunk, run 3 seeds on all 20 typed tasks (script has no seed/typed-all option yet) |
| 21 | **Self-training disagrees across datasets**: refuted on typed (2 tasks, probes) but closed-form TTA and plain self-training gain +2 to +7 on Banking77/CLINC at 5-10 labels/class | learning F vs efficiency #9 | Dataset/branch dependent, not a contradiction in code | No |
| 22 | Minor: 270 of 1,885 synthetic near-OOS negatives share a 5-gram with test oos; `bank_250` arms give F1 0.000 | `explore_lit_synth_neg.log` | Verdict (refuted) unaffected | No |

**The one verdict-changing rerun (idea K).** In `explore/learning_label_efficiency.py`, put `"n": n` after the
`**eval_branch(...)` unpack (or drop `n` from the evaluate dict), change `ns`/`full_ft_ns` 270 -> 269, then on the A100
(~75 min):

```
.venv/bin/python explore/learning_label_efficiency.py --tasks customer_service.action,customer_service.category,customer_service.churn_risk,customer_service.needs_human,customer_service.urgency --out results/tarski/explore_label_efficiency.json
```

Optional, not verdict-changing: `.venv/bin/python explore/learning_active_distillation.py --seed 1 --out results/tarski/explore_active_distillation_s1.json`
(and `--seed 2`) would tell whether E's +1.8 is real.

## What to build into tarski

Ranked by expected value for local-first Slack/ticket routing (accuracy, laptop latency, few labels, adding
routes, out-of-scope). The core (frozen trunk + blocks@14..18+2 branches + shared-trunk serving, H1/H2) is assumed.

1. **Default trunk gte-modernbert-base** (decision models #3): drop-in, same shape; probes gain ~5 points (CLINC probe 90.5 = ModernBERT full FT), typed blocks 80.8, and label-name zero-shot routes at 64-71%.
2. **Label-free out-of-scope stack** (OOS #3, #4, skeptic bug 1): relative Mahalanobis on L2-normalised pooled states (AUROC 0.957, no oos labels), thresholds set from an EM/open-set estimate of the no-route share (within 0.02).
3. **Closed-form route addition** (OOS #1, #9, geometry #10): LDA/KLDA heads, APER-style on [trunk, branch] features: add or remove a route in seconds, exact unlearning, 93.0 vs 94.3 for a full retrain after 11 sessions.
4. **Neural Engine trunk on Macs** (efficiency #6): 0.8 ms vs 8.3 ms to depth 8 at fp16 parity; pair with int8 weights + split channels for memory (efficiency #2).
5. **Harmful-drift alarm** (OOS #5): MSP-CUSUM, no false alarms, flags new routes or garbled input; confirm with 25-50 active labels before retraining.
6. **Incremental thread encoding** (systems #2): band-32 reuse at 53% of trunk FLOPs with 99.4% unchanged decisions; stale-trained branches at 15%.
7. **Anytime / early-exit planner** (far-field #5-6, systems #5, skeptic idea 6): exit when all requested decisions are confident, ride shared depth up to ~16; 71% fewer layers on CLINC. Needs a working threshold-selection rule.
8. **Cheap tier-0 cascade on CPU** (efficiency #1): TF-IDF tier answers 44-68% of Banking77 messages (<1.7% flips) and 23-48% of CLINC (<1%) in 0.02 ms; no exits on JSON states.
9. **Few-label helpers** (efficiency #3, #9, decision models #6): typicality-picked first labels (+6 to +11 at 5/class), closed-form test-time refinement, and LLM-written examples to augment 5 real per route (+6 to +12). The k-shot curve shows a full fine-tune still wins by 3-12 points at 10-25 labels per route.
10. **JSON-specific trunk savings** (systems #4, #7, efficiency #8): schema templating or value-only tokens keep blocks accuracy at 57-67% of trunk/branch FLOPs; worth it only once ticket states are long.
