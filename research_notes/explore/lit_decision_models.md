# Literature scan: decision and classification models (2025–2026)

Checked 2026-09-27. Scope: recent encoders, "System One" typed-decision models, calibration, one-pass
multi-classifier serving, zero/few-shot classification from label descriptions, LLM-as-teacher
distillation, multi-label/hierarchical routing. Lens: what could beat tarski's weak spots (out-of-scope,
zero-shot new routes, very few labels) or add a new capability.

**How this was checked.** Web search, then the abstract, HTML full text or README of each source
(noted per entry as "read: ..."). GitHub-only projects are not peer reviewed; their numbers are the
authors' own. Novelty verdicts are quick checks against `novelty_check.md`, `explore/TRIAGE.md`, the
five lens files and `reports/Appending layers to language models.md`. Check again before claiming
anything.

**Already covered, not re-proposed:** proper-score and REINFORCE losses (learning A/B), outlier exposure
from other tasks (learning J), active distillation (learning E), Mahalanobis, kNN and depth-disagreement
OOS (geometry 6, far-field 3/4), cloze or ModernBERT-Instruct zero-shot (geometry 3), description-anchored
probe weights (geometry 12), attention probes, SetFit, prior-shift correction (skeptic 1), temperature
fitted to hard labels (skeptic 3), syndrome and joint decoding of domain and intent (far-field 1).

**Current numbers the experiments are measured against** (from `results/tarski/`):
- **CLINC150 OOS.** The oos binary branch scores 83.5–84.8% (always answering in-scope: 81.8%).
  - Max-softmax of the intent probe: AUROC 0.904–0.929, FPR95 0.29–0.34.
  - A stacked supervised detector: AUROC 0.957.
  - oos F1 is 0.29 uncorrected and 0.57 after Saerens-EM (`explore_syndrome.json`,
    `explore_oos_priorshift.json`).
- **typed-decisions (customer_service).** blocks@14+2 mean accuracy 0.786, ECE 0.13–0.17 against hard
  labels (`sweep_typed_cs_s*.log`).
- **Banking77.** blocks@18+1 93.1; probes 87–89.
- **No few-shot (k-shot) curve exists** for any branch type.

---

## Ranked summary

| # | Idea to borrow | Source (verified) | Weak spot / capability | A10 min |
|---|---|---|---|---|
| 1 | Conformal decision bundles: one joint coverage guarantee over a message's N decisions, with an answer / clarify / escalate policy and label-shift weighting for oos | CICC (NAACL Findings 2024); PASC (arXiv 2605.18812, 2026); Podkopaev & Ramdas (2021) | OOS, new capability (guarantees) | 0–10 |
| 2 | Out-of-fold confidence head over "prediction geometry", separate from the distribution head; bundle and depth features that only a shared trunk has | Verdict 2.0 (GitHub, 2026) | Calibration and selective decisions on typed-decisions | 10 (80 with full OOF) |
| 3 | Swap in a same-architecture trunk that already has a zero-shot readout (gte-modernbert-base, GLiClass-modern-base) | gte-modernbert (2025); GLiClass (arXiv 2508.07662); GLiNER Guard (arXiv 2605.05277); BTZSC (arXiv 2603.11991) | Zero-shot routes from the same pass | 45–60 |
| 4 | Label-free "none of the above": interpolated inter-class features plus open-domain text as a K+1 class, threshold from in-scope data only | DROID (arXiv 2510.14110, 2025) | OOS for user-defined routes | 10–15 |
| 5 | Late-interaction class anchors (MaxSim against label-name and support-example tokens), plus the missing k-shot curve | FastFit (NAACL 2024 demo) | Very few labels | 20 |
| 6 | Day-0 routes from descriptions: a local LLM writes examples, a refine loop targets confusions | Curate-Train-Refine (arXiv 2601.16530, 2026); Incubator (2024); Vajjala & Shimangaud (2025) | Zero-shot routes | 20–30 |
| 7 | Paired encoder vs decoder trunk (Ettin), plus a causal trunk with bidirectional branch layers | Ettin (ICLR 2026); ModernBERT-or-DeBERTaV3 (IJCNLP-AACL 2025) | Thesis question: does the trunk need to be an encoder? | 60–90 |

**Implementation (2026-09-27).** Ideas 2–7 are implemented as `explore/lit_dm_*.py`, each with a passing
CPU `--smoke`. Full runs are queued in `explore/QUEUE_lit_dm.txt`, k-shot curve first. Idea 1 was left to a
parallel agent.

A final section lists **novelty flags for existing tarski claims and queued ideas**: CLM-8B, a frozen-laya
mixture of expert heads, Verdict as an external typed-decisions baseline, and SingGuard's RI-Mask.

---

## 1. Conformal decision bundles (answer / clarify / escalate with a joint guarantee)

**Status (2026-09-27): not implemented here.** A parallel agent is implementing conformal selection and clarification sets; skipped to avoid duplication.

**Sources**
- den Hengst, Wolter, Altmeyer, Kaygan, "Conformal Intent Classification and Clarification for Fast and
  Accurate Intent Recognition", NAACL Findings 2024. https://arxiv.org/abs/2403.18973 (read: HTML).
  - **Classifier:** fine-tuned bert-base. Marginal split conformal prediction turns scores into sets.
  - **Policy:** a set of size 1 is answered; sizes 2–7 become a clarification question; anything larger
    is rejected as ambiguous.
  - **CLINC150 in-scope:** 99% coverage, 97% singletons.
  - **Banking77:** 98% coverage, 73% singletons, average clarification size 2.84, 4% ambiguity rejection.
  - **OOS extension:** CLINC oos F1/AUROC goes from 0.07/0.88 (vanilla) to 0.91/0.97 (optimised).
- Kotte, "PASC: Pipeline-Aware Conformal Prediction with Joint Coverage Guarantees for Multi-Stage NLP
  and LLM Pipelines", arXiv 2605.18812, May 2026, single author (read: abstract).
  - **Method:** reduces joint coverage over K stages to one scalar conformal problem on the maximum
    nonconformity score across stages.
  - **Result on CoNLL-2003:** 96.4% end-to-end coverage, against 93.4% for Bonferroni and 86.5% for
    independent CP, at the same set size (1.083).
- Podkopaev & Ramdas, "Distribution-free uncertainty quantification for classification under label
  shift", UAI 2021. https://arxiv.org/abs/2103.03323 (read: abstract). Weighted conformal prediction
  with weights estimated from unlabelled target data.

**Idea to borrow.** Each message already gets several decisions from one trunk pass: CLINC has intent,
domain and oos; typed-decisions has five questions per state. Conformalise the whole bundle:
- **Joint sets.** One calibration pass on the maximum per-decision nonconformity (PASC's reduction) gives
  sets that cover all N decisions jointly with probability at least 1−α.
- **Serving policy (CICC):** all singletons means answer; small sets mean ask one clarifying question;
  large sets, or oos inside the set, mean escalate to laya, an LLM or a human.
- **Label shift.** Weight the calibration scores by the Saerens-EM prior already in
  `explore_oos_priorshift`, so the guarantee survives CLINC's shift from 1.6% oos in training to 18.2%
  in test.

This adds a capability tarski lacks: a decision with a stated error rate, plus a principled escalation
signal.

**Novelty.** Conformal intent classification exists (CICC). Joint and simultaneous conformal via a max
score is a standard trick; PASC's claim is modest. Label-shift-weighted CP exists. Applying all three to
N decisions served from one frozen trunk, with shift weighting for oos, was not found. Low-to-moderate
novelty, high practical value.

**Solidity.** CICC is peer reviewed and the method is textbook. PASC is a single-author preprint, but
its core reduction is standard and easy to verify on our own data.

**Experiment**
- **Data:** CLINC150 (3 decisions per message) and typed-decisions (5 per state).
- **Inputs:** cached validation and test logits of the existing branches: probe@22 and blocks@11+2 on
  CLINC, blocks@14+2 on typed customer_service.
- **Arms:**
  - score type: LAC (1−p_y) or APS;
  - calibration: per decision, Bonferroni, or joint-max;
  - conditioning: marginal, or class-conditional (Mondrian) for oos;
  - weighting: unweighted, or label-shift weighted using the EM prior.
- **α:** 0.1 and 0.05.
- **Metrics:**
  - empirical per-decision and joint coverage on test;
  - average set size and share of singletons;
  - % answer / clarify / escalate;
  - oos recall among escalations.
- **Baselines:** CICC's fine-tuned BERT (CLINC 99% coverage with 97% singletons; Banking77 98% with 73%)
  and tarski's oos F1 (0.29 uncorrected, 0.57 with EM).
- **Expected signal:**
  - Unweighted CP under-covers oos on the CLINC test set. This exchangeability failure is itself a
    publishable illustration of the prior shift.
  - Weighted CP restores coverage.
  - Joint-max sets are smaller than Bonferroni at equal joint coverage.
  - On typed-decisions, validation sets of about 31 rows per task make α below ~0.05 impossible
    (resolution 1/(n+1)). Pool calibration scores by question type across tasks and report that as an
    assumption.
- **Cost:** CPU only, under 5 minutes if logits are dumped; about 10 A10 minutes to regenerate them.

---

## 2. Out-of-fold confidence head, separate from the distribution head (Verdict 2.0)

**Status (2026-09-27): implemented, smoke passes, queued.** `explore/lit_dm_confhead.py`. Queue lines `confhead_clinc150` (A10), `confhead_typed` (A100) and `confhead_typed_cs_oof` (A100, 5-fold out-of-fold on the customer_service workflow). Heads: logistic regression and a small MLP, with feature-group ablations; protocols val->test, leave-one-workflow-out, out-of-fold.

**Source.** "openJev-verdict-2.0", Heman10x-NGU, GitHub, 2026, not a paper.
https://github.com/Heman10x-NGU/openJev-verdict-2.0 (read: README). Weights: heman10x/rlcd-modernbert-151m
on Hugging Face.
- **Model:** a 149.6M ModernBERT-base decision model (the awesome list says it is built on GLiClass).
  Question and option markers go in the input; options are phrased as hypotheses ("It is ...").
- **Two heads:**
  - a marker-pointer distribution head: typed-decisions accuracy 77.10% (laya: 76.60%), Brier 0.0636,
    but **ECE 0.1513**;
  - a separate MLP confidence head trained out-of-fold over "prediction geometry": **ECE 0.0144**,
    correctness AUROC 0.7664.
- **Selective accuracy:** 83.44% at 80% coverage, 89.00% at 60%, 91.40% at 50%.
- **Order bias:** symmetric permutation-KL training reduces option-order flips to 4.76%.

**Solidity.** Single-author repository. The confidence-head features, fold count, seeds and training data
are not disclosed. Treat the numbers as plausible but unverified. The method family is established:
meta-model confidence estimation (e.g. ConfidNet, Corbière et al., NeurIPS 2019) and stacking. The
useful fact for us: on the same benchmark and trunk size, the distribution head's ECE (0.15) matches
tarski's typed ECE (0.13–0.17), and a separate head fixed it.

**Idea to borrow.** Stop asking the decision head to be both a good soft-label fit and a good
correctness predictor. Train one small correctness model shared across all decisions, on features that a
shared-trunk system gets for free:
- max probability, margin, entropy, option count, question type;
- probe-vs-branch agreement;
- cross-depth argmax agreement between probe@6, 11, 16 and 22;
- entropy and agreement of the other decisions on the same message.

Temperature scaling (skeptic 3) cannot change the ranking, so it cannot raise AUROC. This can.

**Novelty.** A meta-model confidence head is known. Bundle and depth features for per-decision
correctness were not found. The syndrome experiment used similar features for oos only.

**Experiment**
- **Data:** typed-decisions, all 20 tasks after the bug fix; CLINC150 as a second dataset, counting oos
  as "incorrect" for in-scope intents.
- **Cheap version:** train the head on pooled validation logits of all tasks (about 700 rows),
  leave-one-workflow-out, so we also measure transfer to new decisions.
- **Full version:** 5-fold out-of-fold branch logits on train.
- **Metrics:** correctness AUROC, hard-label ECE, AURC, selective accuracy at 80/60/50% coverage.
- **Baselines:** hard-label temperature-scaled max prob (skeptic 3); Verdict's AUROC 0.766, ECE 0.014 and
  83.4% at 80% coverage; tarski's typed ECE of 0.13–0.17.
- **Expected signal:** AUROC +0.03–0.06 over max prob; ECE below 0.05; selective accuracy at 80% coverage
  4–6 points above full-coverage accuracy. Bundle features should matter most on typed-decisions, where
  the five questions about one state are correlated.
- **Cost:** about 10 A10 minutes to dump logits; the head trains in seconds on CPU. Full out-of-fold is
  20 tasks × 5 folds × ~50 s, about 80 minutes.

Supporting evidence that recalibration matters even for the commercial model: Rafe & Das, "Calibrated
Decisions at Scale", arXiv 2609.24052 (Sept 2026; read: abstract). Recalibrating Jev on in-domain labels
cut its calibration error 3.3×, and Jev reports probabilities on discrete grids (a "resolution floor").

---

## 3. A drop-in trunk that already has a zero-shot readout

**Status (2026-09-27): implemented, smoke passes (gte and GLiClass), queued.** `explore/lit_dm_trunkswap.py`. Queue lines `trunkswap_modernbert` (same-protocol baseline), `trunkswap_gte`, `trunkswap_gliclass` (A10 each). The GLiClass encoder is exported to a plain ModernBertModel (it reproduces the stock forward exactly), and its native readouts are re-implemented from the GLiClass source: joint, chunked at 25 labels, and cached. `lit_dm_common.load_trunk` casts trunks to fp32, because gte-modernbert-base is stored in fp16 and transformers 5 would otherwise load it that way.

**Sources**
- Alibaba-NLP, gte-modernbert-base, 2025. https://huggingface.co/Alibaba-NLP/gte-modernbert-base
  (read: model card). A contrastive embedding model fine-tuned from `answerdotai/ModernBERT-base` (same
  shapes), CLS pooling, MTEB classification average 76.99.
- Stepanov et al., "GLiClass: Generalist Lightweight Model for Sequence Classification Tasks",
  arXiv 2508.07662, 2025 (read: HTML).
  - **Architecture:** a uni-encoder; labels go into the input as `<<LABEL>>` tokens; scoring is a
    dot product or MLP.
  - **Speed:** throughput drops 7.6% from 1 to 128 labels, where a cross-encoder slows ~52×.
  - **Backbones:** the authors found DeBERTa-v3 backbones consistently better than ModernBERT.
  - **Many labels:** weak. `knowledgator/gliclass-modern-base-v3.0` (151M, ModernBERT-base, LoRA r=512)
    scores zero-shot F1 0.356 on Banking77 and 0.558 on average over 14 datasets (model card).
- Minko, Sadiekh, Kokuykin, "GLiNER Guard: Unified Encoder Family for Production LLM Safety and
  Privacy", arXiv 2605.05277, May 2026 (read: HTML). A shared-weight bi-encoder with cached label
  embeddings matches the uni-encoder (74.3 vs 73.8 F1 on safety benchmarks).
- Aarab, "BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models,
  Rerankers and LLMs", arXiv 2603.11991, 2026 (read: HTML; cited in `novelty_check.md` for cost only).
  - Banking77 zero-shot macro-F1: GTE-large embeddings 0.59, best NLI cross-encoder 0.45,
    Qwen3-Reranker-8B 0.70.
  - Embedding models give the best accuracy/latency trade-off; NLI cross-encoders plateau with size.

**Idea to borrow.** ModernBERT-base has no label-free readout except its MLM head (geometry 3). Same-
architecture checkpoints load through `Trunk(base=...)` unchanged:
- gte-modernbert-base gives a zero-shot route as cosine(message, description) at the top layer;
- GLiClass is a decision-tuned trunk.

Branches keep reading mid-depth, so one pass serves trained branches and label-free routes. Unlike
geometry 12, no map M needs training: the contrastive pretraining is the map. GLiNER Guard's result
suggests cached label embeddings lose little against re-reading.

**Novelty.** Zero-shot classification with embeddings is standard. Not found: whether contrastive or
classification post-training of the trunk helps or hurts **mid-depth frozen branches**, and serving both
from one pass. This is a cheap, thesis-relevant measurement.

**Experiment**
- **Trunks:**
  - ModernBERT-base (baseline);
  - gte-modernbert-base (drop-in);
  - the GLiClass-modern-base-v3.0 encoder, with LoRA merged and exported to a plain ModernBertModel by a
    small conversion script; messages go in without label tokens.
- **Per trunk:**
  1. probe depth scan (`explore/` depthscan) on Banking77, CLINC150 and typed;
  2. blocks@14+2 and @18+1 on Banking77, blocks@11+2 on CLINC;
  3. zero-shot:
     - gte: cosine against label descriptions (BTZSC's Banking77 verbalisations; one-line CLINC
       descriptions), with an oos threshold on max cosine set from in-scope validation only;
     - GLiClass: its native label-in-input mode as the "re-read" reference.
- **Baselines:** the ModernBERT-base table (blocks@18+1 93.1; CLINC blocks@11+2 87.0; typed probes);
  BTZSC's GTE-large 0.59 and GLiClass-modern-base 0.356 on Banking77.
- **Expected signal:**
  - gte zero-shot accuracy 0.50–0.60 on Banking77 and 0.60–0.75 on CLINC in-scope.
  - Mid-depth branches within ±1 point of baseline, since contrastive fine-tuning mostly moves the top
    layers.
  - probe@22 up several points, because embedding-trained top layers are linearly separable.
  - The GLiClass trunk on typed-decisions is genuinely uncertain: decision-tuned features, but
    message-only input is off-distribution.
- **Cost:** about 3 minutes of caching per trunk and dataset, plus about 20 minutes of sweeps per trunk:
  45–60 minutes in total.
- **Serving caveat:** a zero-shot route at the top layer forces the full 22-layer pass, which undoes
  per-request truncation. Report the cost.

---

## 4. Label-free "none of the above" for user-defined routes (DROID recipe)

**Status (2026-09-27): implemented, smoke passes, queued.** `explore/lit_dm_droid.py`, queue line `droid` (A10).
- **Arms:** base, mixup, squad (2,000 SQuAD 2.0 questions), droid (mixup + squad), supervised.
- **Configurations:** probe@4, probe@22 and blocks@11+2, on CLINC150 with oos removed from training and on a Banking77 split with 75% of intents known.
- **Supervised arm:** on Banking77 it is an oracle ceiling (it trains on the held-out intents).
- **Cross-dataset negatives:** left to `learning_outlier_exposure.py` (learning J).

**Source.** Rashwan, Zawbaa, Dutta, Assem, "DROID: Dual Representation for Out-of-Scope Intent
Detection", arXiv 2510.14110, Oct 2025 (read: HTML).
- **Model:** two **frozen** encoders (USE, and a TSDAE-adapted RoBERTa) feed a 1.56M-parameter MLP head
  with K+1 outputs.
- **Negatives (no oos labels used):**
  - 500 synthetic feature-space outliers: convex combinations of embeddings from different known classes;
  - 500 SQuAD 2.0 open-domain negatives per epoch.
- **Decision:** a single max-softmax threshold calibrated on in-domain validation only.
- **Results:**
  - CLINC150 unknown F1 95.88 (mean over 25/50/75% known intents; 92.52 at 75%), known F1 93.65;
  - Banking77 unknown F1 94.07.

**Solidity.** The protocol holds intents out as unknown and always adds CLINC's 1,200 oos utterances.
At 25% known classes most of the test set is unknown, which inflates unknown F1. There is no ablation
that separates synthetic from open-domain negatives. arXiv only. The direction is credible; the
magnitudes are not comparable to our setup.

**Idea to borrow.** Today tarski's oos branch needs oos labels, which users never provide for their own
routes. Train every routing branch with an extra "none" class built from two free sources:
- interpolated pooled trunk features between two different classes ("near" outliers);
- generic open-domain text passed once through the trunk and cached ("far" outliers).

Pick the threshold from in-scope validation only.

**Novelty.** Low. This is outlier exposure plus feature mixup. It overlaps learning J (other tasks'
messages as negatives, not implemented). The new pieces relative to our notes are feature-space
interpolation and the "no oos labels anywhere" protocol.

**Experiment**
- **Data:**
  - CLINC150 with **oos removed from training**;
  - a Banking77 open-intent split at 75% known intents, for comparison with DROID.
- **Branches:** probe@4 and probe@22 (best oos depths so far); blocks@11+2, mixing at the pooled output
  before its head.
- **Arms:**
  - mixup negatives, λ ~ U(0.3, 0.7);
  - 2,000 SQuAD 2.0 questions;
  - learning J's cross-dataset negatives (Banking77 messages for CLINC);
  - all combined.
- **Metrics:** AUROC, FPR95, oos F1 and recall at the threshold that keeps 95% of in-scope validation,
  in-scope accuracy.
- **Baselines:**
  - MSP AUROC 0.904–0.929 and FPR95 0.29–0.34;
  - oos binary accuracy 83.5–84.8%;
  - the stacked supervised detector at 0.957 AUROC, which uses oos labels: beating it without labels is
    the bar.
- **Expected signal:**
  - Open-domain negatives cut FPR95 sharply on CLINC, partly because many CLINC oos queries are generic
    questions (a dataset artefact; confirm on the Banking77 open-intent split).
  - Mixup adds at most 1–2 AUROC points.
- **Cost:** 10–15 A10 minutes.

---

## 5. Late-interaction class anchors for few labels (FastFit), and the missing k-shot curve

**Status (2026-09-27): implemented, smoke passes, queued first.** `explore/lit_dm_kshot.py`, queue lines `kshot_banking77` and `kshot_clinc150` (A10).
- **Label counts:** k = 0/1/5/10/25/all per class, with 3 seeds up to k=10, 2 at k=25 and 1 at all.
- **Arms:** prototypes, late interaction (untrained and trained), probe@8, blocks@14+2 (Banking77) or @11+2 (CLINC), and the full fine-tune.
- **Few-shot protocol:** below k=all, the validation set is also k-shot and only fits the temperature. Training runs its full schedule (at least 300 optimiser steps) and keeps the last epoch, since a user with few labels has no large validation set.
- **CLINC:** oos messages are held out of training and scored by AUROC.

**Sources**
- Yehudai & Bandel (IBM), "FastFit: Fast and Effective Few-Shot Text Classification with a Multitude of
  Classes", NAACL 2024 System Demonstrations. https://aclanthology.org/2024.naacl-demo.18/ ·
  https://arxiv.org/abs/2404.12365 (read: HTML).
  - **Method:** class names are encoded as text into the same space as examples; batch contrastive
    training; ColBERT-style token-level similarity (sum of max cosines). The encoder is **fine-tuned**.
  - **5/10-shot results, large backbone:**
    - Banking77: 85.2/88.8, against SetFit 81.7/86.4 and a standard classifier 79.7/87.4;
    - CLINC150: 93.4/95.3, against SetFit 90.4/88.4 and a standard classifier 92.0/94.5.
  - **Caveat:** the token-level vs sentence-level ablation gains only 0.65–1.33 points on average; most
    of the gain comes from contrastive training with label anchors.
- Optional trunk: LightOn GTE-ModernColBERT-v1 (2025), a ModernBERT-based late-interaction model
  trained with PyLate (arXiv 2508.03555).
  https://huggingface.co/lightonai/GTE-ModernColBERT-v1 (read: search result and blog only).

**Idea to borrow.**
- **Class representation:** the frozen trunk's depth-k token states of its name or description, plus any
  support examples.
- **Scoring:** a message scores against each class by MaxSim after a small trained projection
  (768→128), trained with batch contrastive loss.
- **What it buys:** shot 0 already works from names; adding a class or example appends tokens with no
  retraining; token matching may matter more on **frozen** features than on fine-tuned ones.

This also gives the thesis its first accuracy-vs-labels curve, which none of the notes have.

**Novelty.** Moderate. FastFit fine-tunes the encoder and ColBERT is retrieval. A frozen, mid-depth
MaxSim branch with label-token anchors was not found in a quick search.

**Experiment**
- **Data:** Banking77 and CLINC150 (in-scope) at k ∈ {0 (names only), 1, 5, 10, full}, 3 support-set
  seeds.
- **Arms:** probe@k, a prototype (mean-pooled centroid), blocks@18+1, and the late-interaction branch at
  depths 8, 14 and 22.
- **Baselines:** FastFit-large 5-shot (85.2 / 93.4; fine-tuned, a ceiling); tarski full-data 93.1
  (Banking77).
- **Expected signal:** the late-interaction branch beats mean-pool probes by 3–6 points at k ≤ 5; the gap
  closes by k = 10. If it does not, that is useful too: it would mean frozen mid-depth token states are
  not match-ready.
- **Cost:** about 20 A10 minutes; the training sets are tiny and caching dominates.

---

## 6. Day-0 routes from descriptions with a local LLM teacher and a refine loop

**Status (2026-09-27): implemented, smoke passes (Qwen3-0.6B on CPU), queued.** `explore/lit_dm_synthroutes.py`, queue line `synthroutes` (A100; Qwen/Qwen3-8B through transformers, thinking disabled).
- **Arms:** synth, synth+none, refine, refine+real5, real5, real_full.
- **Refine pairs:** the most-confused intent pairs on synthetic held-out data, topped up with the nearest class centroids.
- **Cache:** generated data is saved to `<out>_data.json`, so a rerun skips generation.

**Sources**
- Maheshwari & El Haddad, "Curate-Train-Refine: A Closed-Loop Agentic Framework for Zero Shot
  Classification", arXiv 2601.16530, Jan 2026 (read: HTML).
  - **Loop:** an LLM (GPT-5) curates and generates data, inspects the small model's errors, and
    synthesises targeted examples. The classifier is SetFit on mpnet-base (110M); EuroBERT is also
    tried.
  - **Zero-shot results:** AG News 82.6 vs GLiClass 68.8; SST-5 47.0 vs 43.8; Emotion 50.9 vs 51.9.
  - No intent datasets.
- Peng & Shang, "Incubating Text Classifiers Following User Instruction with Nothing but LLM", arXiv
  2404.10877, 2024 (read: abstract). Instruction-to-data LLM; handles "Other" and interdependent classes.
- Vajjala & Shimangaud, "Text Classification in the LLM Era – Where do we stand?", arXiv 2502.11830, 2025
  (read: abstract). Over 32 datasets, synthetic data from several LLMs beats zero-shot open LLMs.
- Zero-shot LLM reference: "Selecting Open-Weight Language Models for Zero-Shot Intent Classification: A
  Systematic Evaluation of 41 Models", arXiv 2607.27421, Jul 2026 (read: HTML summary). The best 7B
  instruct model reaches an aggregate of about 0.82 over CLINC150, Banking77 and others; per-dataset
  numbers were not checked.

**Idea to borrow.**
1. A route defined only by a description exists on day 0 as a branch trained on examples written by a
   local LLM, which trains in seconds on cached trunk states.
2. One refine round then targets the branch's confusions.
3. Real labels replace synthetic ones as they arrive; learning E's active distillation is the natural
   next stage.

**Novelty.** Low (ZeroGen/SuperGen 2022, Incubator, CTR). The value is a same-data comparison of three
day-0 mechanisms on one trunk: one-way cloze (geometry 3), a zero-shot trunk (entry 3), and synthetic
branches (this entry).

**Experiment**
- **Round 1:**
  - A local instruct LLM (e.g. Qwen3-8B, bf16, vLLM) writes 25 utterances per CLINC intent from the
    name and a one-line description: 3,750 utterances, plus 500 generic oos-like ones.
  - Train blocks@11+2 and probe@22; test on real CLINC.
- **Round 2:** find the 20 most-confused intent pairs on held-out synthetic data and generate 10
  contrastive examples each.
- **Arms:** synthetic only; synthetic plus 5 real shots; real full.
- **Metrics:** in-scope intent accuracy; oos F1.
- **Baselines:**
  - real-label branches: in-scope intent about 93%, 151-way about 84–87%;
  - zero-shot LLM about 0.82, at more than 100× the per-message cost;
  - entry 3's gte zero-shot.
- **Expected signal:** 70–80% in-scope from synthetic data only; +2–4 points from refinement; the oos
  "Other" class stays weak without entry 4.
- **Cost:** 10–15 A10 minutes of generation plus about 5 of training.

---

## 7. Paired encoder vs decoder trunk (Ettin), plus a causal trunk with bidirectional branches

**Status (2026-09-27): implemented, smoke passes (Ettin 17m pair on CPU), queued.** `explore/lit_dm_ettin.py`. Queue lines `ettin_enc` (A10), `ettin_dec_causal` (A100) and `ettin_dec_bibranch` (A10).
- **CausalTrunk:** a subclass inside the script (tarski/ is untouched). It reproduces the stock decoder forward exactly (max abs diff 0).
- **Masking check:** a bidirectional branch changes token 0 when a later token changes; a causal branch does not. Checked with and without padding.
- **Scope:** the DeBERTa-v3 arm is not implemented (it needs a non-ModernBERT trunk wrapper).

**Sources**
- Weller et al., "Seq vs Seq: An Open Suite of Paired Encoders and Decoders" (Ettin), ICLR 2026.
  https://arxiv.org/abs/2507.11412 · https://github.com/jhu-clsp/ettin-encoder-vs-decoder (read: HTML and
  README).
  - **Suite:** encoders and decoders from 17M to 1B with identical data and recipe. Data and 250+
    checkpoints are open.
  - **Drop-in:** `jhu-clsp/ettin-encoder-150m` has the ModernBERT architecture (22 layers, 768 hidden,
    ModernBERT tokenizer).
  - **GLUE average:** 88.9 vs 88.4 for ModernBERT-base (400m: 90.8 vs 90.4 for ModernBERT-large).
  - **MNLI, encoder vs decoder at 150M:** 89.2 vs 85.6. A 400M encoder beats a 1B decoder (91.3 vs 89.9).
  - **Cross-objective conversion:** it underperforms native models.
  - **Limit:** all results are fine-tuned; there is no frozen or probing evaluation.
- Antoun, Sagot, Seddah, "ModernBERT or DeBERTaV3? Examining Architecture and Data Influence on
  Transformer Encoder Models Performance", IJCNLP-AACL 2025. https://arxiv.org/abs/2504.08716 (read:
  abstract). With pretraining data controlled, DeBERTaV3 is more sample-efficient and ModernBERT is
  harder to fine-tune. The GLiClass authors report the same backbone preference.
- Context: the open System One models are mostly decoders:
  - JevK5: Qwen3.5-4B/9B with a distilled LoRA and an option-letter logit readout.
    https://github.com/allebee/jevk5 (read: README). It reports second of 76 on JevBench v1.4 and hard-tier
    ECE 0.054.
  - CLM-8B (see the flags below).
  - Kev (https://github.com/jaredpalmer/kev, not read).

**Idea to borrow.** The Ettin pair is the only controlled way to ask whether a frozen-trunk decision
harness needs bidirectional attention:
- **Encoder side:** a free drop-in check that the results are not ModernBERT-specific.
- **Decoder side:** read-once comes for free, since the message's KV cache is reusable and questions go
  in as a suffix. That is what the one-way laya work (TRIAGE 1) tries to retrofit.
- **Hybrid:** a **causal frozen trunk with bidirectional branch layers**, i.e. copies of decoder layers
  [k, k+d) run with a full mask. Prefix-cacheable, with encoder-like decisions.

A quick search found only whole-model conversions (LLM2Vec, Causal2Vec, NV-Embed), not this hybrid.

**Experiment**
- **Arms:**
  - ettin-encoder-150m (only the model id changes);
  - ettin-decoder-150m (core change: causal mask and mean or last-token pooling in `Trunk.taps`);
  - decoder trunk plus bidirectional blocks branch.
- **Optional DeBERTa-v3-base arm** for the few-label regime; it needs a trunk wrapper, since it has
  relative positions and no RoPE.
- **Grid:** Banking77, CLINC150, typed-decisions × probe depth scan × blocks@{8,14,18}+{1,2}.
- **Baseline:** the ModernBERT-base table.
- **Expected signal:**
  - Ettin encoder within ±1 point of ModernBERT-base.
  - Decoder probes 3–8 points lower, worst on typed-decisions, where early JSON tokens never see later
    fields.
  - Bidirectional branches recover most of that gap. If they do, a decoder trunk with prefix caching
    becomes viable for growing Slack threads (systems 2).
- **Cost:** 60–90 A10 minutes for three trunks × three datasets.

---

## Novelty flags for existing tarski claims and queued ideas

- **CLM-8B** (Contrastive-LM, Sept 2026). https://github.com/Contrastive-LM/CLM (read: README);
  MarkTechPost 2026-09-23.
  - **Design:** a **frozen Qwen3-8B** (last-token pooling) plus a 20M trainable projection head, trained
    with bidirectional InfoNCE. States and candidate actions are embedded separately and cached, behind a
    TypeSafe-compatible API.
  - **Claims:** on par with Jev zero-shot at up to 9× lower latency.
  - **For the thesis:** this is the closest System One instance of "frozen trunk + small trained head".
    Add it to `novelty_check.md` claim 3. The differences are one final-layer readout, bi-encoder
    candidates, and no per-task depth or branch layers.
- **"Mixture of Expert Heads for Calibrated Decision Encoders: A Frozen-Encoder MoE on Laya"** (dev.to,
  2026).
  https://dev.to/vishalmysore/mixture-of-expert-heads-for-calibrated-decision-encoders-a-frozen-encoder-moe-on-laya-3g0e
  (read).
  - **Design:** laya's encoder frozen; 26.5M heads copied per domain and trained on synthetic
    rule-labelled data; a prompted router picks the head.
  - **Result:** 59.6 → 67.3% on 312 author-labelled questions. The evidence is weak (the author tuned the
    router on the test set).
  - **For the thesis:** the same "freeze the decision trunk, fork the head" move. Cite it.
- **Verdict 2.0** is an external **same-size** (ModernBERT-base) typed-decisions model at 77.10% vs
  laya's 76.60%. The typed comparison in the thesis should cite it; its training data is undisclosed.
- **SingGuard RI-Mask** ("SingGuard: A Policy-Adaptive Multimodal LLM Guardrail with Dynamic Reasoning",
  SingGuard Team, arXiv 2606.22873, June 2026).
  - **Mechanism:** rule branches packed into one pass. Each sees the shared content prefix but not the
    other rules, with branch-local position ids.
  - **Result:** more than 5× faster at 30 rules, "near-lossless".
  - Details come from a search snippet of the PDF; the abstract does not mention RI-Mask.
  - **Relevance:** prior art for the packing part of TRIAGE 1 (one-way questions) and passenger tokens.
    Because it is a decoder, the content never attended to the rules anyway. Tarski's distinct part
    remains a **bidirectional** encoder converted with no retraining.
- **GLiNER Guard**'s shared-weight bi-encoder with cached labels (74.3 vs 73.8 F1 against the uni-
  encoder) is new support for geometry 12 (description-anchored routes).

## Scanned, not proposed (and why)

- **NeoBERT** (arXiv 2502.19587), **EuroBERT**, **mmBERT** (arXiv 2509.06888): small English gains over
  ModernBERT, or a multilingual focus. mmBERT-base is the trunk to try if parity with multilingual laya
  matters.
- **ModernBERT-Large-Instruct** (arXiv 2502.03793): already covered by geometry 3.
- **"Should We Still Pretrain Encoders with Masked Language Modeling?"** (ICLR 2026, arXiv 2507.00994):
  a pretraining recipe (CLM then MLM). Not actionable without pretraining.
- **GLiClass as a zero-shot model for our datasets:** weak with many labels (Banking77 F1 0.356), and the
  authors say so. Used only as a trunk in entry 3.
- **GLiNER2** (arXiv 2507.18546): schema-driven classification and extraction in one pass, 205M. A
  reasonable typed-decisions baseline, but not an idea to borrow.
- **Proper scoring / RLCD:** TypeSafe names RLCD but publishes no method. Nothing beyond learning A/B.
- **Hierarchical:** Plaud et al., "Revisiting Hierarchical Text Classification: Inference and Metrics"
  (CoNLL 2024, arXiv 2410.01305) find simple baselines competitive with proper hierarchical inference.
  Domain/intent joint decoding is already in the syndrome run (+0.1 point in-scope).
- **Distillation:** "Bridge-Garden Dilemma" (ICML 2026, arXiv 2605.26246) concerns generation; "LLM on a
  Budget" active KD (arXiv 2511.11574) overlaps learning E.
- **"Soft Contextualized Encoder For User Defined Text Classification"** (arXiv 2601.03450): label-set
  contextualisation via a query soft prompt. News only (Yahoo 44.3%), weak evidence; a possible extension
  of geometry 12.
- **"Domain Restriction via Multi SAE Layer Transitions"** (arXiv 2605.11920): cross-layer OOD on CLINC,
  overlapping depth-disagreement (our AUROC 0.914).
- **Avey-B** (arXiv 2602.15814, attention-free encoder), **Causal2Vec** (arXiv 2507.23386): not relevant
  to a frozen ModernBERT harness right now.
