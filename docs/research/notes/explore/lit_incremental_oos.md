# Literature scan: incremental routes, out-of-scope, shift and drift (2026-09-27)

Scope: recent work (2024-2026, plus a few older foundations missing from our notes) on class-incremental
learning over frozen pre-trained models, open-set / out-of-scope (OOS) intent detection, OOD scores on
frozen features, label-shift estimation, drift alarms and retraining triggers, and conformal routing with
abstention. Each entry gives the idea worth borrowing and one cheap experiment on the harness.

Quoting policy: papers are paraphrased. Every URL was opened or returned by a search during this scan.
Numbers come from the paper's abstract or HTML page unless marked. "Novelty" means relative to our notes
and a quick search, not a full prior-art check.

## What our notes already cover (not re-proposed here)

- Saerens EM prior correction on the supervised `oos` branch (skeptic idea 1; oos-F1 0.29 -> 0.57). BBSE
  is cited there.
- Label-free OOS scores: Mahalanobis and kNN (geometry 6, farfield 3); cross-depth disagreement (farfield 4:
  MOOD, BLOOD); syndrome and consistency scores (farfield 1); layer-wise aggregation (Darrin et al.); ViM.
- LDA / sufficient-statistics heads with exact merge, unlearning and new labels (geometry 10, implemented in
  `explore/geometry_sufficient.py`).
- Weight-imprinted route addition (learning H, `explore/learning_route_imprinting.py`).
- Outlier exposure with other tasks' messages as negatives (learning J, `explore/learning_outlier_exposure.py`).
- Description-anchored zero-shot routes (geometry 12).
- Dual-encoding threshold re-classification (arXiv 2405.19967) and DROID (arXiv 2510.14110), in skeptic
  and frozen_backbone_classification.md.

## Our own numbers these entries build on

From `results/tarski/lambda/explore_syndrome.log` and `explore_depthscan.json`, CLINC150 test set: 4,500
in-scope and 1,000 OOS messages.

| Signal | Best AUROC | At depth |
|---|---|---|
| Supervised stack | 0.957 | 4 |
| Entropy | 0.937 | 4 |
| MSP | 0.929 | 4 |
| kNN (L2-normalised features) | 0.906 | 4 |
| Mahalanobis (z-scored features) | 0.845 | 4 |

- Mahalanobis in the depth scan is 0.861 at depth 1 and falls to about 0.79 at depths 18-20.
- LDA-style nearest-class-mean accuracy on the in-scope intents is 0.89-0.92. The linear probe reaches 0.929.
- At validation-chosen thresholds, OOS recall is low (0.1-0.35) for almost every score. The CLINC split
  has 1.6% OOS in train, 3.2% in validation and 18.2% in test.

---

## Ranked entries

### 1. Kernelised sufficient-statistics routes (KLDA / RanPAC): add a route in closed form and possibly beat the probe

**Status (2026-09-27): implemented.** `explore/lit_oos_klda.py`, queued as `lit_klda` (A10, ~20 min); the
smoke run passes. It includes RanPAC (ReLU projection plus ridge), the class-incremental protocol (LDA and KLDA
exact; a probe retrained on everything seen; a SEQ*-style no-replay probe), and add-a-route-from-k-shots
using additive statistics. It also reports the CLINC OOS AUROC in LDA and KLDA space. The "probe" arm is a
batched logistic regression on z-scored mean pools, a proxy for tarski's `ProbeBranch`.

**Papers**
- Momeni, Mazumder & Liu, "Continual Learning Using a Kernel-Based Method Over Foundation Models" (KLDA),
  AAAI 2025. https://arxiv.org/abs/2412.15571 · code https://github.com/SalehMomeni/KLDA. Extended version:
  "Achieving Upper Bound Accuracy of Joint Training in Continual Learning", https://arxiv.org/abs/2502.12388.
- Successor: Momeni, Xiao & Liu, "AnaCP: Analytic Contrastive Projection", Nov 2025,
  https://arxiv.org/abs/2511.13880. It adapts features analytically, with no gradients.
- Antecedents:
  - McDonnell et al., "RanPAC: Random Projections and Pre-trained Models for Continual Learning", NeurIPS
    2023, https://arxiv.org/abs/2307.02251. A frozen random projection with ReLU, then a ridge / Gram-inverse
    head.
  - Prabhu et al., "RanDumb", NeurIPS 2024, https://arxiv.org/abs/2402.08823. Random Fourier features
    (RFF) plus a streaming linear classifier.

**Idea to borrow.** Put a fixed, seeded RFF map (an approximate RBF kernel, D≈5000) between the frozen
features and an LDA head, with class means and one shared covariance.
- Adding a route adds one class's statistics.
- Because the RFF map is data-independent, geometry 10's guarantees survive unchanged: exact merge across
  users, exact unlearning, and no gradient descent.

**Claims.** Text-specific: CLINC (10 tasks), Banking77 (7 tasks), HWU64 and DBpedia, on frozen BART-base
mean pools. Results from the HTML tables:

| Dataset | NCM | LDA | KLDA | KLDA ensemble of 5 | "Joint" fine-tune |
|---|---|---|---|---|---|
| CLINC | 83.6 | 93.7 | 95.9 | 96.6 | 95.3 |
| Banking77 | 71.1 | 89.1 | 92.2 | 93.0 | 91.4 |

KLDA trains in about 10 s. Two caveats:
- The "joint" upper bound is weak. Our own full fine-tune reaches 94.0 on Banking77, so "matches joint
  training" overstates the result.
- The bandwidth σ is sensitive per backbone: the README lists values from 1e-2 to 5e-6.

The LDA-to-KLDA gain (+2.2 on CLINC, +3.1 on Banking77) is the useful signal.

**Why it matters here.**
- Our probes are 87-89 on Banking77 and 0.929 on CLINC in-scope. Plain LDA in the depth scan is 0.89-0.92.
- If RFF adds its reported +2-3 points on ModernBERT taps, a closed-form, add-a-route head matches or beats
  the SGD probe.
- The same RFF statistics give a Mahalanobis OOS score in kernel space, related to Kernel PCA OOD (entry 3).

**Novelty.** Low as a method: KLDA already runs on CLINC and Banking77. What is new here is narrow:
- per-depth KLDA on intermediate layers of a frozen encoder;
- one shared RFF seed per trunk, so statistics stay additive across users (a local-first property);
- one statistic set serving both routing and OOS.

**Experiment.** Add `--rff D --sigma s --l2norm` to `explore/geometry_sufficient.py`.
- **Datasets and depths.** Banking77 and CLINC in-scope intents, at depths {4, 8, 11, 16, 22}.
- **Arms:**
  1. NCM;
  2. LDA (current);
  3. LDA on L2-normalised features;
  4. KLDA with D ∈ {2000, 5000};
  5. a KLDA ensemble of 5 seeds;
  6. RanPAC (ReLU random projection plus ridge);
  7. the tarski probe;
  8. the best blocks branch.
- **Class-incremental protocol.** KLDA's splits: CLINC 10×15, Banking77 7×11. Report final and average
  incremental accuracy, and the new-route-from-k-examples curve (k = 1, 5, 10, 20) that the script already
  has.
- **Baseline to beat:** the probe at the same depth (0.929 on CLINC; about 0.85-0.88 on Banking77).
- **Expected signal:** KLDA above LDA by 1.5-3 points and within about 1 point of, or above, the probe.
- **Kill criterion:** under +1 point over LDA at every depth.
- **Cost:** about 5 A10 minutes (one cached trunk pass, then CPU/GPU linear algebra on 5000×5000
  matrices).

**Capability:** add a route without retraining the others, with exact merge and unlearning.

---

### 2. Conformal selection for auto-routing: route when the false-route rate is provably ≤ q, escalate the rest

**Status (2026-09-27): implemented.** `explore/lit_oos_conformal_select.py`, queued as
`lit_conformal_select` (A10, ~12 min); the smoke run passes.
- **Arms:** Conformal Labeling; the naive FDR-search threshold; label-shift-weighted selection (EM or true
  prior); batch-online selection (BH within batches of 200, prior re-estimated as traffic arrives; this is
  not OCS-ARC's online BH); and a heuristic predicted-class weight.
- **Leak fix.** Validation is split into a half that picks the probe's L2 and temperature and a half for
  calibration. Using one set for both was an exchangeability leak, found in the smoke run.
- **Limitation.** EM cannot up-weight out-of-scope errors when the decision has no out-of-scope label;
  entry 4 is the fix.

**Papers**
- Huang et al., "Selective Labeling with False Discovery Rate Control" (Conformal Labeling), Oct 2025,
  https://arxiv.org/abs/2510.14581.
  - **Method.** A test item's p-value compares its uncertainty score with the calibration items the model
    got *wrong*. Benjamini-Hochberg (BH) at an adjusted level then selects the items to trust.
  - **Results.** ImageNet: 79.9% power at 9.97% FDR (α = 10%). MMLU: about 55% of items auto-labelled at
    α = 10%.
  - **Baseline behaviour.** Naive confidence thresholds chosen on calibration FDR often violate the target.
- Jin & Candès, "Selection by Prediction with Conformal p-values", JMLR 2023,
  https://arxiv.org/abs/2210.01408. The foundation.
- Gui, Jin & Ren, "Conformal Alignment", NeurIPS 2024, https://arxiv.org/abs/2405.10301. The same idea for
  deciding when to trust foundation-model outputs.
- Streaming variant: Liu et al., "Online Conformal Selection with Accept-to-Reject Changes", Aug 2025,
  https://arxiv.org/abs/2508.13838. FDR control at every time step when accepts are irreversible, which is
  the case for a routed Slack message.
- Label-shift weighting: Podkopaev & Ramdas, "Distribution-free uncertainty quantification for
  classification under label shift", UAI 2021, https://arxiv.org/abs/2103.03323. Reweight calibration
  scores by π_test(y)/π_cal(y), with the weights estimated from unlabelled traffic.

**Idea to borrow.** Replace "auto-route if confidence > τ" with a guarantee: among auto-routed messages,
the expected fraction misrouted is at most q (for example 2%). Everything else is escalated to a human, to
laya, or to a clarification question (entry 6).
- The only inputs are a labelled calibration set, which tarski already holds out for temperature fitting,
  and any score (MSP, entropy, the supervised stack from the syndrome script).

**Claims.** The guarantee is standard, distribution-free and finite-sample, but it needs exchangeability.
The paper lists label shift as unhandled (limitations, Appendix I.1). CLINC is exactly that case: 3.2% OOS
in validation, 18.2% in test. The weighted variant is therefore the interesting arm here.

**Novelty.** The mechanism is known. A hobby repository applies conformal sets to banking-intent routing
(https://github.com/EnzoCanonero/conformal-banking). Not found in a quick search:
- FDR-controlled auto-routing for intent/ticket routing;
- combined with label-shift-weighted calibration (using our EM prior estimate);
- evaluated on CLINC's real prior shift.

**Experiment.**
- **Datasets.** CLINC 151-way (OOS counted as a class, so routing an OOS message to an intent is an error)
  and Banking77.
- **Scores.** MSP, entropy and stack at depth 4, plus the best branch.
- **Calibration.** The validation split: about 20 per intent on CLINC, about 13 on Banking77.
- **Arms:**
  1. MSP threshold chosen on validation for 98% precision (baseline);
  2. Conformal Labeling / BH;
  3. the same with Podkopaev-Ramdas weights from the Saerens EM prior;
  4. the online variant on a shuffled test stream.
- **Metrics:** realised false-route rate among auto-routed messages at q ∈ {1, 2, 5, 10}%, auto-route rate
  (power), and escalation rate.
- **Expected signal.** On Banking77, both baseline and conformal hold the target; conformal has similar or
  slightly lower power. On CLINC, the unweighted versions *violate* q because OOS is 6× more common in test
  than in calibration, and the weighted version restores it.
- **Cost:** under 5 A10 minutes, all post-hoc on cached logits.

**Capability:** "route or escalate" with a stated error guarantee.

---

### 3. Mahalanobis++: L2-normalise before Mahalanobis (and kernel-PCA reconstruction as a second arm)

**Status (2026-09-27): implemented, run first.** `explore/lit_oos_mahapp.py`, queued as `lit_mahapp` (A10,
~5 min); the smoke run passes.
- **Arms:** Mahalanobis on raw, z-scored, L2, centred-L2 and z-then-L2 features; relative Mahalanobis;
  KPCA with the cosine kernel and with cosine plus RFF; kNN; MSP.
- **Also reported:** the norm-score Spearman diagnostic; OOS F1 at the validation threshold, after
  calibration plus EM, and at the oracle rate.

**Papers**
- Mueller & Hein, "Mahalanobis++: Improving OOD Detection via Feature Normalization", ICML 2025,
  https://arxiv.org/abs/2505.18032 · code https://github.com/mueller-mp/maha-norm.
  - **Diagnosis.** Across 44 ImageNet models, feature norms vary strongly and correlate with the
    Mahalanobis score whether or not a sample is OOD, which breaks the Gaussian assumption.
  - **Fix.** L2-normalise before fitting the means and covariance, and again at test time. Mean FPR drops
    from 42.5% to 34.9%, and by 10.9 points on NINCO.
- Fang et al., "Kernel PCA for Out-of-Distribution Detection", NeurIPS 2024,
  https://arxiv.org/abs/2402.02949. Cosine kernel plus RFF of a Gaussian kernel, scored by PCA
  reconstruction error. The follow-up is https://arxiv.org/abs/2505.15284.
- Relative-Mahalanobis baseline, older: Ren et al., "A Simple Fix to Mahalanobis Distance for Improving
  Near-OOD Detection", 2021, https://arxiv.org/abs/2106.09022.
- Supporting evidence: Barkley et al., "Scaling Pretrained Representations Enables Label-Free OOD
  Detection Without Fine-Tuning", May 2026, https://arxiv.org/abs/2605.05638. Over 59 vision and language
  backbone-task pairs, detector choice matters less as the frozen representation improves.

**Why it matters here.** Our Mahalanobis is the worst label-free score (0.80-0.86), and it gets *worse*
with depth. kNN on L2-normalised features of the same states scores 0.906.
- The implementation (`farfield_syndrome.py:137`, geometry depth scan) z-scores each dimension but never
  normalises each sample.
- That is exactly the pathology Mahalanobis++ describes. ModernBERT's massive-activation dimensions make
  per-sample norm variation likely.

**Novelty.** None as a method (vision, 2025). It is a probable bug-level fix for our label-free OOS numbers
and cheap to confirm.

**Experiment.**
- **Setup.** CLINC150, depths 1-22, in-scope training statistics only.
- **Arms:**
  1. current Mahalanobis;
  2. Mahalanobis++;
  3. Mahalanobis++ plus relative Mahalanobis;
  4. KPCA with a cosine kernel;
  5. KPCA with cosine plus RFF;
  6. kNN (current).
- **Diagnostic.** Spearman correlation between feature norm and Mahalanobis score on in-scope test messages
  (the paper's check).
- **Metrics:** AUROC, FPR@95, and OOS F1 after Saerens correction on the thresholded score.
- **Baselines to beat:** kNN at 0.906 and MSP at 0.929 (depth 4).
- **Expected signal:** Mahalanobis++ at 0.90-0.94. The depth trend should flatten, and the norm-score
  correlation should drop.
- **Kill criterion:** a gain under 0.02 AUROC.
- **Cost:** about 2 A10 minutes.

**Capability:** a better label-free "none of the above" gate for user routes that have no OOS labels. It
reuses entry 1's statistics.

---

### 4. Open-set label shift: estimate the no-route fraction of live traffic without any OOS labels

**Status (2026-09-27): implemented.** `explore/lit_oos_openset_shift.py`, queued as `lit_openset_shift`
(A10, ~10 min); the smoke run passes.
- **Estimators:**
  - Ye et al.'s stage 2 (EM over K+1 classes);
  - stage 3 implemented as adjusted classify-and-count, not the paper's exact formula;
  - PULSE-lite: BBE on a cross-fitted logistic discriminator. PULSE's CVIR classifier is not implemented.
- **References:** the Banking77 reference set and a noise-mixed pseudo-OOD set, plus a supervised
  Saerens run in the same job.
- **Smoke run:** the noise pseudo-OOD reference is far too easy (it estimates 0% OOS). Treat that arm as a
  negative control.

**Papers**
- Ye, Tsuchida, Petersson & Barnes, "Open Set Label Shift with Test Time Out-of-Distribution Reference",
  CVPR 2025, https://arxiv.org/abs/2505.05868 · code https://github.com/ChangkunYe/OpenSetLabelShift.
  A retrain-free three-stage estimator:
  1. estimate the source ID/OOD ratio from an OOD detector's mean scores on source data and on a reference
     OOD set (real, or pseudo-OOD made by mixing noise into ID features);
  2. run EM over K+1 classes on unlabelled target data, which yields the target priors *and* the OOD
     fraction;
  3. correct that fraction for detector bias.

  It beats BBSE, RLLS, MLLS and MAPLS on 91.7-100% of vision settings. It needs a reasonably calibrated ID
  classifier.
- Garg et al., "Domain Adaptation under Open Set Label Shift" (PULSE), NeurIPS 2022,
  https://arxiv.org/abs/2207.13048. It reduces the problem to positive-unlabelled (PU) learning, with the
  novel-class fraction estimated by BBE from Garg et al., NeurIPS 2021, https://arxiv.org/abs/2111.00980.
- Liu et al., "Semiparametric Learning from Open-Set Label Shift Data", Sep 2025,
  https://arxiv.org/abs/2509.14522. A statistics paper with confidence intervals for the novel proportion.
  Its real-data evidence is weak: one tabular dataset.

**Idea to borrow.** Our Saerens result needed a *supervised* OOS branch. User-defined routes almost never
have "none of the above" labels. Open-set label shift estimation instead uses the in-scope classifier plus
any OOD score (entry 3, kNN, MSP), and reads unlabelled traffic, to give:
- (a) the fraction of traffic that fits no route, which is a monitoring number in its own right;
- (b) the re-estimated route priors;
- (c) a threshold set by that fraction instead of by an OOS-less validation set.

Learning J's "free outliers" (other tasks' messages) can serve as the OOD reference set.

**Claims.** Evidence so far is vision only (CIFAR, ImageNet-200). Text intent OOS is near-OOD, so detector
bias (stage 3) will matter more. A quick search found no application to intent OOS.

**Experiment.**
- **Setup.** CLINC with all OOS labels removed from training: a 150-way in-scope probe plus an OOD score.
  The reference OOD set is either Banking77 messages or noise-mixed pseudo-OOD.
- **Test pools.** The real test set (18.2% OOS), plus resampled pools at 5, 10, 30 and 50% OOS.
- **Arms:**
  1. Ye et al.'s three stages;
  2. PULSE-lite: a train-in-scope vs test-pool discriminator on pooled states, then BBE;
  3. the validation-threshold baseline.
- **Metrics:** absolute error of the estimated OOS fraction, OOS F1 and 151-way macro-F1 after thresholding
  at the estimated fraction.
- **Reference:** the supervised Saerens result (oos-F1 0.57).
- **Expected signal:** fraction error under 4 points on the real split, and oos-F1 of 0.5-0.65 with *zero*
  OOS labels. A detector at AUROC 0.93 thresholded at the right prior should land around there.
- **Cost:** about 5 A10 minutes, post-hoc.

**Capability:** OOS handling for user routes without negative labels, and a "share of traffic fitting no
route" gauge that also feeds entries 5 and 10.

---

### 5. Label-free harmful-drift alarm plus a label-efficient confirmation (retrain trigger)

**Status (2026-09-27): implemented.** `explore/lit_oos_drift.py`, queued as `lit_drift` (A10, ~10 min); the
smoke run passes.
- **Alarm.** Amoukou-style proxy-risk tests (mean and quantile) with PrPl-EB confidence sequences (Waudby-Smith
  & Ramdas) and an empirical-Bernstein source bound.
- **Baselines:** an OOS-rate test, a calibrated MSP CUSUM and an MMD permutation test.
- **Confirmation:** active inference (Zrnic & Candès), plus prediction-powered uniform sampling.
- **Not implemented:** WATCH (it needs online labels) and the suitability filter.
- **Smoke caveat:** the confidence-sequence monitors cannot fire, because the source bound is loose on
  about 150 messages.

**Papers**
- Amoukou et al., "Sequential Harmful Shift Detection Without Labels", NeurIPS 2024,
  https://arxiv.org/abs/2412.12910.
  - **Method.** Train an error estimator on labelled source data to predict when the primary model is
    wrong; use its outputs as a proxy for the error rate. Alarm when the lower confidence bound on
    production proxy-risk exceeds the source upper bound plus a tolerance. The bounds are Podkopaev-Ramdas
    predictably-mixed empirical-Bernstein confidence sequences, so false alarms are controlled at any
    stopping time.
  - **Results.** Tabular, image and Folktables data. The quantile detector has power 0.83 with false
    discovery proportion 0.11.
- Zrnic & Candès, "Active Statistical Inference", ICML 2024 (oral),
  https://proceedings.mlr.press/v235/zrnic24a.html. Label where the model is uncertain and get valid
  confidence intervals, for example on live accuracy, with far fewer labels.
- Successors:
  - Brawand et al., "Active Multiple-Prediction-Powered Inference", May 2026,
    https://arxiv.org/abs/2605.08429. 10-40% narrower intervals.
  - Ashouritaklimi et al., "Prediction-Powered Active Testing", Jul 2026,
    https://arxiv.org/abs/2607.08347.
- Related:
  - Prinster et al., "WATCH: weighted-conformal martingales", ICML 2025,
    https://arxiv.org/abs/2505.04608. It adapts to benign covariate shift and alarms on harmful shift, but
    needs online labels.
  - Pouget et al., "Suitability Filter", ICML 2025, https://arxiv.org/abs/2505.22356. A hypothesis test
    that accuracy on unlabelled user data dropped by no more than a margin.
  - Lamaakal et al., "Drift2Act", ICLR 2026 workshop, https://arxiv.org/abs/2603.08578. Maps alarms to
    actions (recalibrate, abstain, retrain). Workshop-level.

**Idea to borrow.** Tarski already computes several label-free error signals per message for free: MSP,
entropy, cross-depth disagreement, syndrome inconsistency, OOS score. A small stacked error estimator over
these, trained on validation to predict "branch is wrong", feeds a sequential test. That gives a principled
"your routing got worse" alarm with bounded false alarms. When it fires, active inference picks about 30-50
messages for the user to label and returns a confidence interval on current accuracy. Retrain only if the
interval's upper end is below the accepted level.

**Novelty.** The components are known and were evaluated on tabular and vision data. Not found for text
routing, or using a multi-decision bundle's consistency signals as the error proxy. The workflow is new
and, for users, probably the most valuable of the drift ideas.

**Experiment.** Simulate streams from CLINC test and validation messages:
- **S1, harmful.** 20 held-out intents start arriving at 20% after message 1000. They are misrouted, because
  the branches were trained without them.
- **S2, benign.** In-scope priors shift (Dirichlet α = 0.3); accuracy is roughly unchanged.
- **S3, covariate.** Keyboard-typo noise at 10-20% of characters.
- **Monitors:**
  1. the Amoukou-style proxy-risk confidence sequence over the stacked error estimator;
  2. a mean-MSP CUSUM;
  3. an MMD two-sample test on pooled depth-4 states (the "Failing Loudly" baseline, Rabanser et al. 2019);
  4. OOS-rate monitoring from entry 4.
- **Metrics:** detection delay, false alarms on S2 and on a no-shift stream, and labels needed for active
  inference to certify the accuracy drop at 90% confidence.
- **Expected signal.** The proxy-risk test fires on S1 and S3 and stays quiet on S2. MMD fires on all three,
  because it cannot tell benign from harmful shift. Active inference needs about 2-3× fewer labels than
  uniform sampling.
- **Cost:** about 5 A10 minutes (one trunk pass over noised copies; the rest is CPU).

**Capability:** drift detection and a principled "retrain now" trigger under a label budget.

---

### 6. Conformal clarification sets for routing, with class-conditional and open-set extensions

**Status (2026-09-27): implemented, with a substitution.** `explore/lit_oos_cicc.py`, queued as `lit_cicc`
(A10, ~12 min); the smoke run passes.
- **Implemented:** marginal, classwise and clustered conformal (Ding et al.); LAC and APS scores; the
  route-or-clarify bands.
- **laya on the middle band.** It uses the English checkpoint, falls back to `typed-decisions`, or runs
  without laya if neither is cached; the JSON records which.
- **Not implemented: the CGTC Good-Turing joker.** Good-Turing needs labels seen exactly once, which a
  routing dataset does not have, so it is degenerate here. A conformal outlier gate (Bates et al. 2023) on
  Mahalanobis++ replaces it; 30 CLINC intents are held out to test it.

**Papers**
- den Hengst, Wolter, Altmeyer & Kaygan, "Conformal Intent Classification and Clarification" (CICC),
  NAACL Findings 2024, https://arxiv.org/abs/2403.18973.
  - **Policy.** Build a conformal set: one label, act on it; 2 to th labels (th = 7 by default), ask a
    clarifying question over the set; more than th, reject.
  - **Results.** Seven datasets including CLINC150 and Banking77, with ≥95% coverage.
    - Mean set size is 2.5-3.5.
    - 25-98% of queries resolve without clarification.
    - OOS F1 is 0.07-0.76 for vanilla CICC and 0.90-0.91 when OOS labels are used in calibration.
- Ding, Angelopoulos, Bates, Jordan & Tibshirani, "Class-Conditional Conformal Prediction with Many
  Classes" (clustered conformal), NeurIPS 2023, https://arxiv.org/abs/2306.09335. It pools classes with
  similar score distributions, so per-class coverage holds with about 10-20 calibration examples per class.
  That is our regime.
- Xie, Zhou, Liang, Favaro & Sesia, "Conformal Inference for Open-Set and Imbalanced Classification", Oct
  2025, https://arxiv.org/abs/2510.13037. Good-Turing-based p-values add a "joker" (unseen class) to the set
  with finite-sample coverage even when new labels appear. This is the formal version of "a new route
  showed up".
- Caution: Han & Qu, "The Label Complexity of Class-Conditional Coverage under Distribution Shift", Jul
  2026, https://arxiv.org/abs/2607.18088. Marginal coverage can hide about 70% worst-class coverage under
  shift, and per-class validity without target labels is impossible in general.
- Escalation-target precedent: Zaera et al., "Efficient Out-of-Scope Detection in Dialogue Systems via
  Uncertainty-Driven LLM Routing", ACL 2025 Industry, https://aclanthology.org/2025.acl-industry.25/. Only
  uncertain messages go to a fine-tuned LLM; the system is in production.

**Idea to borrow.** A three-way routing policy (route / clarify-or-cascade / escalate). The middle band
goes to laya, which is already local, or to a "did you mean #billing or #refunds?" prompt. Clustered
conformal keeps small routes from being systematically under-covered. The open-set joker gives a guaranteed
"new route?" outcome.

**Novelty.** CICC already did vanilla conformal on CLINC and Banking77, so the gap is narrow. Not found:
- clustered or class-conditional coverage for intent routing;
- the Good-Turing open-set joker on intents;
- a cascade to a local cross-encoder as the clarification step, with a cost-accuracy curve.

**Experiment.**
- **Setup.** CLINC and Banking77 probes at depth 4 and 22.
- **Arms:**
  1. marginal conformal (APS and LAC scores);
  2. clustered conformal;
  3. the open-set joker, with 30 intents held out from training and test containing them.
- **Metrics:** marginal and worst-decile class coverage, set-size distribution, the share of messages in
  each band, and end-to-end accuracy when the middle band goes to laya (laya's accuracy is known from the
  readonce work).
- **Baseline:** CICC's vanilla numbers and an MSP threshold.
- **Expected signal:** clustered conformal lifts worst-decile coverage from about 0.8 to about 0.9 at similar
  set size. The joker recovers most held-out-intent messages at a stated rate.
- **Cost:** about 5 A10 minutes, plus laya on the middle band: about 20-40 A10 minutes if the band is ~20%
  of 5.5k messages × candidate labels.

**Capability:** abstention with coverage guarantees that hold per route, not just on average.

---

### 7. Bias-corrected calibration plus online prior tracking for every branch, not just `oos`

**Status (2026-09-27): implemented, with a substitution.** `explore/lit_oos_bcts.py`, queued as `lit_bcts`
(A10, ~6 min); the smoke run passes.
- **Calibrators:** none, TS, BCTS and vector scaling (bias L2 chosen by 2-fold CV), plus BBSE.
- **Experiments:** Dirichlet label shift and the real CLINC shift.
- **Not implemented: Baby et al.'s FLH tracker.** The stream compares windowed EM, MAP-EM (a Dirichlet prior
  of C pseudo-counts, added after the smoke run showed windowed EM overfitting when the window holds fewer
  messages than classes) and EWMA online EM against static and oracle.

**Papers**
- Alexandari, Kundaje & Shrikumar, "Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at
  Label Shift Adaptation", ICML 2020, https://arxiv.org/abs/1901.06852. EM works well only if the model is
  calibrated in a *class-wise* sense. Temperature plus per-class bias (BCTS) makes EM beat BBSE and RLLS;
  plain temperature scaling often does not.
- Baby, Garg, Yen, Balakrishnan, Lipton & Wang, "Online Label Shift: Optimal Dynamic Regret meets Practical
  Algorithms", NeurIPS 2023, https://arxiv.org/abs/2305.19570. Unsupervised tracking of drifting class
  marginals, by bootstrapping online-regression estimates; +1-3% accuracy over earlier online methods.
- Weak recent item: Hu & Barria, "FMAPLS / online-FMAPLS", Nov 2025, https://arxiv.org/abs/2511.18615.
  Dirichlet-MAP priors, vision only, journal submission.

**Idea to borrow.** Tarski calibrates with one scalar temperature (skeptic finding 1), which cannot fix
class-wise miscalibration. Two changes:
- (a) Fit a temperature plus a per-class bias on validation. Use L2-regularised biases: we have only 13-20
  validation examples per class.
- (b) Run the Saerens EM continuously over a sliding window of traffic for *every* routing branch, or use
  Baby et al.'s tracker. Route popularity drifts constantly (incident spikes, launches), not only the OOS
  share.

**Novelty.** None as a method. Engineering value is high, and it generalises the skeptic's one-off fix.

**Experiment.**
- **Setup.** Banking77 and CLINC 150-way. Test sets are resampled with Dirichlet(α) priors, α ∈ {0.1, 0.3, 1,
  10}.
- **Arms:** no correction; EM + TS (current); EM + BCTS; EM + vector scaling; BBSE.
- **Streaming arm.** Piecewise-constant prior changes every 500 messages; compare windowed EM (windows of
  200 and 1000) with Baby et al.'s tracker.
- **Metrics:** L1 error of the estimated prior, and accuracy / macro-F1 change.
- **Expected signal:** BCTS cuts prior-estimation error by 20-50% relative to TS and adds 1-5 accuracy
  points at α ≤ 0.3. It may overfit at 13 examples per class, which is worth reporting honestly.
- **Cost:** about 3 A10 minutes, post-hoc.

**Capability:** decisions that adapt to shifting route popularity with no labels.

---

### 8. LLM-synthesised near-OOS negatives for user route sets

**Status (2026-09-27): implemented; generation needs the A100 and Hub access.** `explore/lit_oos_synth_neg.py`,
queued as `lit_synth_neg` (A100, ~25 min including the Qwen/Qwen3-8B download).
- **Hub access.** The command sets `HF_HUB_OFFLINE=0`, because the job runner exports `HF_HUB_OFFLINE=1`.
- **Fallback.** If the download or loading fails, the script records the error and falls back to a
  template stub, so the other arms still run. Check `generator` in the JSON before reading any synth
  number.
- **Testing.** The generation code path was exercised offline with a tiny randomly initialised Qwen3. The
  smoke run uses the stub.

**Papers**
- Abbas, Azmat, Horesh & Yurochkin, "Out-of-Distribution Detection using Synthetic Data Generation", COLM
  2025, https://arxiv.org/abs/2502.03323.
  - **Generation.** Llama-3-70B generates near-OOD data (in-domain demonstrations, other classes) and
    far-OOD data (a two-stage seed-then-generate process).
  - **Training.** Either a binary head on frozen features or a (K+1)-way model.
  - **Results.** FPR95 goes from about 92% (MSP) to 10-13% on near-OOD pairs such as SST-2 and ToxiGen, and
    to 0 on far-OOD pairs.
- Cao et al., "Envisioning Outlier Exposure by Large Language Models" (EOE), ICML 2024,
  https://arxiv.org/abs/2406.00806. LLM-imagined outlier *classes* for CLIP. Conceptual precedent only.

**Idea to borrow.** When a user defines routes, a local LLM writes messages that sound like the domain but
fit none of the routes. They become negatives for the `oos` branch, or a (K+1)-th class.
- Learning J uses *other tasks'* messages, which are far-OOD.
- This targets near-OOS, which is where CLINC OOS actually lives.

**Claims.** The paper's pairs are mostly cross-dataset (SST-2 vs other corpora), which is easier than
CLINC-style OOS. Its "near-OOD" gap to real OOS data is acknowledged. Contamination risk: an LLM may have
memorised CLINC.

**Novelty.** Applying synthetic near-OOS to intent routing with a frozen trunk was not found in a quick
search, but it is a natural step. Moderate.

**Experiment.**
- **Setup.** CLINC with the 250 real OOS training messages removed.
- **Generation.** An 8B instruct model (for example Llama-3.1-8B via vLLM on the A10) writes about 2,000
  near-OOS utterances: about 20 per domain per prompt, with 5 in-domain examples and the domain's intent
  names as "things it must NOT be". Deduplicate against the CLINC test set by 5-gram overlap.
- **Arms:**
  1. `oos` probe trained with synthetic negatives;
  2. with Banking77 negatives (learning J);
  3. with the real 250 OOS (reference);
  4. label-free Mahalanobis++ (entry 3);
  5. synthetic + Banking77.
- **Metrics:** AUROC, and OOS F1 after Saerens correction.
- **Expected signal:** synthetic ≥ Banking77 negatives by 0.02-0.05 AUROC, and within about 0.02 of real OOS.
- **Cost:** about 15 A10 minutes for generation plus 3 for probes.

**Capability:** a working "none of the above" for routes the user defined minutes ago.

---

### 9. Adapt once, then add routes as prototypes on the branch's own features (APER / EASE on forked branches)

**Status (2026-09-27): implemented, APER only.** `explore/lit_oos_aper_branch.py`, queued as `lit_aper_branch`
(A10, ~15 min); the smoke run passes.
- **Implemented:** LDA and KLDA on trunk, branch and concatenated features; a SEQ*-style no-replay
  new-rows head, raw and weight-aligned; a joint blocks branch as reference.
- **Not implemented:** EASE's prototype complement.
- **Baseline fix.** Without centring its inputs and fixing its biases, the no-replay head collapsed old
  routes to 0% in the smoke run. That fix is applied.

**Papers**
- Zhou et al., "Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and
  Adaptivity are All You Need" (SimpleCIL / APER), IJCV 2024, https://arxiv.org/abs/2303.07338.
  - SimpleCIL (prototypes on the frozen pretrained model) is already strong.
  - APER adapts the model on the first session only, then builds every later class's prototype from the
    concatenation of frozen and adapted features.
- Zhou et al., "Expandable Subspace Ensemble" (EASE), CVPR 2024, https://arxiv.org/abs/2403.12030. One light
  adapter per task, a subspace per task, and prototypes of old classes synthesised in new subspaces from
  class similarity in the shared space.
- Zheng, Qiu & Ma, "Learn or Recall? Revisiting Incremental Learning with Pre-trained Language Models"
  (SEQ*), ACL 2024 oral, https://arxiv.org/abs/2312.07887. Across 20+ methods on text (including intent
  classification), pretrained LMs forget little. The classifier is the problem, so freeze old classifier
  rows and train only the new ones. This supports learning H.

**Idea to borrow.** A tarski blocks branch *is* an adapted subspace. When a user adds routes after the
branch was trained, do not retrain the branch. Compute the new routes' prototypes or KLDA statistics on
[trunk@k pool ; branch output pool] (APER).
- When a new branch is trained for a new group of routes, EASE's prototype complement answers the question
  of how to compare its classes with the old branch's without retraining it.

**Novelty.** Vision methods; the mapping onto forked-layer branches is natural. Not found for text branches
on a shared frozen trunk. Modest.

**Experiment.**
- **Setup.** CLINC in-scope. Session 1 is 50 intents: train blocks@11+2 on them. Sessions 2-11 add 10
  intents each, with no gradient steps.
- **Arms for the new routes:**
  1. KLDA on trunk@11 (entry 1);
  2. KLDA on branch features;
  3. KLDA on the concatenation (APER);
  4. SEQ*-style: train only the new head rows on the branch.
- **Metrics:** final and average incremental accuracy against the joint blocks branch on all 150.
- **Expected signal:** the concatenation beats trunk-only by 1-2 points on routes in domains seen in session
  1, and roughly ties elsewhere.
- **Cost:** about 8 A10 minutes (one blocks branch plus features).

**Capability:** adding routes to a *blocks* branch (not just probes) without retraining it.

---

### 10. Mine the OOS inbox for new-route proposals

**Status (2026-09-27): implemented.** `explore/lit_oos_discovery.py`, queued as `lit_discovery` (A10,
~4 min); the smoke run passes.
- **Loop:** flag with Mahalanobis++, cluster with HDBSCAN at depths 4 and 22, promote clusters to LDA
  routes.
- **Compared against:** routes from 10 true labels and from all inbox labels, and an all-held-out upper
  bound.

**Papers**
- Song et al., "Continual Generalized Intent Discovery" (CGID / PLRD), EMNLP Findings 2023,
  https://aclanthology.org/2023.findings-emnlp.289/. Discover OOD intent clusters from a stream and add
  them incrementally, with replay and distillation. It fine-tunes the encoder.
- Aida & Formentin, "Uncertainty-Aware Continual Learning for Open-World Intent Discovery Under an
  Evolving Label Space", Sep 2026, https://arxiv.org/abs/2609.17866. Confidence, posterior uncertainty and
  DP-GMM likelihood flag unknowns; density clustering proposes clusters, and only confident ones are
  promoted. It reports near-zero NMI/ARI, so the evidence is weak.

**Idea to borrow.** The product loop rather than the method. Messages flagged OOS (entries 3 and 4)
accumulate. The trunk clusters them. A dense, tight cluster is proposed to the user as "a new route may be
forming", with representative messages. If accepted, it becomes a route through entry 1's statistics,
optionally with a few user-confirmed labels.

**Claims.** The literature is weak; the 2026 paper admits it recovers local structure, not the taxonomy.
Hence the low rank.

**Novelty.** Low as a method. As a harness capability (OOS gate, clustering, one-click route) it is
unexplored but mostly engineering.

**Experiment.**
- **Setup.** CLINC with 15 intents from different domains held out as unknown. The flagged pool mixes those
  intents' test messages with real OOS messages as noise.
- **Clustering.** HDBSCAN on L2-normalised depth-4 states, and on the RFF space.
- **Metrics:**
  - the share of held-out intents recovered as a proposed cluster (purity ≥ 0.8, at least 10 members);
  - false proposals made from real-OOS noise;
  - test accuracy of a route promoted from the cluster's pseudo-labels, against the same route built from
    10 true labels.
- **Expected signal:** 50-70% of held-out intents recovered, 1-3 junk clusters from real OOS, and promoted
  routes within 5-10 points of the 10-label routes.
- **Cost:** about 3 A10 minutes.

**Capability:** telling the user a new route is needed, before they notice.

---

## Implementation status (2026-09-27)

All 10 entries are implemented as `explore/lit_oos_<idea>.py`, sharing helpers in
`explore/lit_oos_common.py`.
- **Smoke runs.** Every script has a CPU `--smoke` mode, and all ten pass. Smoke numbers come from tiny
  stratified subsamples and are not evidence.
- **Queue.** Full runs are in `explore/QUEUE_lit_oos.txt`, in the suggested order: `lit_mahapp` first,
  then `lit_klda`, `lit_openset_shift`, `lit_conformal_select`, `lit_bcts`, `lit_drift`, `lit_cicc`,
  `lit_synth_neg` (A100), `lit_aper_branch`, `lit_discovery`. That is about 120 GPU-box minutes in total.
- **Training steps.** Every training loop runs at least 300 optimiser steps: full-batch probes run 300
  Adam steps, and blocks branches use `tarski.train.train_branch`.
- **Feature cache.** Pooled trunk states are cached per (texts, depth) under `~/.cache/tarski/lit_oos`,
  so jobs on the same box share trunk passes.

## Summary ranking

| # | Entry | Capability | Novelty | Cost (A10 min) |
|---|---|---|---|---|
| 1 | KLDA / RanPAC kernelised statistics routes | add a route in closed form; maybe beat probes | low (KLDA did text CIL) | ~5 |
| 2 | Conformal selection auto-routing (FDR ≤ q), label-shift-weighted | route or escalate with a guarantee | application gap | <5 |
| 3 | Mahalanobis++ / KPCA for label-free OOS | fixes our weakest detector | none (probable fix) | ~2 |
| 4 | Open-set label shift estimation | OOS without negatives; "% fits no route" gauge | not found for intents | ~5 |
| 5 | Label-free harmful-drift alarm + active inference | retrain trigger under a label budget | not found for text routing | ~5 |
| 6 | CICC + clustered conformal + open-set joker | per-route abstention guarantees, clarify band | narrow | 5 (+20-40 with laya) |
| 7 | BCTS + online prior tracking on all branches | adapts to route-popularity shifts | none | ~3 |
| 8 | LLM-synthesised near-OOS negatives | OOS for freshly defined routes | moderate | ~18 |
| 9 | APER / EASE on forked branches | add routes to blocks branches without retraining | modest | ~8 |
| 10 | OOS-inbox route proposals | detect emerging routes | low; weak literature | ~3 |

Suggested order to run:
- 3 then 1: same script family; 3 takes minutes and may change which OOS score the others should use.
- Then 4, 2 and 7: all post-hoc on cached logits, so they can share one script.
- Then 5.

## Looked at and not proposed

- **ADB** (Zhang, Xu & Lin, AAAI 2021, https://github.com/thuiar/Adaptive-Decision-Boundary). Per-class
  spherical boundaries on fine-tuned features. Could be a baseline arm in entry 3.
- **Multi-cluster boundaries on frozen MiniLM** (arXiv 2607.07974, Jul 2026). The abstract has no numbers;
  worth a baseline only if the numbers turn out strong.
- **WATCH** (ICML 2025). Needs online labels, so it is folded into entry 5 as a related item.
- **Suitability filter** (ICML 2025). Covariate-shift accuracy test. It is an alternative to the proxy-risk
  test in entry 5; not separately proposed.
- **HolUE for open-set text** (arXiv 2604.08560, 2026). Gallery-based uncertainty. The setup differs
  (authorship and topic galleries).
- **Drift diagnosis, real vs virtual** (arXiv 2609.27865, 2026). Vision only, and needs labels.
- **FMAPLS** (arXiv 2511.18615). Vision only, incremental over MAPLS.
- **EOE** (ICML 2024). CLIP-specific. Only its concept carries over, into entry 8.
