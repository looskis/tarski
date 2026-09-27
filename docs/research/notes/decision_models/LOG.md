# Decision models: experiment log

## 2026-09-28: Phase A on JevBench public (231 items), DiffusionGemma 26B-A4B 4-bit on the M6

Setup: OpenJev's own canvases (vendored engine), one read = one decoder pass over a 16-token canvas
(0.1 s) after a prefill of 0.3–4 s; 16 seeded reads + the vocabulary-mean-embedding read per item.
Results file: `results/dlm/anatomy_jevbench.jsonl`; scorer `dlm/metrics.py`.

**Reproduction.** Single-read accuracy 190/231 = 82.3% (easy 48/48, original 71/72, hard 71/111);
the leaderboard's public figure for OpenJev (NVFP4, vLLM) is 81.8%.

**Seed variance (the random-token objection).** Negligible on easy/original (mean TV from the
16-read mean 0.001–0.004; 0–2% of slots over 0.05). On the hard tier: mean TV 0.041, **29% of slots
over 0.05, 5% argmax flips** between a single read and the noise-averaged read.

**What re-reads buy.** OpenJev's auto policy (re-read x3 if entropy > 0.1) brings hard-tier TV to
0.015 and flips to 1%; accuracy unchanged (0.631 vs 0.640, noise at n=111); ECE 0.218 → 0.213.
Re-reads fix the variance, not the calibration.

**Calibration.** Hard-tier ECE ≈ 0.21 for every random-token condition; by type over all tiers:
choice 0.10, noul 0.15, score 0.18. Overconfident, as the DLM literature predicts.

**Mean-embedding read (deterministic, untrained).** Hard tier: accuracy 0.658, ECE 0.184, Brier 0.508;
TV 0.049 and 7% flips against the noise-averaged read, i.e. a different read, not a copy of it, and
a slightly better one. Score-type: 0.833 vs 0.722 (n=18). This is the untrained starting point of a
learned query, so the query has room on calibration.

**Verdict on the kill condition** ("negligible seed variance, no order effect, ECE < 0.05"): not met
on the hard tier on two of three counts; order effects are measured next on the five-question
typed-decisions canvases.

| hard tier, n=111 | acc | Brier | ECE | TV→meanK | flips |
|---|---|---|---|---|---|
| single (seed 0) | 0.640 | 0.514 | 0.218 | 0.032 | 0.05 |
| auto4 (OpenJev default) | 0.631 | 0.505 | 0.213 | 0.015 | 0.01 |
| mean of 16 seeds | 0.640 | 0.503 | 0.208 | 0 | 0 |
| mean-embedding slot | 0.658 | 0.508 | 0.184 | 0.049 | 0.07 |

## 2026-09-28: Phase A on typed-decisions test (400 states, 2,000 slots), same setup

Results file: `results/dlm/anatomy_typed_test.jsonl` (16 seeds + mean-embedding read per slot; five
questions per canvas). Harder than JevBench public for the stock read: single-read accuracy 0.58 (choice),
0.79 (noul), 0.62 (score); ECE 0.20–0.33. Seed variance is JevBench-hard-tier sized on choice and
score slots (26–30% of slots over 0.05 TV from the 16-read mean, 5% argmax flips) and small on noul.

**Mean-embedding slot beats every random-token condition on every metric for every type**, including
the 16-read average, while being one deterministic read: choice acc 0.628 vs 0.600 (meanK), Brier
0.623 vs 0.682, ECE 0.259 vs 0.307. It is not a copy of the noise average (TV 0.08–0.09, 7–8% flips
on choice/score). This is the untrained initialisation of the learned query.

| typed-test | choice (n=600) | noul (n=600) | score (n=800) |
|---|---|---|---|
| single: acc / Brier / ECE | 0.582 / 0.723 / 0.328 | 0.785 / 0.403 / 0.198 | 0.624 / 0.620 / 0.273 |
| auto4 | 0.585 / 0.704 / 0.310 | 0.788 / 0.391 / 0.189 | 0.637 / 0.588 / 0.245 |
| mean of 16 seeds | 0.600 / 0.682 / 0.307 | 0.790 / 0.380 / 0.182 | 0.632 / 0.590 / 0.245 |
| mean-embedding slot | **0.628 / 0.623 / 0.259** | **0.797 / 0.361 / 0.164** | **0.646 / 0.574 / 0.232** |

## 2026-09-28: Phase B.1, label-free learned queries (target = 16-seed mean), typed-decisions

Setup: one learned slot embedding per question type (3 x 2,816 params), initialised at the
vocabulary-mean embedding, frozen model, Adam at 0.01 x label-token RMS, 3 epochs over the 1,200
train states (20 min/epoch on the M6), target = each slot's own 16-read mean distribution. Eval file
`results/dlm/anatomy_typed_test.meanK_type.jsonl`, condition `query:meanK_type`.

**The objective works: the target is wrong.** The query reproduces the 16-read mean in one
deterministic read (TV to meanK 0.013–0.035 by type, under the single-read seed noise of
0.017–0.051; argmax flips 1–3%). But it lands at exactly the 16-read mean's quality
(overall acc 0.673 / Brier 0.555 / ECE 0.244 vs meanK 0.670 / 0.555 / 0.245), which is *worse*
than its own untrained initialisation, the mean-embedding read (0.686 / 0.525 / 0.219). Training
loss 0.235 → 0.230 over three epochs; test numbers flat from epoch 1.

Reading: the noise average is not the best read the model can give, so self-distilling to it
throws away the calibration advantage of the deterministic slot. Whatever makes the
mean-embedding read better than the noise average (it is not a copy of it: TV 0.08) is lost by
pulling it back toward that average. A label-free target has to be a *better* teacher than 16
seeds, not a cheaper copy of it.

| typed-test | choice | noul | score |
|---|---|---|---|
| mean of 16 seeds (acc / Brier / ECE) | 0.600 / 0.682 / 0.307 | 0.790 / 0.380 / 0.182 | 0.632 / 0.590 / 0.245 |
| mean-embedding slot (untrained) | **0.628 / 0.623 / 0.259** | 0.797 / **0.361 / 0.164** | **0.646 / 0.574 / 0.232** |
| query, label-free to meanK | 0.603 / 0.671 / 0.302 | 0.797 / 0.383 / 0.181 | 0.632 / 0.597 / 0.254 |

Next: the gold-target control (same setup, target = gold distribution) tells whether the query
can move *past* the mean-embedding read at all with 1,200 labelled states.

## 2026-09-28: Phase B.2, gold-target queries (control), and what is calibration vs ranking

Same setup as B.1 with target = the gold distribution (typed-decisions gives soft gold). Eval file
`results/dlm/anatomy_typed_test.gold_type.jsonl`, condition `query:gold_type`. Three epochs, 20 min each;
test numbers move at epoch 1 and are flat after (0.719/0.414/0.112 → 0.711/0.398/0.102 overall).

**Three learned slot embeddings (8,448 parameters, frozen 26B model) beat sixteen reads on every
metric.** Overall acc 0.711 vs 0.670 (16-seed mean) and 0.686 (mean-embedding slot); Brier 0.398 vs
0.555 / 0.525; ECE 0.102 vs 0.245 / 0.219. The learned read is a different read, not a sharpened one:
TV 0.27–0.44 from the 16-seed mean, 19–39% argmax flips by type.

**How much of that is just calibration?** `dlm/calibrate.py` fits temperature scaling (and
temperature + label-marginal mixing) on the train anatomy and scores test; temperature cannot change
an argmax. On the mean-embedding read it recovers most of the Brier/ECE gap without labels for the
model: Brier 0.525 → ~0.44, ECE → 0.03–0.09 (T ≈ 2–5: the raw read is overconfident, as the DLM
literature says). Like for like, with both reads temperature-scaled by 2-fold cross-fit on test:

| typed-test, acc / Brier / ECE | choice | noul | score |
|---|---|---|---|
| mean-embedding slot, raw | 0.628 / 0.623 / 0.259 | 0.797 / 0.361 / 0.164 | 0.646 / 0.574 / 0.232 |
| mean-embedding slot + temperature | 0.628 / 0.565 / 0.098 | 0.797 / 0.300 / 0.044 | 0.646 / 0.488 / 0.039 |
| gold query, raw | 0.682 / 0.444 / 0.115 | 0.793 / 0.282 / 0.052 | 0.670 / 0.451 / 0.130 |
| gold query + temperature | **0.682 / 0.418 / 0.042** | 0.793 / **0.277 / 0.025** | **0.670 / 0.415 / 0.049** |
| train label marginal only | 0.418 / 0.660 / 0.081 | 0.492 / 0.500 / 0.010 | 0.338 / 0.731 / 0.018 |

So the query's gain survives calibration: +5.4 pts accuracy on choice and +2.4 on score (none on
noul, where the read was already good), and Brier 0.418 vs 0.565 / 0.277 vs 0.300 / 0.415 vs 0.488
after both are calibrated. Temperature + marginal mixing on the mean-embedding read (the
"is it just a prior?" control) reaches only 0.638 on choice, so the query is not a learned label
prior either. The fitted temperatures tell the two stories: the raw read needs T ≈ 2–5 (overconfident),
the query needs T ≈ 0.5–0.7 (underconfident, because it was trained on soft gold).

Verdict for technique 1 so far: (i) the random slot is the wrong input, and the vocabulary-mean
embedding is a free deterministic improvement; (ii) self-distilling to the noise average is a dead end;
(iii) with 1,200 labelled states, one embedding per question type is a better read than any number of
random-token reads, and the gain is ranking, not calibration. Whether (iii) transfers to unseen
questions is the JevBench zero-shot run (next).

## 2026-09-28: Phase B.3, zero-shot transfer of the queries to JevBench public (231 items)

File `results/dlm/anatomy_jevbench.queries.jsonl` (conditions `query:gold_type`, `query:meanK_type`).
**The typed-decisions gain does not transfer.** Hard tier (n=111): gold query 0.649 / 0.515 / 0.198 vs
mean-embedding slot 0.658 / 0.508 / 0.184 vs single read 0.640 / 0.514 / 0.218; overall 0.827 / 0.250
/ 0.095 vs 0.831 / 0.250 / 0.090. The query barely acts on JevBench canvases at all (TV to the 16-seed
mean 0.03–0.05, against 0.27–0.44 on typed-test), so what it learned is specific to typed-decisions'
question phrasing and label structure, not a per-type way of reading. The meanK query is inert
everywhere. On JevBench the best read remains the untrained mean-embedding slot.

## 2026-09-28: Phase B.4, cross-slot interference and slot isolation (200 typed-test states, 1,000 slots)

File `results/dlm/interference_typed_test.jsonl`; `dlm/interference.py --report`. All reads seed 0.

| condition | TV→joint | flips | acc | Brier | ECE |
|---|---|---|---|---|---|
| joint (OpenJev default: five questions, given order) | 0 | 0 | 0.619 | 0.626 | 0.277 |
| joint, seed 1 (seed effect, for scale) | 0.071 | 0.07 | 0.641 | 0.594 | 0.242 |
| isolate (slots masked from other slots) | 0.083 | 0.08 | 0.617 | 0.614 | 0.266 |
| mean-embedding slot | 0.116 | 0.11 | 0.674 | 0.535 | 0.207 |
| alone (one question per canvas, 5 prefills) | 0.294 | 0.29 | 0.653 | 0.532 | 0.208 |
| **reversed question order** | 0.260 | 0.27 | **0.694** | **0.489** | **0.202** |
| invisible (no row sees any slot) | 0.418 | 0.46 | 0.388 | 0.862 | 0.326 |
| invisible + mean-embedding slot | 0.444 | 0.49 | 0.394 | 0.835 | 0.284 |

Four findings, in order of importance:

1. **Question order is the largest effect measured, four times the seed effect.** Reversing the
   order of the same five questions moves 27% of argmaxes and adds 7.5 points of accuracy overall;
   on choice questions +14 points (0.562 → 0.703), better than reading the question alone (0.610).
   OpenJev's given order is the worst of the three orders measured. A bidirectional decoder with a
   1,024-token sliding window has no business being this order-sensitive, and nothing in OpenJev
   controls for it.
2. **Joint canvases hurt.** One question per canvas beats the five-question canvas by 3.4 points and
   0.09 Brier, at five times the prefill cost. The other questions are context that helps some
   questions and hurts others depending on where they sit.
3. **Slot-to-slot attention is not the channel.** Masking slots from each other changes almost
   nothing (TV 0.08, same accuracy): the contamination travels through the question text and the
   template rows, not through the other slots' random tokens. The slot-isolation mask as a *fix* is
   falsified; it stays as a probe.
4. **Hiding the slots from the template rows breaks the read** (0.39 accuracy): the model's
   prediction at a slot is carried by the surrounding rows attending to that slot in earlier layers
   and the slot reading them back. A useful mechanistic fact about how DiffusionGemma denoises a
   position, and a warning that "a read that depends on nothing else" is not available by masking.

Next: cyclic rotations of the five questions (every question at every canvas position) and
prompt-order vs canvas-order hybrids, to locate the order effect and turn it into a policy.

## 2026-09-28: Phase B.5, expert census (English), JevBench original+easy 120 states and typed-test 100 states

Files `results/dlm/census_{jevbench_en,typed_test}.json`. Gate mass per expert per layer, split by
token role; experts needed for 90% / 99% of mass; the same 30 layers serve the prefill and the canvas.

| per layer (JevBench) | 90% mass | 99% mass |
|---|---|---|
| prompt tokens (state, encoder prefill) | 28–72 experts | 64–104 |
| template rows (question text in the canvas) | 8–34 | 13–67 |
| **slot rows (the answer positions)** | **7–19** | **9–31** |
| union of all three roles | 51 (mean) | 91 (mean) |

- The decision read itself is narrow: the answer positions route through ~10–20 of 128 experts per
  layer, and those sets overlap little with the experts that read the state (Jaccard 0.08–0.27).
- Reading the state is broad: 99% of prompt mass needs 64–104 experts per layer, and the prompt
  expert sets are stable across datasets (Jaccard 0.7–0.9 JevBench vs typed-decisions).
- **32% of (layer, expert) pairs carry under 0.1% of the prompt mass on both datasets** (1,240 of
  3,840). On English decision canvases roughly a third of the routed-expert parameters are dead
  weight, which is the user's "wasted parameters" hypothesis at its first quantitative test. A
  carve at 99% union mass keeps ~91/128 experts (≈30% smaller); at 90% union ≈51/128 (2.5x smaller)
  with an accuracy cost still to be measured.
- Slot expert sets are less stable across datasets (Jaccard 0.3–0.6), but they are small, so the
  overlap statistic is noisy; the top experts by mass are shared.
