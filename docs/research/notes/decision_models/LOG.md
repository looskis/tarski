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

## 2026-09-28: Phase B.6, question order (all 400 typed-test states, 2,000 slots, four families)

File `results/dlm/order_typed_test.jsonl`; `dlm/order.py --report`. Five cyclic rotations of the given
order (rot0 = OpenJev's order), the reversed order, and two hybrids that reorder only the system
prompt or only the canvas. Random slot at seed 0 and the mean-embedding slot for every condition.

| accuracy, random slot | rot0 (given) | rot1 | rot2 | rot3 | rot4 | reversed |
|---|---|---|---|---|---|---|
| agent_trace_observability (500) | **0.570** | 0.558 | 0.620 | 0.676 | 0.556 | 0.632 |
| customer_service (500) | **0.668** | 0.724 | 0.748 | 0.762 | 0.736 | 0.756 |
| invoice_processing (500) | 0.708 | **0.706** | 0.740 | 0.744 | 0.740 | 0.752 |
| security_incidents (500) | 0.692 | 0.694 | 0.716 | **0.688** | 0.710 | 0.716 |
| all (2,000) | 0.659 | 0.670 | 0.706 | 0.718 | 0.685 | 0.714 |

- **The given order is the worst or second-worst of six in every family.** Choosing one order on
  half the states and scoring the other half gives 0.712 (global choice) / 0.722 (per family) against
  0.659: +5 to +6 points on 2,000 slots at identical inference cost; +10 on the first two families,
  +2 to +4 on the other two. With the mean-embedding slot: 0.686 → 0.720 (global held-out choice).
- **Order beats order-averaging.** Averaging the five rotations' distributions fixes calibration
  (Brier 0.626 → 0.480, ECE 0.277 → 0.179 on the first 200 states) but not accuracy (0.667): a
  selection problem, not a noise problem. Best single order + mean slot: 0.707 / 0.438 / 0.141 on
  those states, against OpenJev's 0.619 / 0.626 / 0.277.
- **No label-free selector works.** Per question, taking the order with the highest confidence,
  lowest entropy, or the majority vote across orders all score 0.66–0.69, below the fixed order
  (0.72); the per-item oracle is 0.855, so the headroom is real but the model's confidence cannot
  find it (consistent with the DLM overconfidence results).
- **The prompt-only / canvas-only reversals are a labelling artefact, not a finding.** OpenJev
  assigns question ids by position (q1..q5 in the order given), so reversing only the prompt makes
  "Question q1" describe the last question while the canvas's "q1:" slot is scored as the first;
  the collapse (0.39–0.54) is the model answering the prompt's q1 into the canvas's q1. The hybrid
  cannot split prompt order from canvas order in this format. Position alone explains part of the
  effect (choice slots: pos1 0.555 … pos4 0.688) but the arrangement matters beyond position.
- Caveats: one model, one quantisation; the given orders differ per family (types
  choice-noul-choice-score-score, choice-choice-score-noul-score, score-choice-noul-noul-score,
  noul-choice-score-noul-score), so which rotation wins is not a type pattern; the honest policy is
  "select on held-out labelled states", not "use rot3".

Next: attention-leakage per order (does a slot's attention to other questions' rows predict the bad
orders? that would explain the effect and might give the label-free selector), layer-localised
visibility, and order selection with train/test split and a labelled-budget curve.

## 2026-09-28: Phase C.1, simulated carve at 99% mass (experts chosen on the other dataset)

`dlm/carve.py`: every router restricted to the experts that carry 99% of the census gate mass
(union over token roles), chosen on JevBench's census and evaluated on typed-test, and vice versa.
Disallowed experts get -inf gate scores in prefill and canvas alike, so the forward pass is what a
physically carved model computes.

| keep 0.99 | experts/layer | condition | acc | Brier | ECE | TV→full | flips |
|---|---|---|---|---|---|---|---|
| typed-test 200 (from JevBench census) | 91 (71%) | full / carved, random slot | 0.619 / 0.632 | 0.626 / 0.616 | 0.277 / 0.260 | 0.10 | 0.11 |
| | | full / carved, mean slot | 0.674 / 0.663 | 0.535 / 0.549 | 0.207 / 0.209 | 0.10 | 0.09 |
| JevBench hard 111 (from typed census) | 87 (68%) | full / carved, random slot | 0.640 / 0.631 | 0.514 / 0.566 | 0.218 / 0.222 | 0.11 | 0.14 |
| | | full / carved, mean slot | 0.658 / 0.622 | 0.508 / 0.544 | 0.184 / 0.225 | 0.11 | 0.15 |

Removing 29–32% of the routed experts changes one read in ten (TV 0.10, 10–15% flips) but costs
nothing measurable on typed-test and a little on JevBench hard (−1 to −4 points, Brier +0.04–0.05;
n=111, SE ≈ 4.5 points). The 95% and 90% levels are running; that curve decides the carve.

## 2026-09-28: torch/CUDA port and parity (H100 PCIe, bf16, transformers 5.17)

`dlm/reads_torch.py` mirrors `dlm/reads.py` (embedding-path slots, per-layer masks, router mask,
routing and attention recorders) on `google/diffusiongemma-26B-A4B-it`; `dlm/parity_torch.py`
compares it with the stored MLX 4-bit reads. Easy + original JevBench (30 items): TV 0.001, argmax
agreement 100%. Hard (20 items): TV 0.09–0.10, argmax agreement 90–95%. The pipeline is identical;
**4-bit quantisation alone moves 5–10% of hard-item argmaxes**, which is itself a caveat for every
4-bit number in this log (the H100 queue replicates the order experiment in bf16). Speed: prefill
0.44 s, read 0.13 s unbatched (the M6: 0.3–4 s and 0.1 s).

## 2026-09-28: Phase B.7, attention leakage per question order (H100, bf16, 200 typed-test states)

`dlm/leakage.py`: for every slot under each order, the slot row's attention mass per layer on its own
question's rows, other questions' rows, other slots, the encoder prefix and itself (heads averaged).

**The order effect replicates in bf16** (random slot: rot0 0.629, rot3 0.703, rev 0.682; mean slot:
0.664 / 0.682 / 0.687), so it is not a 4-bit artefact. **Attention mass does not explain it.** The
shares are flat across orders (own 0.17–0.18, other questions 0.15–0.16, prefix 0.47–0.50) and
uncorrelated with correctness (r = 0.01). Selecting the order with the least leakage to other
questions, or the most attention to its own question, scores 0.65–0.68, below the fixed order
(0.70). Whatever the order changes, it is not how much a slot looks at other questions; the prefix,
which carries half the mass, is itself re-encoded under each order because the system prompt lists
the questions, so the encoder is the next suspect (the layer sweep and logit lens by order test
whether the divergence between orders is present from the first layers).

## 2026-09-28: Phase B.8, multi-step reads (H100, bf16, 200 typed-test states, 1,000 slots)

`dlm/steps.py`: the single denoising pass repeated up to four times, feeding each step's logits back
as the self-conditioning signal (`sc`), optionally also writing each step's argmax token into the
slots as the sampler would (`token`); from OpenJev's random slot and from the vocabulary-mean slot.

| start / update | step 1 (stock) | step 2 | step 3 | step 4 |
|---|---|---|---|---|
| random / self-conditioning only, acc / Brier / ECE | 0.638 / 0.605 / 0.264 | 0.656 / 0.610 / 0.287 | 0.659 / 0.623 / 0.295 | 0.663 / 0.628 / 0.299 |
| random / token write-back | 0.638 / 0.605 / 0.264 | 0.608 / 0.723 / 0.351 | 0.609 / 0.748 / 0.367 | 0.608 / 0.755 / 0.372 |
| mean / self-conditioning only | 0.673 / 0.545 / 0.214 | 0.672 / 0.574 / 0.261 | 0.676 / 0.593 / 0.283 | 0.674 / 0.612 / 0.298 |
| mean / token write-back | 0.673 / 0.545 / 0.214 | 0.677 / 0.597 / 0.284 | 0.677 / 0.629 / 0.311 | 0.677 / 0.638 / 0.316 |

**One step is the right number of steps for a classification read.** Every extra step raises
confidence (ECE +0.03–0.10, Brier +0.03–0.15) without improving the ranking: accuracy is flat from
the mean slot and, with token write-back from a random start, drops three points because the model
commits to its first-step guess. Self-conditioning from a random start recovers +2.5 points, which is
the noise of the random token being averaged out, and the mean slot already has that. OpenJev's
single-step read is not leaving accuracy on the table; iterating is how a sampler sharpens a
distribution, which is the opposite of what a calibrated read needs.

## 2026-09-28: Phase C.2, the carve curve (95% and 90% mass)

Same protocol as C.1 (experts chosen on the other dataset's census, union over token roles).

| experts kept / layer | typed-test 200: random slot acc / Brier | mean slot acc / Brier | JevBench hard 111: random | mean |
|---|---|---|---|---|
| 128 (full) | 0.619 / 0.626 | 0.674 / 0.535 | 0.640 / 0.514 | 0.658 / 0.508 |
| 91 / 87 (99%) | 0.632 / 0.616 | 0.663 / 0.549 | 0.631 / 0.566 | 0.622 / 0.544 |
| 65 / 66 (95%) | 0.638 / 0.577 | 0.643 / 0.566 | 0.577 / 0.604 | 0.595 / 0.574 |
| 51 / 54 (90%) | 0.608 / 0.644 | 0.609 / 0.593 | 0.595 / 0.597 | 0.586 / 0.577 |

- **A third of the experts is close to free** (99%: −1 point on typed-test, −1 to −4 on JevBench
  hard with SE ≈ 4.5). **Half costs 3–6 points**, and 60% costs 6–7. The knee sits between 99% and
  95% of the mass for the better (mean-slot) read; the stock random-slot read hides the loss on
  typed-test behind its own noise but not on JevBench hard.
- The experts removed at 99% are the ones the census called idle (under 0.1% of mass); the loss at
  95% shows the long tail of rarely-used experts still does work on hard items. "Wasted parameters"
  is therefore about a third of the routed experts, not the two-thirds a 4B-active budget would
  suggest, unless the carved model is also retrained (not tested).
- Every carve changes 10–25% of individual reads (TV 0.10–0.22) even where accuracy holds: the
  carved model is a different reader of similar quality, which matters for any claim of exact
  reproduction.

## 2026-09-28: the read policy, assembled (400 typed-test states, 2,000 slots, 4-bit laptop reads)

What a stock OpenJev user gets from the three findings that hold, with no training: the
vocabulary-mean slot, one question order chosen on held-out labelled states, and one temperature
per question type (2-fold cross-fit here). Same model, same single pass, same inference cost.

| policy | acc | Brier | ECE |
|---|---|---|---|
| OpenJev default: given order, random slot, one read | 0.659 | 0.586 | 0.267 |
| + vocabulary-mean slot | 0.686 | 0.525 | 0.219 |
| + selected order | 0.712 | 0.454 | 0.177 |
| + temperature per type | **0.712** | **0.402** | **0.025** |
| (temperature alone, on the default read) | 0.659 | 0.453 | 0.038 |
| (mean slot + temperature, no order selection) | 0.686 | 0.437 | 0.028 |

+5 points of accuracy, Brier 0.586 → 0.402, ECE 0.267 → 0.025. Temperature does the calibration,
the slot and the order do the ranking; neither substitutes for the other.

## 2026-09-28: Phase B.9, where the read is assembled: visibility sweep and logit lens (H100, bf16, 200 states)

`dlm/layers.py`. Visibility sweep: the "invisible" mask (no canvas row sees any slot but itself)
applied only in layers below a cut (`inv_below_c`) or only from a cut upward (`inv_from_c`), stock
masks elsewhere; random slot. Logit lens: final norm + lm_head at the slot row after every layer.

| random slot, acc | given order (rot0) | reversed order |
|---|---|---|
| stock masks | 0.638 | 0.679 |
| slots invisible in all layers | 0.359 | 0.605 |
| invisible in layers 0–2 / 0–5 / 0–9 | 0.641 / 0.636 / 0.648 | 0.671 / 0.679 / 0.671 |
| invisible in layers 0–14 / 0–19 / 0–24 | 0.586 / 0.383 / 0.350 | 0.607 / 0.567 / 0.589 |
| invisible from layer 3 / 6 / 10 onward | 0.628 / 0.642 / 0.654 | 0.664 / 0.669 / 0.682 |
| invisible from layer 15 / 20 / 25 onward | 0.649 / 0.633 / 0.637 | 0.682 / 0.682 / 0.676 |

- **The neighbour loop must close early, and briefly.** Letting the template rows see the slots
  only in layers 0–2 is enough (0.628 vs 0.638); hiding the slots for the first ten layers is also
  free (0.648) provided they are visible afterwards; hiding them through layer 19 collapses the read
  (0.383). So the rows around a slot absorb "there is an answer position here" within a few layers of
  first seeing it, at any point before roughly layer 15, and later visibility adds nothing.
- **The answer becomes readable at layer 22–23 of 30**, in both orders and for both slot inputs:
  logit-lens accuracy sits at chance (0.23–0.38) through layer 21, jumps at 22 (0.46–0.50) and
  settles at 23 (0.58–0.65), then refines slightly to 29. Median settling layer 22–23. The loop (by
  ~15) precedes the readout (22–23) by a clear margin.
- **The reversed order depends less on the loop.** With slots fully invisible the given order falls
  28 points and the reversed order 7; with visibility only from layer 20 the given order collapses and
  the reversed order keeps 0.567. Whatever the better order does, it makes the slot's read less
  dependent on its neighbours having seen it, which is the first mechanistic difference between
  orders we have measured. The lens does not separate the orders (both at chance until 22), so the
  divergence has to be measured on the hidden states directly (next: per-layer cosine distance of
  the slot's residual stream and of the encoded state between orders).

## 2026-09-28: Phase B.10, does the learned query stack with the selected order? (400 typed-test states)

`dlm/order_query.py`, 4-bit laptop reads. Gold query (trained under the given order rot0) and the
mean slot, under rot0 and the selected order rot3.

| | given order (rot0) | selected order (rot3) |
|---|---|---|
| random slot | 0.659 / 0.586 / 0.267 | 0.718 / 0.473 / 0.205 |
| mean slot | 0.686 / 0.525 / 0.219 | **0.712** / 0.454 / 0.177 |
| gold query | **0.711** / 0.398 / 0.102 | 0.689 / 0.437 / 0.077 |

No. The query lifts the order it was trained under to the level the selected order reaches with the
untrained mean slot, and under the selected order it *loses* 2.3 points of accuracy (calibration
stays better, as it was trained on soft targets). The query is a patch for one canvas layout: it
learned to compensate for the given order's weakness, which is also why it did not transfer to
JevBench. Order selection is the general lever; the query is redundant with it.

## 2026-09-28: Phase D.1, state-first prompting (200 typed-test states, 1,000 slots, 4-bit laptop)

`dlm/state_first.py`. The encoder is causal (`use_bidirectional_attention = "vision"`: text is
causal), so in OpenJev's format the state is encoded *after* the question list and its encoding
depends on the questions and their order. State-first puts the same question block after the state
in the user turn (system turn keeps only the generic instruction and the format line); the canvas
is unchanged. Control `user_first`: the question block before the state, but in the user turn.

| random slot, acc / Brier / ECE | given order | rot3 | reversed | spread | TV between orders |
|---|---|---|---|---|---|
| stock (OpenJev) | 0.619 / 0.626 / 0.277 | 0.719 / 0.454 / 0.180 | 0.694 / 0.489 / 0.202 | 0.100 | 0.239 |
| **state-first** | **0.695 / 0.502 / 0.209** | 0.704 / 0.492 / 0.213 | 0.684 / 0.551 / 0.253 | **0.020** | 0.202 |
| user-first (control) | 0.623 / 0.597 / 0.254 | 0.724 / 0.428 / 0.156 | 0.675 / 0.510 / 0.217 | 0.101 | 0.238 |

| mean slot | given order | rot3 | reversed | spread |
|---|---|---|---|---|
| stock | 0.674 / 0.535 / 0.207 | 0.707 / 0.438 / 0.141 | 0.701 / 0.450 / 0.162 | 0.033 |
| state-first | 0.692 / 0.479 / 0.176 | 0.675 / 0.500 / 0.211 | 0.678 / 0.494 / 0.184 | 0.017 |
| user-first | 0.650 / 0.517 / 0.193 | 0.701 / 0.416 / 0.125 | 0.681 / 0.456 / 0.159 | 0.051 |

- **The order effect lives mostly in the encoder.** With the state encoded before it sees any
  question, the spread across orders falls from 10 points to 2 (random slot) and the given order
  jumps from 0.619 to 0.695, within 2.4 points of the best stock order. The control shows it is the
  state's position relative to the questions that matters, not which chat turn holds them: questions
  before the state in the user turn behaves exactly like stock (spread 0.101).
- **What it buys and what it does not.** State-first is order-robust without any labels or
  selection, and the state's prefix is identical for every question set and order, so one cached
  encoding serves any schema. It removes the downside of a bad order; it does not add upside: the
  best stock order still edges it (0.719 vs 0.704, random slot; 0.707 vs 0.692, mean slot), so a
  user with labels can still select an order on top. The reads still move with order (TV 0.20):
  the question block and canvas remain order-dependent, but the answer no longer does much.
- This is the mechanism-derived technique: a prompt-format change that follows from the causal
  encoder and the sweep's finding that the read is assembled from the encoded prefix. To replicate in
  bf16 on all 400 states (H100, queued).

## 2026-09-28: Phase B.11, order selection as a procedure (choose on train, score on test)

`dlm/order.py --split train` on the H100 (bf16, 1,200 typed-train states, 6 orders); scored on the
400 typed-test states read on the laptop (4-bit), so selection and evaluation share neither states
nor precision.

| random slot | given order | chosen per family on train | oracle on test |
|---|---|---|---|
| agent_trace_observability | 0.570 | 0.676 | 0.676 |
| customer_service | 0.668 | 0.756 | 0.762 |
| invoice_processing | 0.708 | 0.752 | 0.752 |
| security_incidents | 0.692 | 0.688 | 0.716 |
| all 2,000 slots | 0.659 | **0.718** | 0.727 |

Mean slot: 0.686 → **0.726** (= oracle). A single global order chosen on train gives 0.718 / 0.720.
**Labelled budget:** choosing per family on N random train states (20 draws), test accuracy with the
random slot: N=5 0.699, N=10 0.710, N=20 0.714, N=50 0.719, N=100 0.721; mean slot: N=20 0.714,
N=100 0.723, N=200 0.726. Twenty labelled states per schema recover most of the gain; a hundred
saturate it. The procedure is: read a small labelled sample under each candidate order, keep the
best, and use it for everything after; no model change, no inference cost.

## 2026-09-28: bf16 replications on the H100 (queue complete)

- **Order effect, all 400 typed-test states in bf16** (`results/dlm/h100/order_typed_test_bf16.jsonl`):
  random slot rot0 0.665, rot1 0.669, rot2 0.698, rot3 0.716, rot4 0.682, rev 0.702; mean slot 0.686,
  0.681, 0.693, 0.706, 0.700, 0.711. Same shape and size as at 4-bit (0.659 → 0.718 / 0.686 → 0.720):
  the given order is worst or second-worst, the gap is 5 points on 2,000 slots.
- **Multi-step reads on JevBench hard (111)**: four steps add 1–2 points (noise at n=111) and cost
  0.09–0.18 Brier and 0.09–0.12 ECE, as on typed-test. One step it is.
- **Parity on 60 hard items**: TV 0.11, argmax agreement 88–90%, accuracy 4-bit 0.53–0.55 vs bf16
  0.48–0.52 (SE ≈ 6.5 points). The 4-bit conversion is not systematically worse on hard items; it is
  a different reader of similar quality, which is the same lesson as the carve.

Addendum (state-first × order selection, 200 states, 4-bit): with state-first there is no order
effect left to select on. Choosing the order on half the states and scoring the other half gives
0.690 (random slot) / 0.674 (mean) against the given order's 0.695 / 0.692: selection picks noise.
The two are alternatives, not layers: state-first is label-free and reaches 0.695; stock + selection
needs ~20–100 labelled states and reaches 0.719. Labels still buy about two points.

## 2026-09-28: Phase D.2, where the orders diverge (H100, bf16, 200 states) and state-first in bf16 (400 states)

`dlm/divergence.py`: per-layer cosine similarity between two orders of (a) the encoder's output on the
last 64 prompt tokens (the state's end: identical text under every order) and (b) each slot's decoder
residual stream, matched by question; the seed comparison (same order, seed 0 vs 1) is the floor.

- **The encoded state differs between orders, and the difference grows with depth.** Encoder cosine
  between rot0 and the reversed order: 1.000 at layers 0–3, 0.99 at 8, 0.977 at 15, 0.92 at 18, 0.88 at
  26–28, 0.987 after the final layer. Under the seed comparison it is 1.000 at every layer (the encoder
  never sees the seed). Same curve for rot0 vs rot3. This is the direct evidence for the encoder origin
  of the order effect: the causal encoder reads the state after the question list, and by the upper
  layers a quarter of the state representation's direction depends on which order that list came in.
- Decoder slot streams diverge between orders (cos 0.70–0.95 through layer 21) and converge from
  layer 22 (0.92–0.98), the readout layer from the lens; the seed pair diverges as much in the early
  layers (the slot token differs) and converges to 0.99. Per-slot divergence does not separate flipped
  from unflipped answers (cos differences ≤ 0.04), so the effect is distributed, not a few rows.

**State-first in bf16 on all 400 test states** (`results/dlm/h100/state_first_typed_test_bf16.jsonl`):

| random slot, acc / Brier / ECE | given order | rot3 | reversed | spread |
|---|---|---|---|---|
| stock | 0.665 / 0.583 / 0.268 | 0.716 / 0.489 / 0.216 | 0.702 / 0.503 / 0.227 | 0.050 |
| **state-first** | **0.714 / 0.485 / 0.211** | **0.728 / 0.465 / 0.204** | 0.716 / 0.504 / 0.230 | **0.014** |
| user-first (control) | 0.658 / 0.585 / 0.266 | 0.721 / 0.465 / 0.192 | 0.714 / 0.487 / 0.214 | 0.063 |

Mean slot: stock 0.686 / 0.706 / 0.711 (spread 0.025); state-first 0.705 / 0.718 / 0.712 (spread
0.013); control 0.674 / 0.717 / 0.713. In full precision and on the full test set, state-first is not
only order-robust (spread 1.4 points vs 5.0) but at least as good as the best stock order (0.728 vs
0.716 random slot; 0.718 vs 0.711 mean slot). The label-free fix matches the labelled procedure.

Addendum (state-first on JevBench hard, bf16, 111 single-question items): stock 0.631 / 0.542 / 0.233,
state-first 0.649 / 0.561 / 0.240, questions-first-in-user-turn 0.613 / 0.576 / 0.245 (random slot;
mean slot: 0.631 / 0.649 / 0.613). Order is moot with one question; the format change is neutral on
accuracy (+1.8, within noise) and slightly worse on Brier (+0.02). Safe to adopt for single-question
reads. Paper draft started: https://claude.ai/code/artifact/d9341efe-c1af-4d39-8fc5-56b08905461e

## 2026-09-28: second model, LLaDA-8B-Instruct (masked diffusion, single bidirectional stack), order (400 states)

`dlm/reads_llada.py` (native slot = the model's mask token; "mean" = vocabulary-mean embedding via
inputs_embeds), `results/dlm/h100/order_typed_test_llada.jsonl`, 0.7 s/state on the H100.

| LLaDA-8B, accuracy | rot0 (given) | rot1 | rot2 | rot3 | rot4 | reversed | spread | held-out chosen |
|---|---|---|---|---|---|---|---|---|
| native (mask) slot | **0.497** | 0.562 | 0.573 | 0.564 | 0.565 | 0.569 | 0.075 | 0.558 |
| mean slot | 0.489 | 0.521 | 0.534 | 0.476 | 0.508 | 0.522 | 0.058 | 0.518 |

- **The order effect is general.** A weaker reader (0.50–0.57 vs 0.66–0.72), but the same shape: the
  given order is the worst of six overall and in every family with the native slot, the spread is
  7.5 points, and held-out selection adds 6. Score questions suffer most under the given order (0.370
  → best 0.557). So the effect is not specific to DiffusionGemma's causal encoder; it is a property
  of reading a canvas of questions in one pass. Whether the *state-first* fix also transfers to a
  model without a causal encoder is the running test (a bidirectional stack encodes the state in the
  context of the questions wherever they sit).
- The mean-embedding slot is worse than the native mask in LLaDA (−0.8 to −5 points): the trick is a
  uniform-state phenomenon (a random token is a specific wrong word; a mask is the trained "unknown"
  input and already carries no evidence).

## 2026-09-28: second model, LLaDA-8B, state-first (400 states)

`results/dlm/h100/state_first_typed_test_llada.jsonl`, native (mask) slot, acc / Brier / ECE:

| LLaDA-8B | given order | rot3 | reversed | spread |
|---|---|---|---|---|
| stock | 0.497 / 0.603 / 0.141 | 0.564 / 0.559 / 0.078 | 0.569 / 0.552 / 0.089 | 0.071 |
| state-first | 0.533 / 0.608 / 0.146 | 0.565 / 0.580 / 0.109 | 0.593 / 0.537 / 0.082 | 0.059 |
| questions before state, user turn | 0.500 / 0.606 / 0.139 | 0.572 / 0.566 / 0.092 | 0.551 / 0.557 / 0.097 | 0.072 |

**State-first helps only partially on a model without a causal encoder**: the given order gains 3.6
points (0.497 → 0.533) but the spread across orders stays at 5.9 points (from 7.1), against
DiffusionGemma's 5.0 → 1.4 (bf16) and 10 → 2 (4-bit). That is what the mechanism predicts: in a single
bidirectional stack the state is encoded in the context of the question list wherever the list sits,
so moving the list after the state cannot isolate the state's encoding. The residual gain is
consistent with a positional/recency effect (the state adjacent to the canvas). The mean slot is
worse than the mask under every format in LLaDA (0.42–0.52), confirming it is a uniform-state trick.
Conclusion for the paper: the order effect is general to one-pass canvas reads; the encoder-origin
mechanism and the state-first fix are properties of the causal-encoder architecture, and the fix
works fully exactly where the mechanism says it should.
