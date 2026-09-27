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
