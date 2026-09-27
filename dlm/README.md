# dlm: reading a diffusion language model as a decision model

Experiments on DiffusionGemma 26B-A4B as OpenJev uses it, on Apple silicon (4-bit, MLX), with the
canvases built by OpenJev's own engine so nothing about the prompt differs from the served system.

## The question

OpenJev seeds each answer slot with a random vocabulary token and takes one read-only denoise pass.
DiffusionGemma is uniform-state diffusion, so that read is one sample of the marginal conditioned on
the noise, not the marginal; OpenJev papers over it by re-reading with fresh seeds when a slot's
entropy exceeds 0.1. We ask:

1. **Anatomy.** How much does a single read vary with the seed? What do the re-reads buy? Are the
   entropy-derived confidences calibrated, per question type? Does one slot's answer depend on the
   noise in another slot?
2. **Learned denoising queries.** Replace the random token with a learned input embedding (one per
   question type, ~3k parameters each, weights untouched), trained so that a single read reproduces
   the model's own noise-averaged read (label-free) or the gold distribution. Deterministic reads, no
   re-reads, and, if the diffusion-LM overconfidence literature is right, a handle on calibration.
3. **Slot isolation.** Inference-time canvas masks so slots cannot attend to each other's noise;
   then answer-conditioned second reads for joint structure.

## Files

| File | Purpose |
|---|---|
| `reads.py` | `Reader`: loads the model, reuses `openjev.engine.Engine` for prompt/template/slots, caches prefills, runs the decoder from input embeddings (random token, vocabulary-mean, or a query), logits only at slot rows; slot-isolation masks; routing recorder for the expert census |
| `data.py` | JevBench public items and LocalLLaMA/typed-decisions as Jev-format records |
| `anatomy.py` | Phase A: 16 seeded reads + the mean-embedding read per canvas, resumable JSONL |
| `metrics.py` | Scores anatomy files: single / auto4 / mean4 / meanK / meanemb / query conditions; JevBench accuracy, Brier, ECE; TV and flips against meanK; seed variance |
| `train_query.py` | Learned denoising queries, trained on an anatomy file's targets; evaluates per epoch |
| `apply_query.py` | Adds a trained query's reads to another anatomy file (zero-shot transfer) |
| `smoke.py`, `steer_smoke.py` | Feasibility: parity with the stock read, gradient through the quantised MoE, steering one item |

## Setup

```bash
uv sync --extra dlm --extra benchmarks
# vendored sources (ignored by git): pinned commits in this README
git clone --depth 1 https://github.com/razorback16/openjev third_party/openjev      # a0ddd7d
git clone --depth 1 https://github.com/fstandhartinger/jevbench third_party/jevbench # fd54ea7
.venv/bin/python dlm/smoke.py
```

The weights (`mlx-community/diffusiongemma-26B-A4B-it-4bit`, ~14 GB) download on first use. A read
costs ~0.1 s after a prefill of 0.3–4 s depending on state length; a gradient step through the
decoder for one canvas costs 0.2–1 s at ~19 GB peak memory.

## Runs

```bash
.venv/bin/python dlm/anatomy.py --dataset jevbench   --seeds 16 --out results/dlm/anatomy_jevbench.jsonl
.venv/bin/python dlm/anatomy.py --dataset typed-test --seeds 16 --out results/dlm/anatomy_typed_test.jsonl
.venv/bin/python dlm/anatomy.py --dataset typed-train --seeds 16 --out results/dlm/anatomy_typed_train.jsonl
.venv/bin/python dlm/metrics.py results/dlm/anatomy_jevbench.jsonl --by type
.venv/bin/python dlm/train_query.py --train results/dlm/anatomy_typed_train.jsonl \
    --test results/dlm/anatomy_typed_test.jsonl --target meanK --per type --epochs 3 --name meanK_type
.venv/bin/python dlm/apply_query.py --queries results/dlm/queries_meanK_type.safetensors \
    --dataset jevbench --anatomy results/dlm/anatomy_jevbench.jsonl
```

## Notes

- mlx-vlm's router only stop-gradients its top-k expert indices in training mode; `reads.py` patches
  the router so a backward pass works in eval mode (forward values unchanged).
- Parity with the stock `diffusion_decoder_logits` path is exact on the label distribution; raw bf16
  vocabulary logits differ at the 1e-1 level because only the slot rows go through the output
  projection here.
- JevBench's public items are the evaluation set and are never used to train a query; typed-decisions
  train is the training set, typed-decisions test and JevBench public are held out.
