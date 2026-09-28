# tarski

**≈2× faster. >3× fewer decoder passes. +5.3 percentage points accuracy compared with OpenJev.**

Same DiffusionGemma model, unchanged weights, a better read policy:

| Measure | OpenJev reference | Tarski read policy |
|---|---|---|
| End-to-end latency per state | 995 ms (four passes) | ≈560 ms (one pass) — **1.8× faster** |
| Decoder passes per state | 3.65 on average with automatic re-reads | **1** |
| Decision accuracy | 65.9% (given order, random slot) | **71.2% (+5.3 percentage points)** |

Accuracy: 400 typed-decisions states / 2,000 decisions, comparing single reads with the default
question order and random slot against a selected order and vocabulary-mean slot. Timing: 20
states on the same Apple-silicon laptop with DiffusionGemma 26B-A4B in 4-bit precision; the latency
comparison includes prefill and uses four passes for OpenJev's re-read path. These are separate
accuracy and timing comparisons. [Measurements and methodology](docs/research/notes/decision_models/LOG.md).

Turn diffusion-model research into read policies for your own decisions. Compare prompt layout,
question order and answer-slot initialization on a labelled sample, fit confidence temperatures,
then use the resulting JSON config for inference. Model weights stay fixed.

The CLI is noninteractive. Successful commands return JSON on stdout; predictions are JSONL.
Errors and progress go to stderr. Validation, tuning and evaluation use only the Python standard
library and never load a model. `tarski schema` describes the command and data contract for agents.

## Install

Python 3.12 or newer:

```bash
pip install .
tarski --help
tarski schema
```

For development: `uv sync --group dev`, then `uv run tarski ...`.

## Try the complete offline workflow

These tiny, synthetic fixtures demonstrate the file contract, not model quality:

```bash
tarski init --backend mlx --out /tmp/tarski.base.json
tarski validate --data examples/calibration.reads.jsonl --kind reads
tarski tune --config /tmp/tarski.base.json --reads examples/calibration.reads.jsonl \
  --out /tmp/tarski.fitted.json
tarski eval --config /tmp/tarski.fitted.json --reads examples/evaluation.reads.jsonl
```

`tune` chooses a single global combination of prompt format, order and slot using labelled
calibration questions. Accuracy is the default objective, with Brier score breaking ties;
`--objective brier` selects by uncalibrated Brier score. It then fits one temperature per question
type by negative log likelihood. Temperature changes confidence, not the top label.

The tuning report is **in-sample**. Use a separate labelled file for `eval`; it rejects repeated
record IDs and, when available, repeated state hashes. Keep IDs stable between runs. Existing
experiment files without state hashes cannot detect the same state under a new ID. An unseen
question type uses the config's default temperature and is reported in the evaluation warnings.

## Your data

One JSON object per line, with a stable `id`, a `state`, and a dictionary of `questions`:

```json
{"id":"ticket-1","state":"I was charged twice.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and invoices","support":"Technical issues"}},"urgent":{"type":"noul","instructions":"Is the service unavailable?"}},"gold":{"team":"billing","urgent":"no"}}
```

- `state` can be text, an object or an array.
- `choice` uses 2–255 named criteria; `noul` uses `yes`/`no`; `score` uses an array of 2–10
  descriptions and string labels `"0"`, `"1"`, etc.
- `gold` is optional for inference. Each value can be a label string or an object containing
  `label`. Missing/null labels are unlabelled; tuning and evaluation need at least one label.
- Each record must fit in one answer canvas. Oversized schemas fail explicitly; splitting them
  changes the question context and is not done automatically.
- `--data -` and `--reads -` read JSONL from stdin. Output paths are never overwritten without
  `--force`, and an output cannot replace an input even with that flag.

## Run on your data

The reader adapters reuse the pinned OpenJev prompt and canvas implementation. Install it plus
one of the optional backends. Model weights may download on the first `collect` or `predict`.

```bash
# Shared prompt/canvas implementation, pinned to the research revision:
pip install 'openjev @ git+https://github.com/razorback16/openjev@a0ddd7d928298eccef2c17153b00b5636b6d996a'
# DiffusionGemma on Apple silicon (about 14 GB of weights):
pip install '.[dlm]'
# Or DiffusionGemma / LLaDA on CUDA:
# pip install '.[torch]'

tarski init --backend mlx --out base.json
tarski doctor --config base.json
tarski validate --data calibration.jsonl
tarski collect --config base.json --data calibration.jsonl --out calibration.reads.jsonl \
  --formats stock state_first --orders given reversed rotate:1 --slots native mean
tarski tune --config base.json --reads calibration.reads.jsonl --out fitted.json

tarski collect --config fitted.json --data heldout.jsonl --out heldout.reads.jsonl
tarski eval --config fitted.json --reads heldout.reads.jsonl
tarski predict --config fitted.json --data incoming.jsonl --out predictions.jsonl
```

`doctor` checks dependencies, not hardware capacity or cached weights. `collect` and `predict`
load the model once per command. The first implementation holds records/results in memory and
writes the output atomically when complete; it has no streaming or resume support yet. Start
with a small calibration file: the example sweep runs 12 reads per state. A checkout of OpenJev
under `third_party/openjev` is also recognized by the research readers.

## Config and agent interface

```json
{
  "schema_version": 1,
  "backend": "mlx",
  "model": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
  "read": {
    "prompt_format": "state_first",
    "question_order": "given",
    "slot": "mean",
    "seed": 0
  },
  "calibration": {"default_temperature": 1.0, "temperatures": {}}
}
```

`init --backend torch` selects DiffusionGemma bf16 on CUDA; `--backend llada` selects LLaDA-8B
with stock prompting and its native mask slot. `--model` accepts an alternative checkpoint or local
path of the same architecture. Prompt formats are `stock`, `state_first`, and the experimental
`user_first` control. Order is `given`, `reversed`, or `rotate:N`; a rotation wraps modulo the
number of questions. `native` means a seeded random vocabulary token for DiffusionGemma and the
trained mask token for LLaDA. `mean` replaces the slot with the vocabulary-mean embedding.

A fitted config also records data fingerprints and selection provenance. `collect` records the
backend, checkpoint and seed; tuning and evaluation reject mismatched reader metadata. Collected
probabilities are raw; `predict` and `eval` apply the saved temperatures. Edit unfitted configs
and run `tarski validate --config ...` before use. Retune after changing a fitted read policy.

| Command | Purpose |
|---|---|
| `schema` | Machine-readable command contract and examples |
| `init` | Write a starter config |
| `validate` | Check config/data without loading weights |
| `doctor` | Check reader dependencies |
| `collect` | Run candidate policies on local data |
| `tune` | Select a policy and fit per-type temperatures |
| `eval` | Report accuracy, Brier, NLL and 10-bin ECE on held-out reads |
| `predict` | Emit calibrated label distributions as JSONL |

Exit codes: `0` success, `1` runtime/dependency failure, `2` invalid input, `130` interrupted.
An error is `{"error":{"code":"invalid_input","message":"..."}}` on stderr. There are no prompts.
`--help` is human-readable; `schema` is JSON. Offline commands do not import PyTorch or MLX.

The tuner also reads the stored `dlm/order.py`, `dlm/state_first.py` and `dlm/bitext.py` results.
It recognizes `rotN`, `rev`, their `_mean` variants, and `format|order|slot` names. Hybrid prompt/
canvas-order experiments are reported as skipped because they do not correspond to a deployable
policy. Every question must have the same candidate set, preventing partial runs from changing
which examples get scored. Older result files lack reader metadata: the tool warns and uses the
backend/checkpoint you explicitly supply in the base config.

## Research

[Experiment log](docs/research/notes/decision_models/LOG.md) ·
[Research brief](docs/research/notes/decision_models/BRIEF.md) ·
[Reproduction scripts](dlm/README.md) · [Stored results](results/dlm/)

The experiments identify three useful controls: prompt layout, question order, and per-type
confidence calibration. The vocabulary-mean slot helped DiffusionGemma but hurt LLaDA relative
to its mask token. State-first prompting reduced DiffusionGemma's order sensitivity, with a
smaller effect on LLaDA. These findings motivate candidates and defaults; your calibration and
held-out results determine whether a setting helps your workload. Tuning does not establish
out-of-distribution reliability or change model weights.
