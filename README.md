# tarski

Local decision models. Train small **branches** for your own decisions (which team, how urgent,
which category, escalate or not) on top of a frozen encoder, with your own labelled messages, on
your own machine. One pass through the encoder answers every decision about a message.

```
message ──► trunk (frozen ModernBERT-base, runs once, stops at the deepest split any branch needs)
              │ layer 8 ──► branch "urgent"    (linear probe, 0.03 MB)
              │ layer 14 ─► branch "team"      (2 transformer layers + head, 20 MB)
              └ layer 14 ─► branch "category"  (2 transformer layers + head, 20 MB)
```

- **Accurate enough.** With about 100 labelled messages per route, a 2-layer branch lands within
  about 1 point of fine-tuning the whole encoder (Banking77 92.9 vs 93.8; JSON ticket states 78.5 vs
  79.6).
- **Cheap per decision.** Twenty decisions about one message cost about 10x less than twenty
  separately fine-tuned models, and about 20x less than a question-in-input decision model.
- **Adding a decision never touches the encoder.** A branch trains on cached encoder states in
  seconds to minutes on a laptop, and cannot change any branch already deployed.
- **Knows what it does not know.** Every branch carries an out-of-scope detector, fitted without
  out-of-scope examples, and a calibrated confidence.
- **Speaks Jev's API** (`POST /v1/systemone`), so `typesafe-sdk`, OpenJev routes and stuntd work
  unchanged, with optional fallback to any Jev-compatible server for untrained questions.

The measurements behind these claims are in [docs/research/THESIS.md](docs/research/THESIS.md).

## Install

```bash
uv sync                      # Python 3.12; PyTorch on CUDA, Apple silicon (MPS) or CPU
uv run tarski info           # device, cached encoders, branch store
```

Or into any environment: `pip install .` (add `.[benchmarks]` for the public datasets). The default
encoder, `answerdotai/ModernBERT-base` (~600 MB), downloads from Hugging Face on first use.

## Quickstart

`examples/tickets.csv` has 80 support messages with two decisions each: `team` (billing, outage,
feature, account) and `urgent` (yes, no).

```bash
tarski train examples/tickets.csv            # one branch per label column, ~1 min on a laptop
tarski list                                  # kind, split depth, size, test accuracy, out-of-scope
tarski predict "Prod is down and customers cannot log in"
tarski eval examples/tickets.csv --all-rows  # accuracy, F1, calibration on the file
tarski serve --port 8080
```

`predict` prints one JSON line per message:

```json
{"text": "Prod is down and customers cannot log in",
 "decisions": {"team": {"label": "outage", "confidence": 0.91, "out_of_scope": false},
               "urgent": {"label": "yes", "confidence": 0.84, "out_of_scope": false}}}
```

## Your data

One row per message: a `text` column and one column per decision. A CSV or JSONL file.

```csv
text,team,urgent
"Everything is down and we have a demo at noon",outage,yes
"I was charged twice this month",billing,no
"Could you add dark mode?",feature,
```

- An empty cell means the row is not labelled for that decision; it still trains the others.
- A `split` column with `train` / `val` / `test` is honoured; otherwise rows are split 80/10/10 with
  a fixed seed, so `eval` scores rows that training never saw.
- Labels are the distinct values in a column, sorted. A decision needs at least two.
- `--text-col` and `--tasks` pick the columns; `--max-len` (default 256 tokens) truncates long messages.

How many labels? Around **100 messages per label** puts a branch within about a point of a full
fine-tune. With **10 per label** expect a gap of 4–12 points; the encoder choice below matters more
than the branch there.

## Choosing the encoder, branch and depth

| Option | Default | When to change it |
|---|---|---|
| `--base` | `modernbert` (answerdotai/ModernBERT-base) | `gte` (Alibaba-NLP/gte-modernbert-base, same shape, contrastively trained) lifts a plain probe to full-fine-tune level and makes few-label routes stronger. One seed of evidence so far; try it. |
| `--kind` | `blocks` (1–2 transformer layers + head, ~20 MB) | `probe` (linear head, 0.03 MB, 0.2 ms per extra decision) when you have many decisions or use the `gte` encoder. |
| `--split` | `14` (the depth the branch reads from) | `auto` runs a short 1-layer branch at depths 2, 4, …, 20 and picks the shallowest within 1 point of the best. It needs at least 100 validation rows per decision; with fewer it falls back to 14. Shallower splits make the whole request cheaper. |
| `--depth` | `2` | `1` halves the branch; both are within a point on our benchmarks. |
| `--epochs` | `8` | A floor of 300 optimiser steps applies regardless, so small files run more epochs. Do not lower this for small data. |

All branches in a store share one encoder. Each branch is bound to a fingerprint of the exact
encoder weights it was trained on and refuses to load against another; change the encoder and
retrain (minutes, from cached states).

## Out of scope

A branch answers only the labels it was trained on. To notice messages that fit none of them, each
branch stores the class means and shared covariance of its training messages' encoder states and
scores new messages by relative Mahalanobis distance. The threshold flags about 5% of ordinary
in-scope traffic (`--oos-quantile 0.95`); on CLINC150 this separates out-of-scope messages with
AUROC 0.957 without a single out-of-scope example. Scoring reads the states the branch already
receives, so it costs nothing extra. `predict` and `/v1/decide` report `out_of_scope` and
`oos_score`; `serve --escalate-oos` sends flagged questions to the fallback. `--no-oos` skips it.

The detector is as good as the data it is fitted on. With a few hundred messages per decision it
separates near-miss topics; with a few dozen (the quickstart file) it reliably catches gibberish,
greetings and clearly foreign text but misses some short natural sentences. When the file has fewer
than 30 validation rows the threshold is set from leave-one-out scores of the training rows, which
keeps the flag rate on unseen in-scope messages near the 5% target.

## Evaluate

```bash
tarski eval tickets.csv                 # the file's test split (or the seeded 10% carve-out)
tarski eval new_week.csv --all-rows     # every row of a file the branches never saw
tarski eval banking77                   # a public benchmark (needs the benchmarks extra)
```

Per decision: accuracy, macro-F1, negative log-likelihood, expected calibration error, and the share
of rows the detector flagged (should sit near 5% on in-scope data).

## Serve

```bash
tarski serve --port 8080
tarski serve --port 8080 --fallback http://127.0.0.1:8081 --escalate-below 0.4 --escalate-oos
```

```bash
curl -s localhost:8080/v1/decide -H 'content-type: application/json' \
  -d '{"text": "Prod is down, customers cannot log in", "tasks": ["team", "urgent"]}'
```

Jev-compatible, with question ids matching trained task names:

```python
from typesafe_sdk import TypeSafeClient
client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8080")
client.system_one("I was charged twice", {
    "team":   {"type": "choice", "criteria": {"billing": "...", "outage": "...", "feature": "..."}},
    "urgent": {"type": "noul"},
})
```

A choice question may list a subset of a branch's labels; probabilities are renormalised over it.
Questions with no trained branch, answers below `--escalate-below`, and (with `--escalate-oos`)
messages a branch flags as out of scope go to the fallback server; the `X-Tarski-Sources` and
`X-Tarski-Out-Of-Scope` response headers say which.

| Endpoint | |
|---|---|
| `POST /v1/decide` | `{text or texts, tasks}` → label, probabilities, confidence, out-of-scope flag per task, timing |
| `POST /v1/systemone` | Jev's request and answer shapes |
| `GET /v1/tasks` | trained tasks: kind, depth, size, metrics, resident or not |
| `POST /v1/tasks/{task}/load`, `/unload` | manage residency explicitly (default: LRU over `--max-loaded`) |
| `GET /health` | encoder and device |

`TARSKI_API_KEY` requires a bearer token on `/v1/*`; `TARSKI_FALLBACK_KEY` is sent to the fallback.
`TARSKI_STORE` sets the default branch directory.

## Python

```python
from tarski.engine import Engine

eng = Engine("branches")                                   # loads the encoder once
r = eng.decide(["The site is down for everyone"], ["team", "urgent"])
r["results"][0]["team"]                                    # probabilities over the branch's labels
r["oos"][0]["team"]                                        # {"score": ..., "flag": ...}
```

```python
from tarski import data, train
from tarski.trunk import Trunk

ds = data.load_table("tickets.csv")
trunk = Trunk()                                            # frozen ModernBERT-base on the best device
train.fit(trunk, ds, ["team", "urgent"], split=14, depth=2, store="branches")
```

## How it works

The encoder runs once per message and stops at the deepest layer any requested branch reads. A
branch reads the hidden states at its depth: a **probe** is LayerNorm, mean pooling and a linear
layer; a **blocks** branch is one or two transformer layers copied from the encoder, then the same
readout. Training caches the encoder's states for the training messages once (fp16) and fits only
the branch, so no gradient ever reaches the encoder: adding, retraining or deleting a branch cannot
change another branch's answers, and a unit test checks this bit for bit. Each branch is calibrated
with one temperature fitted on validation data, and carries its out-of-scope detector. At serving
time the encoder stays resident and branches are loaded from megabyte-sized files on demand.

## Research

The paper draft, the registry of 91 tested ideas and the literature notes are under
[docs/research/](docs/research/). `experiments/` holds the hypothesis runs (accuracy sweeps,
latency, cold switching, initialisation, depth selection) and `explore/` one script per exploration
idea; both need `pip install .[research]`. Results are in `results/tarski/`. `readonce/` is a
retired line of work kept only for the scripts that import it.
