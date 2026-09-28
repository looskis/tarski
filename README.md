# tarski

Local-first decision models: train small task branches on a frozen base encoder with your own
labelled data, and serve many decisions per message from one shared pass through the base.

```
message ──► trunk (frozen ModernBERT-base, runs once, stops at the deepest split needed)
              │ layer 8 ──► branch "urgent"    (probe, 60 KB)
              │ layer 11 ─► branch "team"      (2 layers + head, 20 MB)
              └ layer 14 ─► branch "category"  (2 layers + head, 20 MB)
```

- **Adding a decision never touches the base.** Each branch trains on cached trunk states in seconds
  to minutes, on a laptop, and cannot break the branches already deployed.
- **Task switching is a file load, not a model reload.** The base stays resident; branches are loaded
  from disk on demand and evicted least-recently-used.
- **An extra decision costs a few layers, not a full model pass.**
- **Speaks Jev's API** (`POST /v1/systemone`), so `typesafe-sdk`, OpenJev routes and stuntd work
  unchanged. Questions with no trained branch, or low-confidence answers, can fall back to any
  Jev-compatible server (laya, OpenJev, Jev).

## Install

```bash
uv sync            # Python 3.12; PyTorch uses CUDA, Apple silicon (MPS) or CPU
```

The base model (`answerdotai/ModernBERT-base`, ~600 MB) downloads from Hugging Face on first use.

## Train on your data

One row per message, a `text` column, and one column per decision. Empty cells are fine: a row only
trains the decisions it is labelled for. An optional `split` column (`train`/`val`/`test`) is used
if present; otherwise rows are split 80/10/10.

```csv
text,team,urgent
"Everything is down and we have a demo at noon",outage,yes
"I was charged twice this month",billing,no
"Could you add dark mode?",feature,
```

```bash
python -m tarski train tickets.csv --tasks team urgent       # one branch per column
python -m tarski list                                          # size, split depth, test accuracy
python -m tarski predict "The site is down for everyone" --tasks team urgent
```

Training prints test accuracy, macro-F1 and calibration error per task, and fits one temperature per
branch so its confidence is usable for thresholds. Options: `--kind blocks|probe`, `--split` (trunk
depth the branch reads), `--depth` (layers in a blocks branch), `--epochs`.

## Serve

```bash
python -m tarski serve --port 8080
python -m tarski serve --port 8080 --fallback http://127.0.0.1:8081 --escalate-below 0.4
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

| Endpoint | |
|---|---|
| `POST /v1/decide` | `{text or texts, tasks}` → label, probabilities, confidence per task, and timing |
| `POST /v1/systemone` | Jev's request and answer shapes |
| `GET /v1/tasks` | trained tasks, resident or not, size, metrics |
| `POST /v1/tasks/{task}/load`, `/unload` | manage residency explicitly |

`TARSKI_API_KEY` requires a bearer token; `TARSKI_FALLBACK_KEY` is sent to the fallback server.

## Research

`experiments/` holds the thesis measurements: accuracy by split depth and branch type
(`sweep.py`) and the per-message cost of N decisions and of task switching (`latency.py`).
Results go to `results/tarski/`. `readonce/` held a separate line of work on splitting laya's
question-in-input cross-encoder. It was retired on 2026-09-27; only the modules imported by
`explore/` and `experiments/` remain (see `readonce/README.md`).
