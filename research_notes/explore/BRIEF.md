# Exploration brief (shared by all exploration agents)

## The project

`tarski` (repo root `/Users/lookevink/projects/tarski`) is a local-first harness for **decision models**:
many small typed decisions (route to team, category, urgency, out-of-scope, yes/no checks) about each
message (Slack messages, support tickets, JSON states). The user currently uses laya
(github.com/NandhaKishorM/laya), a ModernBERT-large cross-encoder that puts the question and options in
the input and so re-reads the message for every question. Jev (TypeSafe) is the closed commercial
equivalent; OpenJev (github.com/razorback16/openjev) serves open ones behind Jev's API.

What exists (read the code before proposing, it is short):
- `tarski/trunk.py`: frozen `answerdotai/ModernBERT-base` (22 layers, hidden 768, alternating
  global/local attention: every 3rd layer is global, others use a 128-token sliding window). `Trunk.taps`
  runs the trunk once and returns hidden states at requested depths; `Context` holds masks + RoPE.
- `tarski/branches.py`: per-task branches reading the trunk at split depth k. `probe` (LayerNorm, mean
  pool, linear) and `blocks` (d layers copied from base layers [k, k+d), fine-tuned, final norm, pool,
  linear; `init` = next/top/random).
- `tarski/train.py`: `FeatureCache` (trunk states cached once, fp16 CPU), `train_branch`, `fit`
  (per-task training + temperature calibration + metrics: acc, macro-F1, ECE, NLL, Brier vs soft labels).
- `tarski/data.py`: datasets as messages carrying several decisions: `banking77` (1 task, 77 intents),
  `clinc150` (3 tasks per message: intent 151 incl. oos, domain 11, oos binary), `typed-decisions`
  (20 tasks = 4 workflows x 5 questions over JSON states, soft teacher labels; laya's full fine-tune
  scores 0.766 on it), and user CSV/JSONL.
- `tarski/autosplit.py`: per-task split depth chosen from a one-pass linear-probe-by-layer curve.
- `tarski/fused.py`: N same-shape branches run as one batched computation.
- `tarski/engine.py`, `tarski/server.py`, `tarski/cli.py`: resident trunk, LRU-loaded branches, Jev API.
- `experiments/`: `sweep.py`, `latency.py`, `ablation_init.py`, `autosplit_eval.py`; results in
  `results/tarski/` (`tables.md` is the rendered summary). `readonce/` is a sibling line of work
  splitting laya's cross-encoder (do not modify it).

## Results so far (test accuracy; full = full fine-tune of that task)

- Banking77 intent: full 94.0. blocks@18+1 93.1, blocks@14+2 92.8, blocks@8+2 92.0, blocks@4+2 91.1;
  probes at any depth 87.3–88.7. Branches train in ~1 min on the laptop; probes in seconds.
- CLINC150 (partial): probes ~85 mean; blocks@11+2 87.0 mean. **Out-of-scope detection is weak**:
  the oos binary task is ~85–87% while always answering in-scope gives 81.8%.
- typed-decisions: **probes collapse** (many tasks predict the majority class at every depth; mean
  51–55% vs laya 76.6%). JSON states, 240 tokens median, ~1,080 training messages, soft labels.
- laya split without retraining (sibling work): 0.766 at no split, 0.617 when question and message are
  encoded separately for the first 14 layers, 0.453 from 20 layers on.

## Prior art already established (do not re-propose these as novel)

Shared frozen trunk with per-task top layers at per-task depth, trained independently and merged
(Wei, Qi & He, ACL 2022); Mainstream (USENIX ATC 2018); AdapterDrop; Embedding Recycling; Anthropic's
representation re-use classifiers (2025); copy-initialised layers (MORES, LUMEN); multi-LoRA serving
(S-LoRA, Punica); ReFT/LoReFT; steering vectors; task arithmetic/TIES; SetFit; linear probes and
SAE probes; Text-to-LoRA/hypernetworks for generation; early exit. See
`research_notes/novelty_check.md` and `reports/Appending layers to language models.md`.

## Constraints for your work

- Machine: Apple M6, 32 GB unified memory, PyTorch MPS. **A training queue is using the GPU. Do not run
  GPU jobs.** You may run CPU-only smoke tests on tiny subsets (under ~2 minutes each, `device="cpu"`).
- Do not modify files under `tarski/`, `experiments/`, `readonce/`, `tests/`. Put code in
  `explore/<your_lens>_<idea>.py` (importing from `tarski` is fine). If an idea needs a core change,
  describe the change instead.
- Each runnable experiment must have `--smoke` (CPU, tiny subset, finishes quickly, you run it and it
  passes) and a full mode (`--out results/tarski/explore_<name>.json`) that we will queue on the GPU. The
  full mode should finish in under ~30 minutes on the M6 GPU. Log key metrics; compare against the
  relevant baseline from the results above (re-computing it inside the script if needed).
- The Python environment is `/Users/lookevink/projects/tarski/.venv/bin/python`; run from the repo root.
