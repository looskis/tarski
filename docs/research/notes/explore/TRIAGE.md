# Exploration triage (2026-09-27)

Five agents, one lens each (systems, geometry, learning, far-field, skeptic), produced 57 ideas with
prior-art checks in `systems.md`, `geometry.md`, `learning.md`, `farfield.md` and `skeptic.md`. This
file ranks them and records what is queued. Novelty verdicts are the agents' quick searches: check any
idea properly before claiming it.

## Bugs and methodology problems (found by the agents, acted on)

| Problem | Evidence | Action |
|---|---|---|
| Early stopping on ~30 validation rows during LR warm-up returned near-untrained typed-decisions probes (62% stopped at epoch 4 of 20) | `sweep_typed_buggy_earlystop.log`; found independently by far-field, skeptic and geometry | Fixed in `tarski/train.py` (no stopping before half the schedule; below 50 validation rows keep the last epoch). Typed runs set aside and re-queued |
| CLINC out-of-scope prior shift (1.6% train, 3.2% val, 18.2% test) | skeptic | Prior-shift correction queued (`explore_oos_priorshift`); report CLINC oos with this caveat |
| No same-architecture full fine-tune for typed-decisions; laya comparison mixes model sizes | skeptic | Full reference for the 5 customer-service decisions queued (`sweep_typed_cs`) |
| Temperature fitted to soft labels, ECE scored on hard labels (typed-decisions) | skeptic | Open: report both, or fit to hard labels for ECE |
| Probe-guided split selector unvalidated; on flat curves it picks shallow depths | skeptic | `autosplit_eval` queued on all three datasets |
| Banking77/CLINC inputs are shorter than the 128-token local window, so local/global findings only apply to typed-decisions | skeptic, geometry | Note in thesis; receptive-field and global-split tests queued |
| Latency only at batch 1; CPU run was noisy (agents' CPU tests ran concurrently) | skeptic, own observation | Clean CPU latency rerun queued last |

## Ranked ideas

Rank = novelty x feasibility x relevance to local-first routing, after deduplicating across lenses.

| # | Idea (lens) | What it could show | Novelty (agent's check) | Status |
|---|---|---|---|---|
| 1 | **One-way questions on laya** (geometry) | laya becomes read-once with no retraining: the message never attends to the question, so it is encoded once and each question costs only its own tokens. 60-pair CPU sample: 0.700, same as unsplit laya; the readonce block split: 0.333 | Partial: trained asymmetric attention exists (MICE 2026, sparse cross-encoders 2024); no-retraining conversion not found | Running now |
| 2 | **KV-query decision streams** (systems) / **passenger tokens** (geometry) | Each decision is a 1–4 token stream reading the shared trunk's keys/values: ~0.02 GFLOP per extra decision vs ~5 for a 2-layer branch; attention can pick out the JSON field a decision needs | Partial (RPO, VQT, attention probes, aLoRA) | Queued (typed, Banking77; passenger on typed + CLINC) |
| 3 | **Branch merging** (learning) | N independent branches that share an initialisation merged (mean/TIES) into one body: N x d layers become d | Not found at this shape | Implemented (`explore/merge_branches.py`), queued (CLINC, customer service) |
| 4 | **Syndrome decoding of decision bundles** (far-field) | Inconsistency between a message's decisions (intent vs domain) as a label-free out-of-scope score; joint decoding of the bundle | Done in vision (consistency energy, CVPR 2020); not found for NLP decision bundles | Queued |
| 5 | **JSON field gates** (far-field) / **token-centred pooling** (geometry) | Pool trunk states per JSON field; each decision learns which fields to read. Interpretable (`duplicate` reads `purchase_order.id`) | Components known; combination new to this setting | Queued |
| 6 | **Active distillation from local laya** (learning) | How many teacher queries (not human labels) a user needs; laya labels only what the branch is unsure about | Combination not found | Queued |
| 7 | **Incremental re-encoding for Slack threads** (systems) | Reuse cached states when a thread grows; measure whether decisions change | Mechanism known (CacheBlend etc.); not for encoder decision models | Queued |
| 8 | **Global-layer-aligned splits** (geometry) | Are splits just after a global-attention layer better? | Not found | Queued (Banking77 1-layer branches at 15–20; depthscan on typed) |
| 9 | Prior-shift-corrected out-of-scope gate (skeptic) | Fixes most of CLINC's weak oos | Known (Saerens 2002, BBSE 2018) | Queued |
| 10 | Proper-score / REINFORCE / bias-only (K+1 params) objectives (learning) | Whether RL-style rewards help tiny routing heads | Narrow gap | Queued |
| 11 | Shared attention, private MLP in fused branches (far-field) | Attention computed once for N branches | Architecture known; serving consequence new | Not implemented |
| 12 | Per-decision-set pruned trunk (systems) | Drop heads/channels a request's branches do not use, no retraining | Combination likely new | Not implemented |
| 13 | Schema templates for JSON states (systems) | 44% of typed-decisions tokens are schema; encode the schema once | Novel, most work | Not implemented |
| 14 | Description-anchored routes (geometry) | A new route from a label description, no labels | Close to prior work | Not implemented |

Also recorded but not pursued: negative selection and depth-disagreement OOD (reduce to known kNN /
Mahalanobis / MOOD / BLOOD), sleep consolidation (BAM!, RegMean), evolutionary topology search
(Learning to Branch), Kalman fusion over depth (ZTW), self-training, curriculum, route addition by
weight imprinting.

## Queue

`experiments/queue3.sh` (+ `queue3_extra.sh`) runs after the ablation and auto-split evaluations; logs
in `results/tarski/run_<name>.out`, results in `results/tarski/explore_*.json`.
