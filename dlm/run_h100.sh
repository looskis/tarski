#!/bin/sh
# GPU queue (torch backend, bf16): parity with the MLX reads, then the mechanism probes and the
# order-selection procedure. Run on the box:  nohup sh dlm/run_h100.sh > results/dlm/h100_queue.out 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
F='Warning|warn\(|Fetching|torch_dtype|Setting `pad|generation flags|Loading checkpoint'
echo "== parity: torch bf16 vs mlx 4-bit, JevBench hard 60"
$PY dlm/parity_torch.py --anatomy results/dlm/anatomy_jevbench.jsonl --tiers hard --limit 60 --attn sdpa 2>&1 | grep -Ev "$F"
echo "== leakage: attention shares per order, typed-test 200"
$PY dlm/leakage.py --limit 200 --out results/dlm/leakage_typed_test.jsonl 2>&1 | grep -Ev "$F"
echo "== steps: multi-step reads, typed-test 200"
$PY dlm/steps.py --limit 200 --steps 4 --out results/dlm/steps_typed_test.jsonl 2>&1 | grep -Ev "$F"
echo "== layers: visibility sweep + logit lens, typed-test 200"
$PY dlm/layers.py --limit 200 --out results/dlm/layers_typed_test.jsonl 2>&1 | grep -Ev "$F"
echo "== order: rotations on typed-train 1200 (selection set) and typed-test 400 (bf16 replication)"
$PY dlm/order.py --backend torch --split train --limit 1200 --out results/dlm/order_typed_train.jsonl 2>&1 | grep -Ev "$F"
$PY dlm/order.py --backend torch --split test --limit 400 --out results/dlm/order_typed_test_bf16.jsonl 2>&1 | grep -Ev "$F"
echo "== steps on JevBench hard"
$PY dlm/steps.py --dataset jevbench --tiers hard --limit 200 --steps 4 --out results/dlm/steps_jevbench_hard.jsonl 2>&1 | grep -Ev "$F"
echo "== h100 queue done"
