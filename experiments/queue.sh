#!/bin/sh
# Run the remaining thesis experiments one after another, after the given PIDs exit.
# Latency runs last and alone, so no training competes with them for the GPU.
# Usage: sh experiments/queue.sh [pid ...]
cd "$(dirname "$0")/.." || exit 1
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 10; done
done
PY=.venv/bin/python
R=results/tarski
$PY -m experiments.sweep --dataset clinc150 --full --probe-splits 6 11 16 22 --block-splits 6 11 16 20 \
  --out $R/sweep_clinc150.json > $R/run_clinc150.out 2>&1
$PY -m experiments.sweep --dataset typed-decisions --tasks customer_service.action customer_service.category \
  customer_service.churn_risk customer_service.needs_human customer_service.urgency --probe-splits \
  --block-splits 8 14 20 --depths 2 --block-epochs 5 --out $R/sweep_typed_cs.json > $R/run_typed_cs.out 2>&1
$PY -m experiments.latency --device mps --out $R/latency_mps.json > $R/run_latency_mps.out 2>&1
$PY -m experiments.latency --device cpu --reps 10 --out $R/latency_cpu.json > $R/run_latency_cpu.out 2>&1
echo QUEUE_DONE
