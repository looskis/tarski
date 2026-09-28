#!/bin/bash
# Replaces lane 1 of lambda_queue.sh after its first step (typed_cs_s0): a lighter plan, since the
# min-300-step budget makes every typed-decisions run (full fine-tunes included) ~34 epochs.
# Usage (on the instance): bash experiments/lambda_lane1b.sh <pid of the running typed_cs_s0 python>
cd "$(dirname "$0")/.." || exit 1
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=.venv/bin/python
R=results/tarski
CS="customer_service.action customer_service.category customer_service.churn_risk customer_service.needs_human customer_service.urgency"
run() { name=$1; shift; echo "$(date +%H:%M:%S) start $name" >> $R/lane1.log
        "$@" > "$R/run_$name.out" 2>&1; echo "$(date +%H:%M:%S) end $name (exit $?)" >> $R/lane1.log; }
while kill -0 "$1" 2>/dev/null; do sleep 10; done
echo "$(date +%H:%M:%S) end typed_cs_s0 (lane1b took over)" >> $R/lane1.log
run typed_probes_s0 $PY -m experiments.sweep --dataset typed-decisions --probe-splits 6 11 16 22 --block-splits \
  --seed 0 --out $R/sweep_typed_s0.json
for s in 1 2; do
  run typed_cs_s$s $PY -m experiments.sweep --dataset typed-decisions --tasks $CS --probe-splits 22 --block-splits 20 \
    --depths 2 --block-epochs 5 --full --full-tasks $CS --seed $s --out $R/sweep_typed_cs_s$s.json
done
echo "$(date +%H:%M:%S) LANE1_DONE" >> $R/lane1.log
