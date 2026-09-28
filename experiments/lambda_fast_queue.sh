#!/bin/bash
# Runs ON the A100 (.lambda_instance_fast.json): the jobs the shared A10 could not fit.
#   lane A: typed-decisions full fine-tune references, customer service, seeds 0-2 (full config only)
#   lane B: KV-query decision streams on typed-decisions with a 60-minute budget (27 was not enough on the A10)
cd "$(dirname "$0")/.." || exit 1
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
R=results/tarski
CS="customer_service.action customer_service.category customer_service.churn_risk customer_service.needs_human customer_service.urgency"
run() { lane=$1; name=$2; shift 2; echo "$(date +%H:%M:%S) start $name" >> $R/fast_lane$lane.log
        "$@" > "$R/run_$name.out" 2>&1; echo "$(date +%H:%M:%S) end $name (exit $?)" >> $R/fast_lane$lane.log; }
laneA() { for s in 0 1 2; do
  run A typed_cs_full_s$s $PY -m experiments.sweep --dataset typed-decisions --tasks $CS --probe-splits --block-splits \
    --full --full-tasks $CS --seed $s --out $R/sweep_typed_cs_full_s$s.json
done; echo "$(date +%H:%M:%S) LANE_A_DONE" >> $R/fast_laneA.log; }
laneB() {
  run B kvquery_typed_long $PY explore/systems_kvquery.py --dataset typed-decisions --split 11 --budget-min 60 \
    --out $R/explore_kvquery_typed_long.json
  echo "$(date +%H:%M:%S) LANE_B_DONE" >> $R/fast_laneB.log; }
laneA & laneB &
wait
