#!/bin/bash
# Replaces lane A of lambda_fast_queue.sh after seed 0: only seed 1 here (seed 2 runs on the A10).
cd "$(dirname "$0")/.." || exit 1
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/tarski
CS="customer_service.action customer_service.category customer_service.churn_risk customer_service.needs_human customer_service.urgency"
while kill -0 "$1" 2>/dev/null; do sleep 10; done
echo "$(date +%H:%M:%S) end typed_cs_full_s0 (laneA2 took over)" >> $R/fast_laneA.log
echo "$(date +%H:%M:%S) start typed_cs_full_s1" >> $R/fast_laneA.log
.venv/bin/python -m experiments.sweep --dataset typed-decisions --tasks $CS --probe-splits --block-splits \
  --full --full-tasks $CS --seed 1 --out $R/sweep_typed_cs_full_s1.json > $R/run_typed_cs_full_s1.out 2>&1
echo "$(date +%H:%M:%S) end typed_cs_full_s1 (exit $?)" >> $R/fast_laneA.log
echo "$(date +%H:%M:%S) LANE_A_DONE" >> $R/fast_laneA.log
