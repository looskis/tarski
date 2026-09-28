#!/bin/bash
# Typed-decisions full fine-tune references (customer service, seeds 0-2), run once lanes 1 and 2 are done:
# with three lanes sharing the A10's 24 GB, the full fine-tune ran out of memory. The sweeps resume, so
# only the missing "full" config runs for each seed.
cd "$(dirname "$0")/.." || exit 1
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/tarski
CS="customer_service.action customer_service.category customer_service.churn_risk customer_service.needs_human customer_service.urgency"
until grep -q LANE1_DONE $R/lane1.log 2>/dev/null && grep -q LANE2_DONE $R/lane2.log 2>/dev/null; do sleep 30; done
for s in 0 1 2; do
  splits="--probe-splits 22 --block-splits 20"; [ $s = 0 ] && splits="--probe-splits 6 11 16 22 --block-splits 8 14 20"
  echo "$(date +%H:%M:%S) start typed_cs_full_s$s" >> $R/lane_full.log
  .venv/bin/python -m experiments.sweep --dataset typed-decisions --tasks $CS $splits --depths 2 --block-epochs 5 \
    --full --full-tasks $CS --seed $s --out $R/sweep_typed_cs_s$s.json > $R/run_typed_cs_full_s$s.out 2>&1
  echo "$(date +%H:%M:%S) end typed_cs_full_s$s (exit $?)" >> $R/lane_full.log
done
echo "$(date +%H:%M:%S) FULL_DONE" >> $R/lane_full.log
