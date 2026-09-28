#!/bin/bash
# Runs ON the Lambda instance (started via experiments/lambda.sh run). Three lanes share the GPU:
#   lane 1: typed-decisions with seeds (customer-service probes, blocks and full fine-tune; probes on all 20)
#   lane 2: seed replication of the key Banking77 / CLINC150 configs, full fine-tune included
#   lane 3: exploration experiments (explore/*.py)
# Each step logs to results/tarski/run_<name>.out; lane progress goes to results/tarski/lane<N>.log.
cd "$(dirname "$0")/.." || exit 1
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# three lanes share 30 vCPUs: uncapped, each process spins ~12 OpenMP threads and the GPU idles
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=.venv/bin/python
R=results/tarski
CS="customer_service.action customer_service.category customer_service.churn_risk customer_service.needs_human customer_service.urgency"
mkdir -p $R

run() { lane=$1; name=$2; shift 2; echo "$(date +%H:%M:%S) start $name" >> $R/lane$lane.log
        "$@" > "$R/run_$name.out" 2>&1; echo "$(date +%H:%M:%S) end $name (exit $?)" >> $R/lane$lane.log; }

lane1() {
  for s in 0 1 2; do
    run 1 typed_cs_s$s $PY -m experiments.sweep --dataset typed-decisions --tasks $CS --probe-splits 6 11 16 22 \
      --block-splits 8 14 20 --depths 2 --block-epochs 5 --full --full-tasks $CS --seed $s --out $R/sweep_typed_cs_s$s.json
  done
  for s in 0 1 2; do
    run 1 typed_probes_s$s $PY -m experiments.sweep --dataset typed-decisions --probe-splits 6 11 16 22 --block-splits \
      --seed $s --out $R/sweep_typed_s$s.json
  done
  echo "$(date +%H:%M:%S) LANE1_DONE" >> $R/lane1.log
}

lane2() {
  for s in 0 1 2; do
    run 2 banking_s$s $PY -m experiments.sweep --dataset banking77 --probe-splits 8 22 --block-splits 4 8 14 18 \
      --depths 1 2 --full --seed $s --out $R/sweep_banking77_s$s.json
    run 2 clinc_s$s $PY -m experiments.sweep --dataset clinc150 --probe-splits 6 22 --block-splits 6 11 16 20 \
      --depths 2 --full --seed $s --out $R/sweep_clinc150_s$s.json
  done
  echo "$(date +%H:%M:%S) LANE2_DONE" >> $R/lane2.log
}

lane3() {
  run 3 kvquery_typed $PY explore/systems_kvquery.py --dataset typed-decisions --split 11 --out $R/explore_kvquery_typed.json
  run 3 coarsegrain $PY explore/farfield_coarsegrain.py --out $R/explore_coarsegrain.json
  run 3 merge_typed_cs $PY explore/merge_branches.py --dataset typed-decisions --tasks $CS --split 14 \
    --out $R/explore_merge_typed_cs.json
  run 3 merge_clinc $PY explore/merge_branches.py --dataset clinc150 --split 11 --out $R/explore_merge_clinc150.json
  run 3 syndrome $PY explore/farfield_syndrome.py --out $R/explore_syndrome.json
  run 3 oos_priorshift $PY -m explore.skeptic_oos_priorshift --out $R/explore_oos_priorshift.json
  run 3 passenger_typed $PY explore/geometry_passenger.py --datasets typed-decisions --cloze --out $R/explore_passenger_typed.json
  run 3 depthscan $PY explore/geometry_depthscan.py --out $R/explore_depthscan.json
  run 3 active_distillation $PY explore/learning_active_distillation.py --out $R/explore_active_distillation.json
  run 3 proper_score $PY explore/learning_proper_score_objectives.py --out $R/explore_proper_score_objectives.json
  run 3 threadinc $PY explore/systems_threadinc.py --dataset clinc150 --split 11 --thread-len 10 \
    --out $R/explore_threadinc_clinc150.json
  run 3 kvquery_banking $PY explore/systems_kvquery.py --dataset banking77 --split 8 --seeds 0 \
    --out $R/explore_kvquery_banking77.json
  run 3 passenger_clinc $PY explore/geometry_passenger.py --datasets clinc150 --out $R/explore_passenger_clinc.json
  run 3 receptive_field $PY -m explore.skeptic_receptive_field --out $R/explore_receptive_field.json
  run 3 globalsplit_banking $PY -m experiments.sweep --dataset banking77 --probe-splits --block-splits 15 16 17 18 19 20 \
    --depths 1 --out $R/explore_globalsplit_banking77.json
  echo "$(date +%H:%M:%S) LANE3_DONE" >> $R/lane3.log
}

lane1 & lane2 & lane3 &
wait
echo "$(date +%H:%M:%S) ALL_LANES_DONE" >> $R/lanes.log
