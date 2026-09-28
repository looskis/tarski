#!/bin/sh
# Third batch, after the given PIDs exit:
#   1. typed-decisions reruns with the fixed early stopping (tarski/train.py), plus a same-architecture
#      full fine-tune reference for the five customer-service decisions
#   2. exploration experiments (explore/*.py), highest priority first
#   3. a clean CPU latency run (nothing else running)
# A failing step logs and the queue moves on. Extra lines can be appended to experiments/queue3_extra.sh.
cd "$(dirname "$0")/.." || exit 1
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 10; done
done
PY=.venv/bin/python
R=results/tarski
CS="customer_service.action customer_service.category customer_service.churn_risk customer_service.needs_human customer_service.urgency"

run() { name=$1; shift; echo "$(date +%H:%M:%S) start $name"; "$@" > "$R/run_$name.out" 2>&1; echo "$(date +%H:%M:%S) end $name (exit $?)"; }

run typed_probes $PY -m experiments.sweep --dataset typed-decisions --probe-splits 6 11 16 22 --block-splits \
  --out $R/sweep_typed.json
run typed_cs $PY -m experiments.sweep --dataset typed-decisions --tasks $CS --probe-splits --block-splits 8 14 20 \
  --depths 2 --block-epochs 5 --full --full-tasks $CS --out $R/sweep_typed_cs.json

run kvquery_typed $PY explore/systems_kvquery.py --dataset typed-decisions --split 11 --out $R/explore_kvquery_typed.json
run coarsegrain $PY explore/farfield_coarsegrain.py --out $R/explore_coarsegrain.json
run syndrome $PY explore/farfield_syndrome.py --out $R/explore_syndrome.json
run oos_priorshift $PY -m explore.skeptic_oos_priorshift --out $R/explore_oos_priorshift.json
run merge_clinc $PY explore/merge_branches.py --dataset clinc150 --split 11 --out $R/explore_merge_clinc150.json
run merge_typed_cs $PY explore/merge_branches.py --dataset typed-decisions --tasks $CS --split 14 \
  --out $R/explore_merge_typed_cs.json
run threadinc $PY explore/systems_threadinc.py --dataset clinc150 --split 11 --thread-len 10 \
  --out $R/explore_threadinc_clinc150.json
run kvquery_banking $PY explore/systems_kvquery.py --dataset banking77 --split 8 --seeds 0 \
  --out $R/explore_kvquery_banking77.json
run active_distillation $PY explore/learning_active_distillation.py --out $R/explore_active_distillation.json
run proper_score $PY explore/learning_proper_score_objectives.py --out $R/explore_proper_score_objectives.json
run receptive_field $PY -m explore.skeptic_receptive_field --out $R/explore_receptive_field.json
[ -f experiments/queue3_extra.sh ] && . experiments/queue3_extra.sh

run latency_cpu_clean $PY -m experiments.latency --device cpu --reps 20 --out $R/latency_cpu_clean.json
echo QUEUE3_DONE
