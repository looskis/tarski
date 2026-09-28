#!/bin/bash
# A tiny job queue for a GPU box. Jobs are shell snippets in jobs/pending/<NN>_<name>.sh; each lane claims
# the oldest one (atomic mv), runs it, logs to results/tarski/run_<name>.out and moves it to jobs/done/.
# Usage (on the instance): bash experiments/jobrunner.sh <lanes>      (submit with experiments/lambda.sh submit)
cd "$(dirname "$0")/.." || exit 1
LANES=${1:-2}
mkdir -p jobs/pending jobs/running jobs/done results/tarski
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( $(nproc) / (LANES + 1) )) MKL_NUM_THREADS=$(( $(nproc) / (LANES + 1) ))
lane() {
  while true; do
    [ -f jobs/STOP ] && break
    job=$(ls jobs/pending 2>/dev/null | sort | head -1)
    if [ -z "$job" ]; then sleep 15; continue; fi
    mv "jobs/pending/$job" "jobs/running/$job" 2>/dev/null || continue     # another lane took it
    name=${job%.sh}; name=${name#*_}
    echo "$(date +%H:%M:%S) lane$1 start $name" >> jobs/log
    bash "jobs/running/$job" > "results/tarski/run_$name.out" 2>&1
    echo "$(date +%H:%M:%S) lane$1 end $name (exit $?)" >> jobs/log
    mv "jobs/running/$job" jobs/done/
  done
}
for i in $(seq 1 "$LANES"); do lane "$i" & done
wait
