#!/usr/bin/env bash
# Run tarski experiments on the Lambda instance recorded in $LAMBDA_STATE (default .lambda_instance.json).
#   experiments/lambda.sh setup            # copy code, install deps (cu126 torch), warm the model/dataset caches
#   experiments/lambda.sh sync             # copy code only
#   experiments/lambda.sh run NAME CMD...  # run CMD in the background on the instance, log to results/tarski/NAME.out
#   experiments/lambda.sh ssh CMD...       # run a command
#   experiments/lambda.sh pull             # copy results/tarski back into results/tarski/lambda/
#   experiments/lambda.sh submit NAME CMD  # add a job to the instance's queue (experiments/jobrunner.sh runs it)
#   experiments/lambda.sh jobs             # show the queue and its log
# Copies tarski/, experiments/, explore/, tests/, pyproject.toml, uv.lock and the local sweep results
# needed for resuming. Never copies .env.local or other local files.
set -euo pipefail
cd "$(dirname "$0")/.."
KEY=${LAMBDA_SSH_KEY:?set LAMBDA_SSH_KEY to the private key registered with Lambda}
KNOWN=${LAMBDA_KNOWN_HOSTS:-$(dirname "$KEY")/lambda_known_hosts}
STATE=${LAMBDA_STATE:-.lambda_instance.json}
IP=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['ip'])" "$STATE")
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN" -o ServerAliveInterval=30 "ubuntu@$IP")
RSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN"
ENV='cd tarski && export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error'

sync_code() {
  rsync -az -e "$RSH" --exclude __pycache__ --exclude '*.pyc' tarski experiments explore tests readonce pyproject.toml uv.lock \
    "ubuntu@$IP:tarski/"
}

case "$1" in
  setup)
    sync_code
    "${SSH[@]}" "$ENV && mkdir -p results/tarski && (command -v uv || python3 -m pip install --user -q uv) >/dev/null &&
      env -u UV_NO_SYNC uv sync -q --python 3.12 &&
      uv pip install -q --python .venv/bin/python --reinstall 'torch==${TORCH_VERSION:-2.14.0}' \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA:-cu126} &&
      .venv/bin/python -c \"
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
from tarski import data; from tarski.trunk import Trunk; Trunk()
for n in data.BENCHMARKS: print(data.load(n).summary())
import laya; laya.load('convaiinnovations/laya'); laya.load('convaiinnovations/laya', subfolder='typed-decisions')
print('caches warm')\""
    ;;
  sync) sync_code ;;
  run)
    name=$2; shift 2
    "${SSH[@]}" "$ENV && { nohup bash -c '$*' > results/tarski/$name.out 2>&1 < /dev/null & } && echo started $name"
    ;;
  ssh) shift; "${SSH[@]}" "$ENV && $*" ;;
  pull)
    dest=results/tarski/lambda; [ "$STATE" = ".lambda_instance_fast.json" ] && dest=results/tarski/lambda_a100
    mkdir -p "$dest"
    rsync -az -e "$RSH" "ubuntu@${IP}:tarski/results/tarski/" "$dest/"
    ;;
  submit)
    name=$2; shift 2
    printf '%s\n' "$*" | "${SSH[@]}" "$ENV && mkdir -p jobs/pending && n=\$(date +%s%N) && cat > jobs/pending/\${n}_$name.sh && echo queued $name"
    ;;
  jobs)
    "${SSH[@]}" "$ENV && echo pending: \$(ls jobs/pending 2>/dev/null | sed 's/^[0-9]*_//' | tr '\n' ' '); echo running: \$(ls jobs/running 2>/dev/null | sed 's/^[0-9]*_//' | tr '\n' ' '); tail -${2:-8} jobs/log 2>/dev/null"
    ;;
  *) echo "unknown command $1"; exit 2 ;;
esac
