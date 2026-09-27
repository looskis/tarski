#!/usr/bin/env bash
# DLM experiments on the Lambda GPU recorded in .lambda_instance_dlm.json (launched by readonce.lambda_cloud).
#   dlm/lambda_dlm.sh setup            # copy code, make a venv with cu126 torch + transformers, download the model
#   dlm/lambda_dlm.sh sync             # copy code and small results only
#   dlm/lambda_dlm.sh ssh CMD...       # run a command on the box (in ~/tarski, venv on PATH)
#   dlm/lambda_dlm.sh run NAME CMD...  # run CMD in the background on the box, log to results/dlm/NAME.out
#   dlm/lambda_dlm.sh pull             # copy results/dlm/*.jsonl|json|out back into results/dlm/h100/
# Copies dlm/, third_party/openjev, third_party/jevbench and the small results the experiments read.
# Never copies .env.local. The SSH private key path comes from LAMBDA_KEY.
set -e
cd "$(dirname "$0")/.."
STATE=${LAMBDA_STATE:-.lambda_instance_dlm.json}
KEY=${LAMBDA_KEY:?set LAMBDA_KEY to the private key path}
KNOWN=${LAMBDA_KNOWN_HOSTS:-$(dirname "$KEY")/lambda_known_hosts}
IP=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['ip'])" "$STATE")
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN" -o ServerAliveInterval=30 "ubuntu@$IP")
RSYNC=(/usr/bin/rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN")
ENV='cd ~/tarski && export PATH=$HOME/.local/bin:$HOME/tarski/.venv/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1'
sync_code() {
  "${SSH[@]}" "mkdir -p ~/tarski/third_party ~/tarski/results/dlm"
  "${RSYNC[@]}" --exclude __pycache__ dlm "ubuntu@${IP}:tarski/"
  "${RSYNC[@]}" --exclude __pycache__ --exclude .git third_party/openjev third_party/jevbench "ubuntu@${IP}:tarski/third_party/"
  "${RSYNC[@]}" results/dlm/anatomy_jevbench.jsonl results/dlm/census_*.json results/dlm/queries_*.safetensors results/dlm/queries_*.json "ubuntu@${IP}:tarski/results/dlm/"
}
case "$1" in
  setup)
    sync_code
    "${SSH[@]}" "$ENV && (command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null) &&
      export PATH=\$HOME/.local/bin:\$PATH && uv venv --python 3.12 .venv -q &&
      uv pip install -q --python .venv/bin/python torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126 &&
      uv pip install -q --python .venv/bin/python 'transformers==5.17.0' datasets accelerate safetensors numpy httpx huggingface_hub hf_transfer sentencepiece &&
      .venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))' &&
      (.venv/bin/hf download google/diffusiongemma-26B-A4B-it >/dev/null 2>&1 || .venv/bin/huggingface-cli download google/diffusiongemma-26B-A4B-it >/dev/null) &&
      echo 'model cached' && df -h ~ | tail -1"
    ;;
  sync) sync_code ;;
  ssh) shift; "${SSH[@]}" "$ENV && $*" ;;
  run) shift; NAME=$1; shift; "${SSH[@]}" "$ENV && nohup sh -c '$*' > results/dlm/$NAME.out 2>&1 &" ; echo "started $NAME" ;;
  pull) mkdir -p results/dlm/h100; "${RSYNC[@]}" "ubuntu@${IP}:tarski/results/dlm/*.jsonl" "ubuntu@${IP}:tarski/results/dlm/*.json" "ubuntu@${IP}:tarski/results/dlm/*.out" results/dlm/h100/ 2>/dev/null || true; ls results/dlm/h100/ ;;
  *) sed -n 2,8p "$0" ;;
esac
