#!/bin/sh
# Second batch, after the given PIDs exit: initialisation ablation and the auto-split evaluation.
cd "$(dirname "$0")/.." || exit 1
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 10; done
done
PY=.venv/bin/python
R=results/tarski
$PY -m experiments.ablation_init --out $R/ablation_init.json > $R/run_ablation_init.out 2>&1
$PY -m experiments.autosplit_eval --dataset banking77 --blocks --out $R/autosplit_banking77.json > $R/run_autosplit_banking77.out 2>&1
$PY -m experiments.autosplit_eval --dataset clinc150 --blocks --out $R/autosplit_clinc150.json > $R/run_autosplit_clinc150.out 2>&1
$PY -m experiments.autosplit_eval --dataset typed-decisions --out $R/autosplit_typed.json > $R/run_autosplit_typed.out 2>&1
echo QUEUE2_DONE
