#!/bin/sh
# Simulated carves at three keep levels, experts chosen on the other dataset's census. Waits for any
# running dlm model process to exit first (one model process at a time on 32 GB).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
F='Warning|warn\(|Fetching'
while pgrep -f "dlm/(order|anatomy|interference|census|train_query|apply_query)\.py" >/dev/null; do sleep 30; done
for keep in 0.99 0.95 0.90; do
  echo "== carve keep $keep: typed-test 200 (experts from JevBench census)"
  $PY dlm/carve.py --census results/dlm/census_jevbench_en.json --keep $keep --dataset typed-test --limit 200 \
      --out results/dlm/carve_typed_test_$keep.jsonl 2>&1 | grep -Ev "$F"
  echo "== carve keep $keep: JevBench hard 111 (experts from typed-test census)"
  $PY dlm/carve.py --census results/dlm/census_typed_test.json --keep $keep --dataset jevbench --tiers hard --limit 200 \
      --out results/dlm/carve_jevbench_hard_$keep.jsonl 2>&1 | grep -Ev "$F"
done
echo "== carve done"
