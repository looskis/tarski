#!/bin/sh
# Everything that needs the GPU after the anatomy reads exist, in one sequence (one model process at a time).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
F='Warning|warn\(|Fetching'
echo "== train: learned queries, label-free (meanK), per type"
$PY dlm/train_query.py --train results/dlm/anatomy_typed_train.jsonl --test results/dlm/anatomy_typed_test.jsonl \
    --target meanK --per type --epochs 3 --name meanK_type 2>&1 | grep -Ev "$F"
echo "== train: gold-target control, per type"
$PY dlm/train_query.py --train results/dlm/anatomy_typed_train.jsonl --test results/dlm/anatomy_typed_test.jsonl \
    --target gold --per type --epochs 3 --name gold_type 2>&1 | grep -Ev "$F"
echo "== transfer: queries on JevBench public"
$PY dlm/apply_query.py --queries results/dlm/queries_meanK_type.safetensors --dataset jevbench \
    --anatomy results/dlm/anatomy_jevbench.jsonl 2>&1 | grep -Ev "$F"
$PY dlm/apply_query.py --queries results/dlm/queries_gold_type.safetensors --dataset jevbench \
    --anatomy results/dlm/anatomy_jevbench.meanK_type.jsonl --out results/dlm/anatomy_jevbench.queries.jsonl 2>&1 | grep -Ev "$F"
echo "== interference and slot isolation on typed-decisions test"
$PY dlm/interference.py --limit 200 --out results/dlm/interference_typed_test.jsonl 2>&1 | grep -Ev "$F"
echo "== expert census"
$PY dlm/census.py --dataset jevbench --limit 120 --tiers original easy --out results/dlm/census_jevbench_en.json 2>&1 | grep -Ev "$F"
$PY dlm/census.py --dataset typed-test --limit 100 --out results/dlm/census_typed_test.json 2>&1 | grep -Ev "$F"
echo "== phase B done"
