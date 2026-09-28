# Sourced by experiments/queue3.sh before the clean CPU latency run (uses its run(), $PY, $R).
run depthscan $PY explore/geometry_depthscan.py --out $R/explore_depthscan.json
run passenger_typed $PY explore/geometry_passenger.py --datasets typed-decisions --cloze --out $R/explore_passenger_typed.json
run passenger_clinc $PY explore/geometry_passenger.py --datasets clinc150 --out $R/explore_passenger_clinc.json
run globalsplit_banking $PY -m experiments.sweep --dataset banking77 --probe-splits --block-splits 15 16 17 18 19 20 --depths 1 --out $R/explore_globalsplit_banking77.json
# typed-decisions re-runs with the step-based budget (min 300 optimiser steps); full fine-tune for the
# remaining three customer-service tasks is left for a GPU with more memory
run typed_probes_steps $PY -m experiments.sweep --dataset typed-decisions --probe-splits 6 11 16 22 --block-splits \
  --out $R/sweep_typed_steps.json
run typed_cs_steps $PY -m experiments.sweep --dataset typed-decisions --tasks $CS --probe-splits --block-splits 8 14 20 \
  --depths 2 --block-epochs 5 --out $R/sweep_typed_cs_steps.json
