#!/bin/bash
source /mnt/data/mdevos/conflux-chunk-reducer-20261102/env-cached.sh
python scripts/benchmark_conflux_v2_sweep.py \
  --configs experiments/conflux-v2/chunk-reducer-64/v2.json experiments/conflux-v2/chunk-reducer-64/moshpit.json \
  --output "$STUDY_ROOT/results-cached" --gpus 0 1 0 1 0 1
TASK_SWEEP_EXIT=$?
printf '%s\n' "$TASK_SWEEP_EXIT" > "$STUDY_ROOT/exit-code-cached"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$STUDY_ROOT/finished-at-cached"
exit "$TASK_SWEEP_EXIT"
