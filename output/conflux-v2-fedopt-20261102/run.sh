#!/bin/bash
source /mnt/data/mdevos/conflux-fedopt-20261102/env.sh
python scripts/benchmark_conflux_fedopt.py --output "$STUDY_ROOT/results" --gpus 0 1 0 1 0 1
TASK_STUDY_EXIT=$?
printf '%s\n' "$TASK_STUDY_EXIT" > "$STUDY_ROOT/exit-code"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$STUDY_ROOT/finished-at"
exit "$TASK_STUDY_EXIT"
