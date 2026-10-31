#!/bin/bash
source /mnt/data/mdevos/conflux-v2-20261031/conflux-v2-env.sh
python - <<'PY'
import json
from pathlib import Path
from dasklearn.experiment_runner import provenance
root=Path.cwd()
export=json.loads((root/'source-export.json').read_text())
assert provenance(root)['source_sha256']==export['source_sha256'],'Source export mismatch'
PY
TASK_VERIFY_EXIT=$?
if [ "$TASK_VERIFY_EXIT" -ne 0 ]; then
  printf '%s\n' "$TASK_VERIFY_EXIT" > "$STUDY_ROOT/exit-code"
  exit "$TASK_VERIFY_EXIT"
fi
python scripts/benchmark_convergence.py --study experiments/conflux-v2/study.json \
  --data "$STUDY_ROOT/datasets" --output "$STUDY_ROOT/results/full" --gpus 0 1
TASK_STUDY_EXIT=$?
python scripts/benchmark_convergence.py --stage report \
  --data "$STUDY_ROOT/datasets" --output "$STUDY_ROOT/results/full"
printf '%s\n' "$TASK_STUDY_EXIT" > "$STUDY_ROOT/exit-code"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$STUDY_ROOT/finished-at"
exit "$TASK_STUDY_EXIT"
