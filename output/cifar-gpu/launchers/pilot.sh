#!/bin/bash
set -euo pipefail
source /mnt/data/mdevos/conflux/env.sh
profile=$1
for lr in .002 .01 .03; do
  python scripts/run_cifar.py --dataset-base-path /mnt/data/mdevos/conflux/datasets --output /mnt/data/mdevos/conflux/results/pilot64-${profile}-lr${lr} --profiles "$profile" --variants shatter-r2 shatter-r4 conflux-v2 --partitioners dirichlet --seeds 100 --peers 64 --sample-size 16 --chunks 8 --duration 300 --test-period 60 --learning-rate "$lr" --evaluation-split validation
 done
