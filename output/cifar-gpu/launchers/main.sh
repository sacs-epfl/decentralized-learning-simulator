#!/bin/bash
set -euo pipefail
source /mnt/data/mdevos/conflux/env.sh
for attempt in $(seq 1 40); do
  completed=$(find /mnt/data/mdevos/conflux/results -path '*/pilot64-*/COMPLETE' | wc -l)
  if [ "$completed" -eq 6 ]; then
    python scripts/dispatch_cifar.py --root /mnt/data/mdevos/conflux --peers 128 --duration 900 --test-period 90
    python scripts/report_cifar.py --input /mnt/data/mdevos/conflux/results/main128
    exit 0
  fi
  sleep 15
done
printf 'Pilot did not complete; main sweep was not started.\n' >&2
exit 1
