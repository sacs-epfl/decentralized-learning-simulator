# Expanded study launched on sacs006

The full study is running detached on **both GPUs** in `conflux-gpu`.
All new server files are under `/mnt/data/mdevos/conflux-v2-20261031`.
The frozen numerical source is commit `8b54232` in `code-study`, with SHA-256
`6e70c5f85f00492896b6858daa2219d977fd8fa9023bf390da0fc0bdef8b2b43`.
Original research history remains in `codex/conflux-v2-before-rebase`.
The remote Git branches have not been rewritten.

The launch snapshot recorded **four successful tuning runs, two running, and
zero failures**. This is a launch-time snapshot; query the server for current
progress. No confirmatory conclusion is available yet. There are **96 tuning
runs and 576 confirmation runs**, with an estimated total cost of roughly
8–15 hours based on short pilots. The script selects learning rates on validation
only, then executes all paired confirmation runs and generates the comparison.
No recurring monitor or automatic chat follow-up was scheduled.

The full Linux/CUDA suite passed **422 tests**. Final configuration, capacity,
heldout and scheduler checks passed **53 tests**; the final frozen driver's
end-to-end and resume check also passed. All eight short GPU pilots completed.
Passing logs and compact pilot metrics/configurations are included here;
per-run pilot records are stored in `pilot/records` to distinguish them from
server-only raw run directories.

## Matrix

| Workload | Model parameters | Pinned payload bytes | Simulated horizon |
|---|---:|---:|---:|
| Digits / MLP | 2,410 | 11,034 | 60 s |
| CIFAR-10 / GNLeNet | 85,354 | 346,251 | 600 s |
| Tiny ImageNet / MobileNet V3 Large | 4,458,232 | 18,040,763 | 1,800 s |
| Tiny ImageNet / ResNet18 | 11,279,112 | 45,195,718 | 1,800 s |

Six methods: Conflux V2, Shatter r=2, D-PSGD, Moshpit, Teleportation ring,
and QG momentum. Each workload covers IID / Dirichlet alpha 0.1 and uniform /
network-only / compute-only / joint heterogeneity, with paired seeds 0, 1, 2.
See [the complete design](../../docs/experiments/convergence-20261031.md) for
capacity assumptions, tuning, baseline settings, time-to-accuracy censoring,
heldout evaluation, and the distinction between protocol time and GPU wall time.

## Manual follow-up

```sh
ssh mdevos@sacs006
source /mnt/data/mdevos/conflux-v2-20261031/conflux-v2-env.sh
python scripts/benchmark_convergence.py --stage status \
  --data ../datasets --output ../results/full
# The launch PID is in ../full.pid. Logs are in ../full.log.
# Per-run DAG progress is in results/full/runs/NAME/data/*/execution-progress.json.

# Generate a report from completed runs at any time:
python scripts/benchmark_convergence.py --stage report \
  --data ../datasets --output ../results/full

# After an interruption, resume the identical frozen study:
python scripts/benchmark_convergence.py --study experiments/conflux-v2/study.json \
  --data ../datasets --output ../results/full --gpus 0 1
```

The driver holds an exclusive output lock and refuses a concurrent second run.
Resume rejects changed source/settings, skips completed runs, and archives
incomplete attempts. Keep `code-study` frozen while the study is active.

The final artifacts will be under `results/full`: `REPORT.md`, `summary.csv`,
`curves.csv`, `time-to-accuracy.csv`, `paired-comparisons.csv`,
`speedup-summary.csv`, and four workload plots. `COMPLETE` is written only when
all requested runs succeed. The detached launcher records `exit-code` and
`finished-at` at the server study root. Raw DAGs and per-node metrics remain
on the server; compact artifacts can be copied back after completion.
