# Conflux V2 research

This branch adds Conflux V2 to the shared simulator on `main`. Shared execution
optimizations and literature baselines remain the same code on both branches.
Conflux V2-specific implementation lives in `dasklearn/simulation/conflux_v2/`;
its registration, CLI options and runner support are the small integration layer.

## Protocol

A reproducible sample trains from the current aggregate. A temporary coordinator
agrees on one contributor set, containing the first
`ceil(sample_size * success_fraction)` ready contributors. Every chunk reducer
uses that same set. Each contributor uploads a chunk once to its designated
reducer; the reducer averages contributors in a fixed order and sends the result
to each recipient in the next sample. Those recipients reconstruct the same
sample aggregate. Reducer assignment balances chunk load relative to public
bandwidth capacities and may use nodes outside the training sample.

With full participation within a sample (`success_fraction=1`),
`--immediate-uploads` starts each contributor's uploads as soon as local training
finishes. The contributor set is already known, so the readiness/plan exchange
can be skipped. Reducers still wait for every contributor. This optimization is
opt-in; partial-success rounds retain the coordinated path. `--relay-count` caps
potential reducers; the chunk count determines how many chunks can be placed.

Stable membership is required. Offline-node recovery, coordinator/reducer failure
and membership changes are not implemented. Partial-success selection can bias
learning when speed and client data correlate. Chunking and aggregation compute
are instantaneous in simulated time, consistent with the baseline conventions.

## Running

Small CPU validation:

```sh
python scripts/validate_algorithms.py --variants shatter-r2 conflux-v2 moshpit \
  --peers 8 --sample-size 4 --chunks 4 --relay-count 4 --immediate-uploads \
  --profiles heterogeneous --partitioners iid --seeds 0 \
  --duration 10 --test-period 2 --output output/v2-smoke-new
```

CIFAR-10 with the optimized 16-trainer/chunk/reducer configuration:

```sh
python scripts/run_cifar.py --output output/cifar-v2-new \
  --dataset-base-path /path/to/cifar-cache --variants conflux-v2 moshpit \
  --profiles uniform heterogeneous --partitioners iid dirichlet --seeds 0 1 2 \
  --peers 64 --sample-size 16 --chunks 16 --relay-count 16 --immediate-uploads \
  --duration 900 --test-period 150 --alpha .1 --learning-rate .01 \
  --momentum 0 --local-steps 5 --device cuda:0
```

The coordinator also supports `--algorithm conflux-v2`. Its chunk flag is
`--chunks-in-sample`; comparison runners use `--chunks`. CUDA residency/batching
opt-outs and reference fallbacks are inherited from main. Use fresh output paths:
source hashes change when code is reorganized even if protocol math does not.

## Evaluation and correctness

The primary readout is mean accuracy/loss across all nodes' stored completed
checkpoints at a fixed simulated time. It includes idle sampled nodes' older
checkpoints. The latest completed V2 sample aggregate is a separate quantity,
identified by the maximum recorded round; all recipients of that round must
have identical metrics. `latest-aggregate.csv` retains that readout and node
staleness. Selected trainers receive the current round input before training;
older stored node checkpoints do not imply stale-gradient training.

`tests/test_conflux_v2.py` contains the V2-specific numerical, compute-serialization,
streaming, reconstruction-cache and batched-execution checks separated from the
shared tests. `tests/test_conflux_v2_overlap.py` checks immediate uploads, reducer
placement, contributor ordering and partial-success fallback. Run the complete
maintained suite with `python -m pytest` on this branch.

## Research records

- [Baseline comparisons and upload/reducer ablations](experiments/peer-baselines.md).
- [DSpodFL comparison](experiments/dspodfl.md).
- [DeMo comparison](experiments/demo.md).
- [Original research log, including Tiny ImageNet and execution optimizations](experiments/research-log.md).
- Compact results, manifests, plots and audit scripts retain their original
  paths under `output/`. Raw DAGs and per-node files remain in the server paths
  recorded by those manifests.

The original archive branch is unchanged. Existing frozen execution directories
on sacs006 were not edited. Historical metrics still refer to their original
commits/source hashes; this branch layout does not relabel them as fresh results.
Generic changes belong on `main` first, then merge into this branch.

After separation, this branch passes 139 maintained tests on macOS/CPU (six
CUDA skips) and all 145 on Linux/CUDA on sacs006. A fresh three-protocol smoke
run executes Shatter, full-sample immediate-upload V2 and Moshpit, preserving
complete per-node snapshots. These are reorganization regression checks, not
new convergence experiments.
