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
- [QG momentum comparison: fresh one-step CIFAR-10 matrix](experiments/qgm.md).
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

## Current main integration and expanded study

The 31 October 2026 integration rebases the research delta onto main `bda202a`
and migrates V2 to explicit task references and main's compute/payload APIs.
The original full history is retained at `codex/conflux-v2-before-rebase`;
historical records keep their original provenance. V2 settings work with the
shared YAML/JSON runner, including independent controlled capacity profiles.
See [the expanded study](experiments/convergence-20261031.md) for the four
workloads, six algorithms, validation-only tuning, and paired confirmation plan.

## Reducer-side server optimizers (experimental)

The `codex/conflux-v2-fedopt` branch adds `server_optimizer: fedavgm` and
`server_optimizer: fedadam`. The default remains the original FedAvg protocol.
Each optimizer shard retains the current base weights and its own moment history;
trainers continue to use ordinary local SGD. Server pseudo-gradients are the base
minus the common, uniformly averaged client model. FedAvgM uses unnormalized SGD
momentum. FedAdam uses EMA first/second moments, `v0 = server_tau**2`, no bias
correction, and denominator `sqrt(v) + server_tau`, following FedOpt Algorithm 2.
Non-parameter buffers are averaged rather than optimized; LoRA is unsupported.

`reducer_epoch_rounds: 0` fixes chunk ownership, `1` preserves per-round rotation,
and larger values retain a layout for that many logical rounds. At an ownership
change the previous owner sends the base chunk, moments and logical step to the
new owner through the normal contended network. The new owner waits for that
state and all contributors before updating. State is never broadcast to trainers.

`state_bits: 8` quantizes moments only on handoff, using independent blocks of
`state_block_size` elements. First moments use signed symmetric int8 and second
moments nonnegative uint8, each with FP32 block scales. Owners compute in FP32;
base weights stay FP32. Wire size includes those arrays, scales, an eight-byte
step counter, and the shared API's normal header/metadata charge. Every recipient
still reconstructs the same model even when moments are approximate.

`server_step_ms` charges CPU optimizer time per chunk on a serialized optimizer
resource; `state_pack_ms` charges quantization time per migrated chunk. Defaults
are zero, retaining the simulator's original instantaneous aggregation convention.
The experimental comparison explicitly charges 0.1 ms and 0.05 ms respectively;
these are controlled costs, not measurements of an edge device.

The regular version-1 configuration in
`experiments/conflux-v2/fedopt/tuning.json` describes a 64-node, 16-trainer,
128-chunk/16-reducer CIFAR-10/GNLeNet study, non-IID alpha 0.1, joint network/compute
heterogeneity and 900 simulated seconds. It gives server SGD (FedAvgM beta1=0),
server momentum and FedAdam equal twelve-candidate client/server-rate searches on
validation seed 10011. Rates are selected by validation accuracy area, then frozen
for seeds 0/1/2. Confirmation compares fixed owners, FP32/8-bit rotating handoffs,
eight-round ownership epochs, original fixed/rotating FedAvg and Moshpit. Full
final test scores are never used to select rates. Network and CPU costs are charged
in protocol time; GPU execution wall time is separate.

```sh
python scripts/benchmark_conflux_fedopt.py --stage plan --output /path/to/fedopt-results
python scripts/benchmark_conflux_fedopt.py --output /path/to/fedopt-results --gpus 0 1
# Repeat an identical command to resume; changed source/settings are rejected.
python scripts/benchmark_conflux_fedopt.py --stage status --output /path/to/fedopt-results
```

Tests compare every chunked round to independently calculated full-vector FedOpt,
check state/round ownership, exact byte savings of actual quantized packets, common
recipient models, partial participation, charged compute, and validation-only rate
selection. Scheduled handoff payload totals are separate from actual transmitted
bytes: a time-limited run can stop while a transfer is still pending or in flight.
