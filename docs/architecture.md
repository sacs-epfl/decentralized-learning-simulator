# Repository structure and workflow

`main` is the shared simulator and algorithm collection. `conflux-v2` develops
Conflux V2 on top of it. `codex/conflux-v2-validation` preserves the original
experiment history, source hashes, result tables and launch scripts. No branches
or experiment records are deleted by the separation.

## Code layout

| Path | Responsibility |
|---|---|
| `dasklearn/algorithms.py` | Lazy algorithm/settings registration shared by CLI and brokers. |
| `dasklearn/experiments.py`, `experiment_runner.py` | Configuration validation, sweep expansion, isolated runs and manifests. |
| `dasklearn/settings_factory.py` | Shared CLI-to-settings conversion for coordinator and local execution. |
| `dasklearn/simulation/<algorithm>/` | Protocol events, communication and algorithm settings. |
| `dasklearn/simulation/bandwidth_scheduler.py` | Shared bandwidth/latency accounting. |
| `dasklearn/tasks/` | DAG references, local execution, memory bounds and CUDA training backend. |
| `dasklearn/functions.py` | Compute functions executed by local workers or brokers. |
| `dasklearn/optimizer/` | Optimizer-specific numerical code. |
| `dasklearn/datasets/`, `models/` | Data partitions, evaluation splits and model definitions. |
| `scripts/` | Generic dataset preparation, benchmarks and reports. |
| `tests/` | Maintained mathematical, causal, numerical and integration checks. |
| `tests/legacy/` | Older fixtures awaiting migration to the current APIs. |
| `docs/` | Protocol contracts, performance behavior and workflow. |

## Adding an algorithm

Start with the [protocol integration guide](adding-protocols.md).

Implement its settings, simulation/client events and any required compute
functions. Use explicit `TaskRef` inputs with `compute` or `create_compute_task`
and use `send` / `send_control` for transport. Register numerical contracts with
`register_operation`. Register settings and simulation class names in `ALGORITHMS` so both
the coordinator and brokers agree on serialization. Add CLI-specific options in
`args.py`; comparison aliases belong in `simulation/peer_baselines.py` and the
comparison runner. Include mathematical/causal tests, optimizer-state ownership,
wire-byte accounting and unsupported-setting checks. Document deviations from
the source algorithm, especially optimizer changes, membership assumptions and
simulated compute costs.

## Simulator programming model

See [simulator programming model](simulator-design-improvements.md)
for the implemented compute, payload, protocol helper and runtime feedback APIs.
Static protocols build the full DAG before execution; adaptive protocols extend
it at feedback decisions. Workflow task format version 2 replaces older DAGs.

## Branch workflow

Land generic fixes, optimizations, new datasets and baselines on `main` first.
Merge `main` into `conflux-v2` to reuse them. Conflux V2-specific code, options,
tests and experiments stay on the research branch. Avoid merging the entire
research branch back into `main`; port only the shared changes.

The initial separation is a curated commit based on the current upstream main,
not a merge of the research history. Thus main's history does not import the
Conflux V2 or unfinished PushSum implementations. Upstream speech dependencies
and legacy model definitions are retained.

Experiment execution code on sacs006 remains frozen in its existing directories.
Historical manifests retain their original hashes; the reorganization does not
claim that a renamed/refactored runner has the same source hash. Use new output
directories for new runs. Raw DAGs, checkpoints and per-node files stay where the
original manifests record them.

## Separation checks

The maintained main suite passes 114 tests on macOS/CPU (six CUDA skips) and
120 on Linux/CUDA on sacs006. A ten-run synthetic smoke matrix executes Shatter,
Moshpit, Teleportation, DSpodFL and DeMo with uniform/heterogeneous capacities.
All snapshots include all eight nodes. CLI parsing and saved-settings restoration
are checked independently of protocol execution.

A syntax-tree comparison against the archived validation branch confirms that
101 existing runtime modules retain identical executable code, including the
baseline implementations and execution optimizations. The intentional existing
runtime differences are the broker's shared settings registry and restoration
of upstream main's legacy FEMNIST model. The new registry is tested for every
registered algorithm; saved runs retain their algorithm and settings.
