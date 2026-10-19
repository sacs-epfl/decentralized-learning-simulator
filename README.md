# Decentralized Learning Simulator

Simulate distributed learning protocols and execute their PyTorch training DAGs.
The discrete-event simulator models local compute, bandwidth contention, latency
and message traffic. A local executor or distributed brokers/workers then perform
the actual training and evaluation. **Simulated protocol time and execution wall
time are separate measurements.**

## Branches

- **`main`**: shared simulator, datasets, performance improvements and algorithm implementations.
- **[`conflux-v2`](https://github.com/sacs-epfl/decentralized-learning-simulator/tree/conflux-v2)**:
  Conflux V2 development, comparison experiments and their recorded results.
- **`codex/conflux-v2-validation`**: unchanged archive of the original experiments
  and execution-source provenance.

New general improvements and baselines belong on `main`; merge `main` into the
research branch as it evolves. Conflux V2 is not a dependency of this branch.
See [repository structure and development workflow](docs/architecture.md).

## Setup

```sh
conda env create -f environment.yml
conda activate decentralized-learning
python -m pytest -q
```

The environment specifies Python 3.12 and the modern Torch/TorchVision versions
used for validation. CPU execution works without CUDA. For a GPU host, install a
compatible CUDA Torch build in the Conda environment. `requirements.txt` also
supports an existing environment. Hugging Face-backed datasets require
`flwr-datasets`; speech transforms use `librosa`.

## Algorithms

| Implementation | Coordinator `--algorithm` | Notes |
|---|---|---|
| Shatter | `shatter` | Chunked model exchange; degree via `--r`, chunks via `--k`. |
| Moshpit SGD | `moshpit` | Group collectives and bandwidth-balanced chunk ownership. |
| Teleportation | `teleportation` | Sampled model tokens; ring/hypercube mixing. |
| D-Local-SGD | comparison alias `d-local-sgd-ring` | Fixed full-participation ring specialization. |
| DSpodFL | `dspodfl` | Sporadic local work and symmetric old-model exchange. |
| DeMo | `demo` | Persistent residuals, sparse DCT packets and ring all-gather. |
| Original protocols | `fl`, `dpsgd`, `epidemic`, `gossip`, `super-gossip`, `adpsgd`, `lubor`, `conflux` | Retained implementations; validation coverage varies. |

[Baseline contracts and sources](docs/peer-baselines.md),
[DSpodFL semantics](docs/dspodfl.md), [DeMo semantics](docs/demo.md).
Stable-membership restrictions, optimizer assumptions and adaptations are
explicit in those documents. PushSum remains on its existing experimental branch.

## Run a small comparison

This runs real protocol-generated DAGs on CPU, with synthetic data and no downloads:

```sh
python scripts/validate_algorithms.py --variants shatter-r2 moshpit \
  --peers 8 --sample-size 4 --profiles uniform heterogeneous \
  --partitioners iid dirichlet --seeds 0 --duration 20 --test-period 5 \
  --output output/smoke
```

The runner also supports bundled handwritten digits (`--dataset digits`).
`validate_conflux.py` remains a compatibility entry point for older commands.
CIFAR-10 and Tiny ImageNet use `prepare_cifar.py` / `prepare_tiny.py` followed by
`run_cifar.py` / `run_tiny.py`; use `--help` for their dataset paths and settings.
Each run records parameters, versions, a source hash and per-run outputs.

At fixed simulated times, accuracy is the mean of all nodes' latest completed
local checkpoints, including nodes that have not recently updated. No uncharged
global parameter average is constructed. Unreached accuracy targets are censored.
Controlled heterogeneous profiles vary compute and bandwidth independently;
these profiles are distinct from FedScale traces (`data/fedscale_traces`).

## Coordinator and brokers

Generate a DAG without executing it:

```sh
python main.py --algorithm moshpit --dataset synthetic --model mlp \
  --peers 8 --group-size 4 --momentum 0 --rounds 3 --dry-run
```

Without `--dry-run`, the coordinator waits for its configured brokers. Start a
broker with `python broker.py --coordinator tcp://HOST:5555 --port 5556 --workers 2`.
All registered settings types are supported by the shared broker registry.
Use `--from-dir PATH` to load a saved workflow/settings pair.

## Performance and validation

Shared improvements include direct immutable model-state copying, bounded DAG
execution and reconstruction caches, cached dataset tensors, reproducible training,
GPU workspaces and guarded batching of independent SGD tasks. These optimizations
preserve algorithm updates; CUDA equivalence tests compare complete model states.

The local execution runners enable supported CUDA optimizations automatically:
`--no-gpu-residency` and `--no-training-batching` opt out. CPU and unsupported
workloads use the reference path. [Performance details](docs/performance.md).

Default pytest discovery covers the maintained tests. Older protocol fixtures
using removed settings/reference APIs are retained in `tests/legacy` for migration
and excluded from default discovery. The old Task tests have been migrated to the
current reference/index API. Historical results remain on the research/archive
branches; new experiments should use fresh output directories.
