# Decentralized Learning Simulator — Conflux V2 research

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

This branch extends `main` with Conflux V2. General improvements and baselines
belong on `main`; merge `main` into this branch as it evolves.
See [Conflux V2 protocol and experiments](docs/conflux-v2.md).
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

| Algorithm | CLI name / runner alias | Paper or original source | Implementation notes |
|---|---|---|---|
| FedAvg / centralized FL | `fl` | [Communication-Efficient Learning of Deep Networks from Decentralized Data — AISTATS 2017](https://proceedings.mlr.press/v54/mcmahan17a.html) | Legacy centralized averaging baseline. |
| D-PSGD | `dpsgd` | [Can Decentralized Algorithms Outperform Centralized Algorithms? — NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/f75526659f31040afeb61cb7133e4e6d-Abstract.html) | Legacy fixed-topology implementation. |
| AD-PSGD (A-DPSGD) | `adpsgd` | [Asynchronous Decentralized Parallel Stochastic Gradient Descent — ICML 2018](https://proceedings.mlr.press/v80/lian18a.html) | Legacy asynchronous implementation. |
| Epidemic Learning | `epidemic` | [Epidemic Learning: Boosting Decentralized Learning with Randomized Communication — NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html) | Legacy changing-topology implementation. |
| Gossip Learning | `gossip` | [Gossip Learning as a Decentralized Alternative to Federated Learning — DAIS 2019](https://www.inf.u-szeged.hu/~ihegedus/publ/dais19a.pdf) | Legacy periodic gossip and model-age weighting. |
| Super Gossip Learning | `super-gossip` | [Budaj's EPFL thesis, §4.5 — 2024](https://www.epfl.ch/labs/sacs/wp-content/uploads/2024/07/2024-spring-lubor-1.pdf#page=23); [original project](https://github.com/sacs-epfl/async-dl) | Internal research variant; no separate peer-reviewed paper identified. |
| Heterogeneous Gossip Learning (HGL) | `lubor` | [Budaj's EPFL thesis, Chapter 6 — 2024](https://www.epfl.ch/labs/sacs/wp-content/uploads/2024/07/2024-spring-lubor-1.pdf#page=39) | Thesis prototype; retains its historical CLI name. |
| Shatter | `shatter`; `shatter-r2`, `shatter-r4` in runners | [Noiseless Privacy-Preserving Decentralized Learning — PoPETs 2025](https://petsymposium.org/popets/2025/popets-2025-0043.php) | Real-node chunked-exchange specialization. |
| Moshpit SGD | `moshpit` | [Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices — NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/97275a23ca44226c9964043c8462be96-Abstract.html) | Stable-membership group collectives. |
| Teleportation | `teleportation`; `teleportation-ring`, `teleportation-hypercube` in runners | [Scalable Decentralized Learning with Teleportation — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b3732a13897c4cea145c3bdece80de64-Abstract-Conference.html) | Algorithm 1 specialization; hypercube is not the paper's Base-2 graph. |
| D-Local-SGD | `d-local-sgd-ring` in runners | [A Unified Theory of Decentralized SGD with Changing Topology and Local Updates — ICML 2020](https://proceedings.mlr.press/v119/koloskova20a.html) | Fixed full-participation ring with configurable local updates. |
| DSpodFL | `dspodfl` | [Decentralized Sporadic Federated Learning: A Unified Algorithmic Framework with Convergence Guarantees — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html) | One-step rule and explicit local-SGD block extension. |
| DeMo | `demo` | [DeMo: Decoupled Momentum Optimization — ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17790fc9b4828206f99a0dea35da3657-Abstract-Conference.html) | Reference Signum optimizer; ring all-gather and CNN adaptation. |
| Conflux (original) | `conflux` | Unpublished prototype; related sampling work: [Practical Federated Learning without a Server — EuroMLSys 2025](https://euromlsys.eu/pdf/euromlsys25-10.pdf). | Chunked sampling prototype; not an implementation of Plexus. |
| Conflux V2 | `conflux-v2` | [Research protocol documentation](docs/conflux-v2.md); unpublished prototype. | Research branch only. |

[Baseline contracts and sources](docs/peer-baselines.md),
[DSpodFL semantics](docs/dspodfl.md), [DeMo semantics](docs/demo.md).
Paper links identify the source algorithms, not a claim of exact paper reproduction.
Stable-membership restrictions, optimizer assumptions and adaptations are
explicit in those documents. Legacy entries have not all been re-audited. PushSum remains on its existing experimental branch.

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
