# Decentralized Learning Simulator

Simulate distributed learning protocols and execute their PyTorch training DAGs.
The discrete-event simulator models local compute, bandwidth contention, latency
and message traffic. A local executor or distributed brokers/workers then perform
the actual training and evaluation. **Simulated protocol time and execution wall
time are separate measurements.**

Protocols use explicit task references and shared compute and payload APIs.
Accuracy-dependent decisions can pause simulation for numerical feedback and
extend the DAG. See the [programming model](docs/simulator-design-improvements.md)
and run `python scripts/run_feedback.py --output output/feedback-example` for a
working example. Existing algorithms and runners use the new API; regenerate
saved workflows created before task format version 2.
For implementation steps and a minimal ring protocol, see
[adding a protocol](docs/adding-protocols.md).

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
supports an existing environment. Both dependency files include Hugging Face
`datasets` and Flower `flwr-datasets`; speech transforms use `librosa`.

## LoRA fine-tuning

LoRA supports pretrained Hugging Face text and vision transformers with a shared frozen base per
worker and adapter-only client state. Install `requirements-lora.txt` inside your
conda environment, then run `scripts/run_lora.py` for local execution without
brokers. See [LoRA setup, CLI examples, memory model and validation](docs/lora.md).

## Algorithms

Validation evidence uses the same independent categories throughout:
**Numerical**, **Protocol**, **Reference**, **Integration**, and **Legacy**.
See the [definitions and scope](docs/baseline-validation.md#meaning-of-the-readme-labels-and-remaining-work).
Each row links to its evidence and limitations; categories describe recorded
checks rather than a ranking. An omitted category is not claimed for that row.

| Algorithm | CLI name / runner alias | Paper or original source | Implementation notes | Validation evidence |
|---|---|---|---|---|
| FedAvg / centralized FL | `fl` | [Communication-Efficient Learning of Deep Networks from Decentralized Data — AISTATS 2017](https://proceedings.mlr.press/v54/mcmahan17a.html) | Fixed local steps; uniform weights by default, example weighting optional. | [Numerical · Protocol · Integration](docs/baseline-validation.md#first-maintenance-audit-19-october-2026) · [Limits](docs/baseline-validation.md#centralized-fl--fedavg) |
| D-PSGD | `dpsgd` | [Can Decentralized Algorithms Outperform Centralized Algorithms? — NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/f75526659f31040afeb61cb7133e4e6d-Abstract.html) | Fixed-topology train-then-average variant; differs from the original paper’s update order. | [Numerical · Protocol · Integration](docs/baseline-validation.md#first-maintenance-audit-19-october-2026) · [Limits](docs/baseline-validation.md#d-psgd-preserve-train-then-average) |
| AD-PSGD (A-DPSGD) | `adpsgd` | [Asynchronous Decentralized Parallel Stochastic Gradient Descent — ICML 2018](https://proceedings.mlr.press/v80/lian18a.html) | Active/passive asynchronous specialization; single-gradient passive updates. | [Numerical · Protocol · Integration](docs/baseline-validation.md#first-maintenance-audit-19-october-2026) · [Limits](docs/baseline-validation.md#ad-psgd) |
| Epidemic Learning | `epidemic` | [Epidemic Learning: Boosting Decentralized Learning with Randomized Communication — NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html) | Changing directed/regular graphs; paper equivalence not established. | [Numerical · Integration](docs/baseline-validation.md#first-maintenance-audit-19-october-2026) · [Limits](docs/baseline-validation.md#epidemic) |
| Gossip Learning | `gossip` | [Gossip Learning as a Decentralized Alternative to Federated Learning — DAIS 2019](https://www.inf.u-szeged.hu/~ihegedus/publ/dais19a.pdf) | Legacy periodic gossip and model-age weighting. | [Legacy](docs/baseline-validation.md#legacy-implementations) · [Limits](docs/baseline-validation.md#legacy-implementations) |
| Super Gossip Learning | `super-gossip` | [Budaj's EPFL thesis, §4.5 — 2024](https://www.epfl.ch/labs/sacs/wp-content/uploads/2024/07/2024-spring-lubor-1.pdf#page=23); [original project](https://github.com/sacs-epfl/async-dl) | Internal research variant; no separate peer-reviewed paper identified. | [Legacy](docs/baseline-validation.md#legacy-implementations) · [Limits](docs/baseline-validation.md#legacy-implementations) |
| Heterogeneous Gossip Learning (HGL) | `lubor` | [Budaj's EPFL thesis, Chapter 6 — 2024](https://www.epfl.ch/labs/sacs/wp-content/uploads/2024/07/2024-spring-lubor-1.pdf#page=39) | Thesis prototype; retains its historical CLI name. | [Legacy](docs/baseline-validation.md#legacy-implementations) · [Limits](docs/baseline-validation.md#legacy-implementations) |
| Shatter | `shatter`; `shatter-r2`, `shatter-r4` in runners | [Noiseless Privacy-Preserving Decentralized Learning — PoPETs 2025](https://petsymposium.org/popets/2025/popets-2025-0043.php) | Real-node chunked-exchange specialization. | [Protocol · Integration](tests/test_protocol_execution.py) · [Limits](docs/baseline-validation.md#shatter) |
| Moshpit SGD | `moshpit` | [Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices — NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/97275a23ca44226c9964043c8462be96-Abstract.html) | Stable-membership group collectives. | [Numerical · Protocol · Integration](tests/test_peer_baselines.py) · [Limits](docs/peer-baselines.md#implemented-protocol-contracts) |
| Teleportation | `teleportation`; `teleportation-ring`, `teleportation-hypercube` in runners | [Scalable Decentralized Learning with Teleportation — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b3732a13897c4cea145c3bdece80de64-Abstract-Conference.html) | Algorithm 1 specialization; hypercube is not the paper's Base-2 graph. | [Numerical · Protocol · Integration](tests/test_peer_baselines.py) · [Limits](docs/peer-baselines.md#implemented-protocol-contracts) |
| D-Local-SGD | `d-local-sgd-ring` in runners | [A Unified Theory of Decentralized SGD with Changing Topology and Local Updates — ICML 2020](https://proceedings.mlr.press/v119/koloskova20a.html) | Fixed full-participation ring with configurable local updates. | [Numerical · Protocol · Integration](tests/test_peer_baselines.py) · [Limits](docs/peer-baselines.md#implemented-protocol-contracts) |
| Quasi-Global Momentum / DSGDm controls | `qgm`; `dsgdm`, `dsgd` in runners | [Quasi-global Momentum: Accelerating Decentralized Deep Learning on Heterogeneous Data — ICML 2021](https://proceedings.mlr.press/v139/lin21c.html) | One-step QG-DSGDm, optional Nesterov, persistent local-momentum and plain controls. | [Numerical · Protocol · Reference · Integration](docs/qgm.md#validation) · [Limits](docs/qgm.md) |
| SWIFT | `swift` | [SWIFT: Rapid Decentralized Federated Learning via Wait-Free Model Communication — ICLR 2023](https://arxiv.org/abs/2210.14026) | Wait-free cached neighbor models, pre-average gradients, periodic averaging, regular-graph CCS. | [Numerical · Protocol · Integration](docs/swift.md#validation) · [Limits](docs/swift.md#timing-payloads-and-supported-settings) |
| GT-DSGD | `gt-dsgd` | [An improved convergence analysis for decentralized online stochastic non-convex optimization — IEEE TSP 2021](https://arxiv.org/abs/2008.04195) | Algorithm 1: stochastic gradient tracking with two explicit adapt-then-combine exchanges per iteration. | [Numerical · Protocol · Integration](docs/gt-dsgd.md#validation) · [Limits](docs/gt-dsgd.md#protocol-and-timing) |
| DeTAG | `detag` | [Optimal Complexity in Decentralized Training — ICML 2021](https://proceedings.mlr.press/v139/lu21a.html); [journal extension — JMLR 2023](https://jmlr.org/papers/v24/22-0044.html) | Phase-based accelerated gossip, fixed-parameter gradient averaging and explicit compute/network overlap. | [Numerical · Protocol · Integration](docs/detag.md#validation) · [Limits](docs/detag.md#gossip-and-timing) |
| RelaySGD | `relaysgd` | [RelaySum for Decentralized Deep Learning on Heterogeneous Data — NeurIPS 2021](https://arxiv.org/abs/2110.04175) | Explicit tree relaying of model sums and population counts, with bounded hop delay. | [Numerical · Protocol · Reference · Integration](docs/relaysgd.md#validation) · [Limits](docs/relaysgd.md#timing-and-communication) |
| Stochastic Gradient Push | `sgp` | [Stochastic Gradient Push for Distributed Deep Learning — ICML 2019](https://proceedings.mlr.press/v97/assran19a.html) | Algorithm 1: numerator SGD and scalar-mass debiasing on directed, optionally time-varying graphs. | [Numerical · Protocol · Reference · Integration](docs/sgp.md#validation) · [Limits](docs/sgp.md#directed-topology-and-timing) |
| Decentralized AMSGrad | `damsgrad` | [On the Convergence of Decentralized Adaptive Gradient Methods — ACML 2022 / PMLR 2023](https://proceedings.mlr.press/v189/chen23b.html) | Algorithm 3: local moments, dynamic average consensus on adaptive state, delayed denominator, and overlapping old-state exchange. | [Numerical · Protocol · Integration](docs/damsgrad.md#validation) · [Limits](docs/damsgrad.md#communication-timing-and-supported-scope) |
| FSPDA-SA | `fspda` | [A Stochastic Approximation Approach for Efficient Decentralized Optimization on Random Networks — arXiv v2, 2025](https://arxiv.org/abs/2410.18774v2) | Simultaneous primal–dual SA with random undirected edges and shared unscaled coordinate masks; follows paper (7)/(36). | [Numerical · Protocol · Integration](docs/fspda.md#validation) · [Limits](docs/fspda.md#random-network-communication-and-supported-scope) |
| MoTEF | `motef` | [Towards Faster Decentralized Stochastic Optimization with Communication Compression — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/39b77b5e422b4e070e2811b73ea9bcf7-Abstract-Conference.html) | Base Algorithm 1: momentum tracking with persistent compressed model/tracker estimates and charged startup. | [Numerical · Protocol · Reference · Integration](docs/motef.md#validation) · [Limits](docs/motef.md#communication-and-supported-scope) |
| MoTEF-VR | `motef-vr` | Same ICLR 2025 paper, Algorithm 2 | Coupled fresh minibatch gradient pairs, momentum tracking and compressed error feedback; two charged gradients per iteration. | [Validation and limits](docs/motef-vr.md) |
| CHOCO-SGD | `choco` | [Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication — ICML 2019](https://proceedings.mlr.press/v97/koloskova19a.html); [deep-learning extension — ICLR 2020](https://arxiv.org/abs/1907.09356) | Persistent public estimates; Top-k/random-k compression; explicit local-SGD block extension. | [Numerical · Protocol · Reference · Integration](docs/choco.md#usage-and-validation) · [Limits](docs/choco.md#supported-model-and-optimizer-state) |
| DSpodFL | `dspodfl` | [Decentralized Sporadic Federated Learning: A Unified Algorithmic Framework with Convergence Guarantees — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html) | One-step rule and explicit local-SGD block extension. | [Numerical · Protocol · Integration](tests/test_dspodfl.py) · [Limits](docs/dspodfl.md) |
| DeMo | `demo` | [DeMo: Decoupled Momentum Optimization — ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17790fc9b4828206f99a0dea35da3657-Abstract-Conference.html) | Reference Signum optimizer; ring all-gather and CNN adaptation. | [Numerical · Protocol · Reference · Integration](docs/demo.md#validation) · [Limits](docs/demo.md) |
| Conflux (original) | `conflux` | Unpublished prototype; related sampling work: [Practical Federated Learning without a Server — EuroMLSys 2025](https://euromlsys.eu/pdf/euromlsys25-10.pdf). | Chunked sampling prototype; not an implementation of Plexus. | [Integration · Legacy](tests/test_protocol_execution.py) · [Limits](docs/baseline-validation.md#legacy-implementations) |

[Baseline contracts and sources](docs/peer-baselines.md),
[DSpodFL semantics](docs/dspodfl.md), [DeMo semantics](docs/demo.md),
[CHOCO-SGD semantics and usage](docs/choco.md),
[Quasi-Global Momentum](docs/qgm.md), [SWIFT](docs/swift.md), [GT-DSGD](docs/gt-dsgd.md), [DeTAG](docs/detag.md), [RelaySGD](docs/relaysgd.md), [SGP](docs/sgp.md), [MoTEF](docs/motef.md), [MoTEF-VR](docs/motef-vr.md), [FSPDA-SA](docs/fspda.md), [Decentralized AMSGrad](docs/damsgrad.md).
Paper links identify the source algorithms, not a claim of exact paper reproduction.
Validation labels describe specific checks, not a certification of every option.
Reference means an executed comparison with authors' code; comparison with
paper equations is Numerical. CPU/CUDA, broker replay and benchmark coverage
are specified in the linked evidence rather than implied by a label.
[Validation contracts, evidence and remaining gaps](docs/baseline-validation.md).
Stable-membership restrictions, optimizer assumptions and adaptations are
explicit in those documents. Legacy entries have not all been re-audited.
The historical `pushsum` branch is a separate experimental protocol; the maintained
SGP implementation follows the published algorithm and authors' reference.

## Run a small comparison

For paired, fixed-time rankings with live accuracy/traffic curves, use the
[algorithm leaderboard](docs/leaderboard.md). Budgets are configurable simulated
seconds; the offline example covers all registered algorithms:

The multi-workload dashboard adds a dataset/model selector for CIFAR-10/GNLeNet
and CIFAR-100/ResNet-18-GN:

```sh
python scripts/serve_leaderboard.py --catalog experiments/leaderboard-workloads.json --port 8768
```

```sh
python scripts/run_benchmark.py experiments/leaderboard.yaml
# In another terminal:
python scripts/serve_leaderboard.py output/leaderboard
```

Open <http://127.0.0.1:8765>. Rankings show accuracy at each budget, average
accuracy over time, worst-client performance, traffic, time to accuracy, best
observed accuracy, and actual execution time. Results remain provisional until
the full paired seed set is available. The synthetic example is untuned screening.

For automatic validation-only LR selection with equal trial budgets, followed by
fresh-seed test evaluation of five algorithms on CIFAR-10/GNLeNet with 32 clients:

```sh
python scripts/run_benchmark.py experiments/leaderboard-cifar10-32.yaml --plan
python scripts/run_benchmark.py experiments/leaderboard-cifar10-32.yaml
```

Prepare `data/cifar10` first; see the [tuning protocol and live phase view](docs/leaderboard.md#automated-equal-budget-lr-tuning).

To reuse the selected LRs and evaluate the uniform average of all client models
through 900 simulated seconds, with four workers across two GPUs:

```sh
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_benchmark.py experiments/leaderboard-cifar10-32-global900.yaml
```

See [global-model evaluation and the frozen-LR rerun](docs/leaderboard.md#global-model-rerun-with-frozen-lrs-and-a-longer-horizon).

The extreme Dirichlet alpha 0.1 / heterogeneous-hardware scenario reuses those
LRs and runs separately. Both scenarios are selectable on one live dashboard,
which defaults to the 900-second budget:

```sh
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_benchmark.py experiments/leaderboard-cifar10-32-dirichlet01-heterogeneous.yaml
python scripts/serve_leaderboard.py --catalog experiments/leaderboard-settings.json --port 8766
```

The catalog also includes IID/heterogeneous and non-IID/homogeneous experiments.
See [the full data/hardware matrix](docs/leaderboard.md#remaining-datahardware-combinations)
for their configurations and concurrent execution commands.

To tune five additional algorithms and extend all four comparisons to ten
algorithms, see [the leaderboard extension](docs/leaderboard.md#five-algorithm-extension-across-all-four-settings).

The peer-algorithm batch adds Teleportation, Moshpit, Epidemic Learning,
Gossip Learning and Shatter, giving 15 algorithms with one local step.
AD-PSGD and DeTAG are deferred; their artifacts are preserved. See
[the peer-algorithm batch](docs/leaderboard.md#five-peer-algorithm-batch-with-ad-psgd-and-detag-deferred)
for frozen validation-selected LRs and all four settings.
The [deferred-algorithm investigation backlog](docs/leaderboard.md#deferred-algorithms-investigation-backlog)
records the SWIFT, AD-PSGD and DeTAG blockers and steps for re-entry.
The [centralized reference](docs/leaderboard.md#step-budgeted-centralized-reference)
tunes its LR, trains one node on all training data for a fixed SGD-step budget,
and displays held-out accuracy/loss as dashed horizontal lines on the dashboard.
The active leaderboard has been reset for a [fresh methodology campaign](docs/leaderboard.md#fresh-methodology-campaign-4-november-2026).
It starts with D-PSGD on a fixed 5-regular graph, jointly tunes learning rate and
local steps on IID/homogeneous validation data, and freezes the winner across four
settings. Algorithm names now open pages with explanations, paper links, parameter
tables and live tuning evidence. The previous catalog is preserved as
`experiments/leaderboard-legacy-settings.json`.

The [D-PSGD k-regular extension](docs/leaderboard.md#d-psgd-with-log2n-regular-topology)
adds a separate seeded 5-regular graph variant for 32 clients, bringing the
leaderboard to 16 algorithms while preserving D-PSGD-ring and prior results.
The [DSpodFL extension](docs/leaderboard.md#dspodfl-extension) adds an LR-tuned
variant across all four settings, extending the comparison to 17 variants while
preserving previous runs. FSPDA-SA is deferred for executor runtime; its partial
results and investigation steps are preserved in the backlog.

For configuration files and parameter sweeps, use the shared experiment runner:

```sh
python scripts/run_experiment.py experiments/smoke.yaml
python scripts/run_experiment.py experiments/food101.yaml --plan
```

YAML/JSON files support shared settings, Cartesian matrices or named runs, and
CLI overrides. Each run gets its own process, log, resolved configuration and
manifest. See [experiment configuration](docs/experiments.md) for execution modes,
output protection, and the pinned Food-101 LoRA comparison.

This runs real protocol-generated DAGs on CPU, with synthetic data and no downloads:

```sh
python scripts/validate_algorithms.py --variants shatter-r2 moshpit \
  --peers 8 --sample-size 4 --profiles uniform heterogeneous \
  --partitioners iid dirichlet --seeds 0 --duration 20 --test-period 5 \
  --output output/smoke
```

The runner also supports bundled handwritten digits (`--dataset digits`).
`validate_conflux.py` remains a compatibility entry point for older commands.
CIFAR-10 and Tiny ImageNet can load directly from Hugging Face. For offline
benchmark runs, save them with `scripts/prepare_dataset.py` and pass the directory
to `run_cifar.py` / `run_tiny.py`; see [dataset configuration](docs/datasets.md).
Each run records parameters, versions, a source hash and per-run outputs.

At fixed simulated times, accuracy is the mean of all nodes' latest completed
local checkpoints, including nodes that have not recently updated. No uncharged
global parameter average is constructed. Unreached accuracy targets are censored.
Controlled heterogeneous profiles vary compute and bandwidth independently;
these profiles are distinct from FedScale traces (`data/fedscale_traces`).

Optional [loss-based membership inference audits](docs/membership-inference.md)
measure leakage through a participant's own local model, using independent
nonmember calibration and class-matched evaluation cohorts. Run
`python scripts/compare_membership.py --output output/membership-comparison`
for a matched-training-budget D-PSGD/Shatter comparison on offline digits data.
Attack results include balanced accuracy, ROC-AUC, TPR and observed FPR; empirical
attack resistance is not a formal privacy guarantee.
Add `--view incoming` and use a fresh output directory to attack actual delivered
D-PSGD models and Shatter chunks. Missing chunks are filled from receiver-known
state; outputs report observation coverage and the fraction of state received.
`--cohort-policy global_nonmembers` retains the global nonmember label distribution;
the default `class_matched` policy controls for label-distribution leakage. Report
the policy when comparing results, especially under non-IID partitions.

## Coordinator and brokers

Generate a DAG without executing it:

```sh
python main.py --algorithm moshpit --dataset synthetic --model mlp \
  --peers 8 --group-size 4 --momentum 0 --rounds 3 --dry-run
```

Without `--dry-run`, the coordinator waits for its configured brokers. Start a
broker with `python broker.py --coordinator tcp://HOST:5555 --port 5556 --workers 2`.
All registered settings types are supported by the shared broker registry.
Use `--from-dir PATH` to load a saved workflow/settings pair. Add `--local`
to execute the saved DAG in process without regenerating the simulation.

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
