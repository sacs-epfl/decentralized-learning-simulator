# Literature baselines for Conflux V2

## Selection

| Method | Peer-reviewed source | Why compare it |
| --- | --- | --- |
| Moshpit SGD | [NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/97275a23ca44226c9964043c8462be96-Abstract.html) | Small-group chunked all-reduce on heterogeneous devices; the closest collective-communication baseline. |
| Teleportation | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b3732a13897c4cea145c3bdece80de64-Abstract-Conference.html) | Randomly sampled clients inherit logical model tokens, train, and gossip within the active sample. |
| D-Local-SGD | [D-PSGD, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/f75526659f31040afeb61cb7133e4e6d-Abstract.html) | Full-participation sparse mixing; establishes whether sampling and chunked collectives improve on a basic decentralized reference. |
| Shatter | [PoPETs 2025](https://petsymposium.org/popets/2025/popets-2025-0043.php) | Existing chunked virtual-node baseline, retained in every comparison. |

[AD-PSGD (ICML 2018)](https://proceedings.mlr.press/v80/lian18a.html)
is also relevant to heterogeneous compute, but its concurrent update/communication
semantics need a separate audit. The existing implementation is not certified by
this work. Push-sum methods remain outside the requested scope.

## Implemented protocol contracts

**Moshpit:** algorithms 1 and 2, specialized to stable membership. A seeded
coordinate grid implements rolling group keys by cycling the grouping axis;
members of a group separate in the next dimension. Each group performs an
explicit readiness/plan exchange followed by butterfly reduce-scatter and
all-gather. Chunk ownership stays within the group. Fractions solve the minimax
load-balancing problem in [Appendix G](https://arxiv.org/html/2103.03239#A7),
using symmetric upload/download capacities. A monotone feasibility solver avoids
adding an LP dependency to runtime code; tests independently compare it with
SciPy's LP solver. Integer partitioning preserves every tensor element and byte,
including zero-sized partitions.

The default is one averaging iteration between local-SGD blocks, as in the
paper's ImageNet configuration. `averaging_steps` is configurable; a complete
grid reaches exact consensus after one iteration per dimension. DHT discovery,
failed collectives, recovery and churn are excluded. Membership discovery is an
oracle, consistent with the V2 sampling abstraction. All readiness messages,
model payloads and metadata headers consume the shared network scheduler.

**Teleportation:** [Algorithm 1](https://arxiv.org/html/2501.15259#S3), with
uniform sampling without replacement and random distinct token assignment.
Each token trains on its selected client's data, mixes with same-round trained
neighbor models, and moves to the next token owner through a charged full-model
transfer. Mixing waits for both the local training result and all required
neighbors, even if incoming messages arrive first. Self-handoffs use no network.
The ring and hypercube choices both use symmetric, doubly stochastic mixing
including a self-loop. Hypercube is an additional valid Algorithm 1 topology;
it is not the paper's Base-2 Graph. No claim reproduces the paper's tuning scheme.

The prior Teleportation code used sorted sample IDs as tokens, could mix before
its own training completed, and misindexed next-round state. It was replaced
with an event-driven state machine and mathematical regression tests.

**D-Local-SGD ring:** the same tested mixing engine with fixed tokens, all
clients active and no model handoff between clients. With one local step this
is decentralized SGD in adapt-then-combine form; with the benchmark's five
local steps it is explicitly a local-SGD extension, not a verbatim reproduction
of the 2017 algorithm.

All methods use the existing model/data code, bandwidth contention, latency,
training DAG executor and evaluation machinery. Stable membership is required.
The new baselines reject nonzero momentum rather than silently discarding
token-carried optimizer state. The comparison uses plain SGD for every method.

## Benchmark design

The initial screening uses CIFAR-10/GNLeNet, 64 clients, three seeds, uniform and
heterogeneous capacities, and IID and Dirichlet-alpha-0.1 data. Sampled methods
activate 16 clients; V2/Shatter use 8 chunks, Moshpit uses groups of at most 8.
The model payload is pinned at 346,251 bytes for all protocols. Every run uses
the same 900-second simulated budget, snapshots every 150 seconds, five local
SGD steps, batch 32, learning rate .01, zero momentum and weight decay. No
baseline-specific learning-rate or topology tuning uses heldout test data.
The existing native dataset split has 45,000 training, 5,000 tuning and 10,000
heldout test images; all checkpoint metrics use the complete test split.

Primary accuracy/loss is the mean of all physical nodes' stored completed
post-aggregation checkpoints, including stale or initially idle nodes. This
matches the existing V2/Shatter metric, and is not the papers' hypothetical
global parameter average. Teleportation's inactive models are replaced on
reactivation; the stored-node readout therefore also measures dissemination.
V2's latest common aggregate and node staleness are saved separately and must
not be substituted for the network-wide metric when ranking methods.

Traffic includes control messages and unfinished transfer progress. Observing
network checkpoints does not mutate the scheduler or its rounding. Time to
accuracy is the first observed checkpoint reaching a target; unreached targets
remain censored. Simulator/GPU execution time is recorded separately from
simulated learning time. Three seeds and a short common-parameter sweep provide
screening evidence, not a universal ranking or tuned paper reproduction.

Run a sweep with `scripts/run_cifar.py --variants conflux-v2 shatter-r2
teleportation-ring teleportation-hypercube d-local-sgd-ring moshpit`; use
`--group-size` and `--averaging-steps` for Moshpit. The Tiny ImageNet runner also
accepts these variants, with `--momentum 0` required for the new methods.
`scripts/report_baselines.py --root <results>` checks both profile directories,
audits every per-node checkpoint, and produces accuracy, loss, traffic and
censored time-to-accuracy comparisons.
