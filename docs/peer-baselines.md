# Literature baselines

| Method | Peer-reviewed source | Why compare it |
| --- | --- | --- |
| Moshpit SGD | [NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/97275a23ca44226c9964043c8462be96-Abstract.html) | Small-group chunked all-reduce on heterogeneous devices; the closest collective-communication baseline. |
| Teleportation | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b3732a13897c4cea145c3bdece80de64-Abstract-Conference.html) | Randomly sampled clients inherit logical model tokens, train, and gossip within the active sample. |
| D-Local-SGD | [Unified decentralized/local SGD, ICML 2020](https://proceedings.mlr.press/v119/koloskova20a.html); historical [D-PSGD, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/f75526659f31040afeb61cb7133e4e6d-Abstract.html) | Full-participation sparse mixing; establishes whether sampling and chunked collectives improve on a basic decentralized reference. |
| Shatter | [PoPETs 2025](https://petsymposium.org/popets/2025/popets-2025-0043.php) | Existing chunked virtual-node baseline, retained in every comparison. |


See also [DSpodFL](dspodfl.md) and [DeMo](demo.md).

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
oracle. All readiness messages,
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
of the 2017 algorithm. It fits Algorithm 1 of Koloskova et al. (ICML 2020),
using identity mixing for the intermediate local updates and a fixed ring
mixing matrix after the fifth update.

All methods use the existing model/data code, bandwidth contention, latency,
training DAG executor and evaluation machinery. Stable membership is required.
In these implementations, chunking and aggregation have zero simulated
compute cost; local SGD has an explicit simulated duration. Measured GPU
execution time does not feed back into protocol time.
The new baselines reject nonzero momentum rather than silently discarding
token-carried optimizer state. The comparison uses plain SGD for every method.

## Shared benchmarking

Use `scripts/validate_algorithms.py` for synthetic/digits workloads,
`scripts/run_cifar.py` for native CIFAR-10, and `scripts/run_tiny.py` for
Tiny ImageNet with compatible models/optimizers. The baseline aliases are
`moshpit`, `teleportation-ring`, `teleportation-hypercube`, `d-local-sgd-ring`,
`dspodfl` and `demo`. The coordinator accepts `--algorithm teleportation` with
`--teleportation-topology ring` or `hypercube` instead of the aliases.

Algorithm-specific numerical and causal tests live in `tests/test_peer_baselines.py`,
`tests/test_dspodfl.py` and `tests/test_demo.py`. Historical comparison tables,
plots and frozen experiment manifests remain on the `conflux-v2` research branch;
they are not new benchmark claims for a changed source tree.
