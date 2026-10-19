# Baseline validation contracts

The README distinguishes the source paper from the implementation's tested
contract. A successful run or registry/settings round-trip does not establish
algorithmic correctness. These checks cover the stated specialization, not every
paper option, topology, optimizer, model or failure mode.

## First maintenance audit: 19 October 2026

The maintained tests in `tests/test_classic_baselines.py` now execute the older
AD-PSGD, D-PSGD, centralized FL and Epidemic implementations. Old fixtures remain
in `tests/legacy`; the new tests replace their obsolete API assumptions rather
than enabling those fixtures unchanged.

The numerical oracle is independent full-batch logistic regression on four
unequal client datasets, with known initialization and direct PyTorch SGD.
D-PSGD and FL are compared over four rounds, including different local-step
counts, partial/full participation, synchronous/local round barriers, and skewed
compute/network completion order. AD-PSGD tests calculate gradients independently
at their original input models, apply them to the current mixed models, and check
that both ends of completed exchanges use the same pair of pre-mix values.
The tests explicitly require mixing during gradient computation.

Additional checks cover regular-graph mean preservation, directed Epidemic
mixing and isolated-node representation, exact traffic totals for completed FL
and D-PSGD runs, deterministic DAGs/metrics, current model publication at fixed
snapshots, duplicate/stale FL replies, settings rejection, saved-DAG output
routing, and assignment of the FL server to a broker. The optimized executor is
compared against the serial executor across complete DAG outputs, including
AD-PSGD gradient payloads, on CPU and CUDA.

### D-PSGD: preserve train, then average

The existing implementation deliberately remains:

```
y_i = LocalSGD(x_i, data_i, H)
x_i_next = sum_j W_ij * y_j
```

For H=1 and plain SGD this is `W * (x - learning_rate * g(x))` in node-row
notation. The [original paper](https://papers.nips.cc/paper_files/paper/2017/file/f75526659f31040afeb61cb7133e4e6d-Paper.pdf)
uses `W * x - learning_rate * g(x)`. These are different updates once local
models/gradients differ. Train-then-average is a valid decentralized local-SGD
choice; the audit does not replace it with the original paper's update order.
The default has local round dependencies; `--synchronous` additionally imposes a
global simulated-time round barrier. Regular ring graphs use uniform neighbor
and self weights. These tests do not establish the original paper's theorem for
this variant or validate churn.

### Centralized FL / FedAvg

Clients start from the server's common model, perform a fixed number of local
SGD steps, and upload their results. `--fl-weighting uniform` preserves the
historical default, giving selected clients equal weight. `--fl-weighting examples`
weights selected clients by their training partition sizes, normalized over that
sample. Unequal partition sizes and partial participation are tested explicitly.
`sample_size=0` in saved/programmatic settings means full participation; the CLI
retains its existing sample-size default.

This is a fixed-local-step adaptation of [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf),
whose presentation uses local epochs. There is no new tuning or epoch-based
reproduction claim. The historical server has essentially unlimited bandwidth;
client upload/download bandwidth and latency are still charged. This assumption
must accompany comparisons with decentralized protocols.

Fixed snapshots evaluate the clients' stored local models. Round-based server
evaluations are separate in `accuracies_<server_id>.csv`; they are not included
in the client-wide mean. Stable membership is required; unsupported churn is
rejected rather than leaving the server waiting indefinitely.

### AD-PSGD

This remains an active/passive bipartite specialization, not a reproduction of
all [paper implementation modes](https://proceedings.mlr.press/v80/lian18a/lian18a.pdf).
Active workers perform H local steps, send a model, wait for the exchange reply,
and then train again. Passive workers continuously compute single gradients.
A gradient finishing after an exchange is applied to that worker's current model,
using the gradient computed at the older model. Each gradient carries its own
DAG dependency. Both exchange participants average the exact same pre-mix pair;
multiple active workers can select the same passive worker.

The broken lazy passive-training bookkeeping was replaced with explicit
single-gradient completion events. This is a correctness repair with a changed
execution trace; historical AD-PSGD results should not be reused for the repaired
implementation. All workers start with identical seeded weights. Per-task SGD
optimizer state follows the shared trainer's existing convention: momentum is
fresh per task, not persistent across exchanges. The numerical contract uses
plain SGD, uniform pair weights and stable membership. Age weighting and other
optimizer choices are not certified by this audit. AD-PSGD uses the simulated
duration budget rather than a global synchronous round limit.

### Epidemic

Oracle mode uses seeded regular graphs; local mode independently samples outgoing
neighbors each round. Local mode now constructs every node explicitly, including
nodes with zero incoming neighbors. Received trained models and the local trained
model are averaged uniformly. The tests check these graph/mixing semantics and
reproducibility. A directed graph's receiving-side uniform averages generally do
not preserve the global mean; no doubly stochastic or paper-equivalence claim is
made for local mode. Churn handling has not been audited.

## Evidence and reproduction

Run the numerical checks:

```sh
python -m pytest tests/test_classic_baselines.py -q
```

Run the offline smoke matrix (four nodes, two seeds, two partition types, two
capacity profiles, four protocols = 32 cases):

```sh
python scripts/validate_classic_baselines.py --output output/classic-validation/new-matrix
```

Add `--distributed` to replay each protocol's first-seed heterogeneous non-IID DAG
through one broker/two workers and two brokers/one worker. These are real local
TCP brokers and spawned worker processes, not mocked calls. The replay compares
all saved per-node accuracy/loss rows and separate FL server rows with serial
execution. It does not compare every intermediate tensor across processes;
complete tensor comparisons are made in the local optimized-execution tests.
Use a new output directory for each invocation. A configurable broker
`--advertise-host` and automatic OS route selection avoid depending on `ifconfig`
or a particular platform's network-interface output.

The matrix has four half-second snapshots per case with four nodes each. Metrics
must be finite and model round IDs positive; communication counters must balance.
The two-second workload is deliberately too small for performance or convergence
rankings. Capacity profiles are synthetic, not FedScale traces. Compact recorded
results and environment/source hashes are in [classic-baselines.json](validation/classic-baselines.json).
The full maintained suite passed **152 tests on macOS/CPU** (nine CUDA skips)
and **161 tests on Linux/CUDA** on sacs006. The 32-case macOS matrix and a
16-case single-seed Linux matrix passed. Six macOS and eight Linux broker
replays matched their serial reference metrics. These are validation runs, not
accuracy comparisons between algorithms. Raw run artifacts stay under the
recorded local/server output directories.

## Meaning of the README labels and remaining work

| Label | Evidence | Remaining limitations |
|---|---|---|
| Equation / update checks | Small numerical oracles check specific updates; graph, ordering and state invariants are tested. | Not an assertion of full paper reproduction or all settings. |
| Integration checks | Real training DAG execution; for AD-PSGD, D-PSGD and FL, real broker/worker replay matches serial metrics. | Multi-host networking and failures remain untested. |
| Protocol checks | Training/exchange dependencies, chunk reconstruction, traffic and snapshots are checked. | Not a separate complete reference implementation of the paper. |
| Optimizer reference check | DeMo's optimizer has a separate multi-step check against its source reference; DCT and residual identities are tested. | Simulator transport, CNN adaptation and all model architectures are not established by that optimizer check. |
| Legacy | Source is present and import/settings checks exist; some shared execution tests may apply. | No maintained independent algorithm-level oracle sufficient for trusted comparisons. |

Moshpit and Teleportation have numerical collective/mixing checks; DSpodFL has
one-step/block-update and mixing-matrix tests; DeMo has DCT, residual, collision
averaging and optimizer reference checks. These remain useful but bounded
contracts. Priorities after this audit are independent multi-step training
trajectories for those methods, migration of Gossip/Super-Gossip/HGL, model-buffer
and optimizer-state coverage on more architectures, supported failure modes,
and then replicated small CIFAR-10 convergence checks. No additional large
accuracy benchmark or learning-rate selection was performed in this audit.

## CHOCO-SGD addition

CHOCO-SGD has independent multi-round numerical checks, persistent compression
state checks, and an optional pinned reference-helper audit. See the separate
[CHOCO contract and validation evidence](choco.md) for its update order, supported
compressors, local-SGD extension and communication accounting. Earlier audit
counts and source hashes above describe the earlier classic-baseline snapshot.
