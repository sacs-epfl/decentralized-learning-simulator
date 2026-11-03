# RelaySGD using RelaySum

`relaysgd` implements Algorithm 1 (RelaySGD, model relaying) from
[Vogels et al., NeurIPS 2021](https://arxiv.org/abs/2110.04175), on a single
fixed tree. The authors' [implementation](https://github.com/epfml/relaysgd)
was checked at revision `e1e332bfaf6a2b29f1677b7fa31a27b6e0d2802a`.
This baseline uses one plain SGD minibatch step per relay iteration, stable
membership, and no communication overlap with the next gradient computation.
It supports balanced binary, path and star trees.

## Relay state and update

Let `p_i^r` be client i's model parameters after its local SGD step in iteration
r. Each directed edge i→j sends a dense sum and an integer contribution count:

```
m_i→j^r = p_i^r + sum(m_k→i^(r-1), k in neighbors(i) except j)
c_i→j^r = 1     + sum(c_k→i^(r-1), k in neighbors(i) except j)
```

Incoming state initially contains no messages or counts. Forwarding reads the
previous iteration's immutable inbox and excludes the destination branch.
Current-iteration arrivals are buffered separately, so early arrivals cannot
introduce echoes or shorten the logical relay delay. A tree gives every origin
exactly one route. No client reads a remote model without a charged transfer.

After its gradient and all same-iteration neighbor messages complete, client i
forms `S_i = p_i^r + sum(m_j→i^r)` and `C_i = 1 + sum(c_j→i^r)`.
The default, `--relaysgd-normalization counts`, uses `x_i = S_i / C_i`,
following Algorithm 1's startup count. With
`--relaysgd-normalization initial`, it uses
`x_i = (S_i + (N-C_i)*x_initial) / N`.

The latter matches the deployed reference's `world_size` mode with its initial
model fallback. Its `counts` mode matches the default here. The README's short
academic example instead fills missing contributions with the current local
parcel; that third startup convention is excluded. The selected convention is
recorded in settings and `relaysgd-tree.json`.

An adjacent client's current parcel arrives in the current iteration; each
additional hop introduces one older iteration. At iteration r, the represented
origins are those at distance at most r, and each appears exactly once. Counts
reach N after each node's eccentricity in iterations. The resulting uniform
origin weights apply to parcels of different ages. They do not imply an average
of all clients' current parcels, or conservation of that current-parcel mean.
Path/star choices expose the tradeoff between hop delay and degree contention.

## Timing and communication

Clients wait only for their own gradient and their neighbors' current iteration
messages. The default has no global barrier; `--synchronous` adds a global
barrier before each iteration. Heterogeneous compute and network capacity
change completion times, while retaining the same logical numerical trajectory
for a fixed oracle stream. The paper's finite hop delay is an iteration bound,
not a wall-clock bound under arbitrary slow clients.

Each complete iteration sends exactly `2*(N-1)` directed messages. A packet
fuses one actual dense parameter vector and an 8-byte int64 count, plus one
generic control header and serialized iteration metadata. Even startup packets
occupy a full vector. Forwarded subtree sums remain one vector regardless of
subtree size. The simulator charges each real edge transfer with shared upload,
download and latency constraints; it does not use the reference's maximum-degree
byte proxy. `model_size_bytes` does not override this explicit payload layout.

Vector additions, topology construction and normalization have zero simulated
compute cost. The ordinary SGD `train` operation supplies gradient duration
and CPU/CUDA numerical execution, including the existing GPU batching path.
Measured executor wall time is separate from simulated protocol time. Numerical
operations have importable contracts and explicit DAG dependencies, allowing
saved-DAG replay and worker execution. Source models and incoming sums are
immutable. Non-parameter buffers stay with the trained local model.

The default balanced tree uses heap-index edges `((i-1)//2, i)`. A path uses
consecutive client IDs; a star has client 0 at its center. Shared seeded model
initialization and known tree membership have zero setup cost. Arbitrary graph
spanning-tree discovery, double binary trees, RelaySum/Grad, momentum, message
drops, communication/computation overlap, churn and LoRA are excluded.
`local_steps=1` and `momentum=0` are required. Coupled SGD weight decay is
supported. Parameters must be fully trainable floating tensors of one dtype.

Snapshots evaluate the last completed relay iteration, including initialization
before the first completion. `rounds` and individual test intervals count local
SGD/relay iterations. `stop=duration` continues to the common horizon;
`stop=rounds` caps iterations. `relaysgd-rounds.csv` records gradient readiness,
iteration completion and represented population count.

## Usage

In `conflux-gpu`:

```sh
python main.py --local --algorithm relaysgd --dataset synthetic --model mlp \
  --peers 8 --rounds 10 --duration 30 --local-steps 1 --momentum 0 \
  --learning-rate .05 --relaysgd-topology balanced \
  --relaysgd-normalization counts --test-method snapshot --test-period 5 \
  --work-dir output/relaysgd-validation/local
```

Options are `--relaysgd-topology balanced|path|star` and
`--relaysgd-normalization counts|initial`. The synthetic/digits, CIFAR and
Tiny ImageNet runners accept `relaysgd` and these options. Select one local step
and zero momentum explicitly when a shared runner's defaults differ.

## Validation

`tests/test_relaysgd.py` independently enumerates individual origin histories
and tree distances to calculate every expected parcel, sum, count and model.
It checks three trees, both startup rules, skewed compute and optional global
barriers, including exact packet bytes. Other cases cover early/duplicate
arrivals, causal waiting, single-worker SGD, small trees, local buffers,
immutable state, duration stopping, snapshots, reproducibility, streaming
execution, saved-DAG replay and CPU/CUDA agreement.

`tests/check_relaysgd_reference.py` executes the pinned authors'
`RelayMechanism`, forwarding and receive-sum arithmetic unchanged, replacing
only distributed transport/topology discovery with deterministic in-memory
messages. Seven clients run eight iterations for each tree and normalization
with a shared nonzero initial model and non-IID quadratic objectives. All
outgoing sums/counts and resulting models agree; maximum parameter error is
`3.33e-16` in float64. This checks arithmetic rather than MPI execution or its
reported byte estimates. Reproduce with:

```sh
git clone https://github.com/epfml/relaysgd.git /tmp/relaysgd-reference
git -C /tmp/relaysgd-reference checkout e1e332bfaf6a2b29f1677b7fa31a27b6e0d2802a
python tests/check_relaysgd_reference.py /tmp/relaysgd-reference
python -m pytest -q
```

Fresh suite, CLI and screening results are recorded in
[the validation record](validation/relaysgd.json), with
[per-seed summaries](validation/relaysgd-digits-summary.csv) and
[fixed-time curves](validation/relaysgd-digits-curves.csv).

On 3 November 2026 the full suite passed **590 tests**, including 50
RelaySGD-specific cases, in `conflux-gpu` on an RTX 5000 Ada GPU. The public
CUDA CLI and both large-dataset runners' option parsing also passed.

## Short digits screening

The 48-run screening includes default balanced-tree RelaySGD, plain ring DSGD,
path-tree RelaySGD, and balanced-tree RelaySGD with initial-model padding.
The latter two sensitivity configurations and the common learning rate were
declared before execution; every configuration is reported. Eight clients use
a 64-32-10 ReLU MLP, batch 32, learning rate .05, zero momentum/decay, and one
minibatch step per iteration. Every local model is evaluated on all 445 held-out
digits examples at 1, 2 and 3 simulated seconds.

Final accuracy averages over eight local models, then reports mean ± sample
standard deviation across seeds 0/1/2, in percentage points:

| Capacity | Data | Ring DSGD | RelaySGD balanced/counts | RelaySGD path/counts | RelaySGD balanced/initial |
| --- | --- | ---: | ---: | ---: | ---: |
| Uniform | IID | 87.00 ± 3.57 | 70.86 ± 4.47 | 65.21 ± 0.62 | 70.60 ± 4.24 |
| Uniform | Dirichlet α=.3 | 79.29 ± 2.12 | 44.67 ± 9.38 | 39.77 ± 9.41 | 44.35 ± 9.26 |
| Heterogeneous | IID | 33.06 ± 18.00 | 14.97 ± 5.65 | 13.99 ± 5.07 | 14.68 ± 5.46 |
| Heterogeneous | Dirichlet α=.3 | 22.98 ± 12.29 | 15.37 ± 8.21 | 14.94 ± 7.70 | 15.31 ± 8.14 |

RelaySGD has lower mean accuracy than ring DSGD in every condition of this
short screening. The path has lower means than the balanced tree, and initial
padding changes the balanced results only slightly. These results do not
establish a longer-run ranking or reproduce the paper's tuned deep-learning
experiments. The tree/ring comparison changes topology as well as the algorithm;
it cannot isolate the effect of relay arithmetic.

Default RelaySGD sends 36.40 MB under uniform capacity, versus 41.83 MB for
ring DSGD, about 13% less. They complete 2136 local model updates and schedule
2144 gradient tasks each, including tasks in progress at the horizon. The tree
has 14 directed edges versus the ring's 16. Heterogeneous means are 6.68 MB and
388.33 completed updates for balanced RelaySGD, versus 7.62 MB and 386.33 for
DSGD. Counters include partial final-horizon transfers. Dense parameters are
9640 bytes; RelaySGD adds its 8-byte count before the common header/metadata.

Nominal bandwidth is 100 MB/s and latency 2 ms. Uniform gradients take 9 ms.
Heterogeneous clients independently receive gradient times from 6/15/36/90 ms,
with a quarter at 4× nominal bandwidth and the remainder .5×. Data settings
and capacities are exactly paired within seed/profile across methods and
configurations; mean capacity differs across profiles. Actual CUDA wall times
overlap other validation work and depend on batching calibration, so they are
not protocol speed measurements.

Reproduce the default comparison with:

```sh
python scripts/validate_algorithms.py --variants relaysgd dsgd --dataset digits \
  --peers 8 --local-steps 1 --duration 3 --test-period 1 --bandwidth 100000000 \
  --latency-ms 2 --learning-rate .05 --profiles uniform heterogeneous \
  --partitioners iid dirichlet --seeds 0 1 2 --device cuda:0 \
  --output output/relaysgd-validation/digits-balanced
```

For sensitivity runs, keep the same common options and use `--variants relaysgd`
with either `--relaysgd-topology path` or `--relaysgd-normalization initial`,
and a separate output directory. The validation record contains all three
commands, source hashes and frozen manifests. Each run was checked for finite
metrics, complete snapshots, paired capacities, matching sent/received bytes,
and exact distance-based population counts for every completed iteration.
