# DeTAG

`detag` implements plain DeTAG (Algorithms 2/3) from
[Lu and De Sa, ICML 2021](https://proceedings.mlr.press/v139/lu21a.html), using
the phase-gradient normalization clarified in their
[JMLR 2023 extension](https://jmlr.org/papers/v24/22-0044.html).
The [corrected conference manuscript](https://arxiv.org/abs/2006.08085v4) and
[authors' erratum](https://www.yucheng-lu.me/files/DeTAG_errata.pdf) were also
checked. The erratum changes the analysis, not Algorithms 2/3.
This is a fixed-topology, stable-membership specialization of plain SGD DeTAG;
DeTAGM, DeTAG*, local-SGD substitutions, warmup and dynamic phase lengths are
excluded.

DeTAG is currently deferred from the CIFAR-10/GNLeNet leaderboard after the
tested LR grid produced exploding or non-finite validation loss. See the
[investigation backlog](leaderboard.md#detag-numerical-stability) for the observed
outcomes, numerical checks and proposed replacement LR search.

## Phase equations

Each phase contains `R = phase_length` inner iterations. Model parameters `x`
stay fixed throughout those gradient evaluations. Initialize the tracker and
previous phase gradient to zero. Start accelerated gossip on a private model
replica and the current tracker, independently, with `z_previous = z_current`
for both channels at the **start of every phase**. Each gossip step is:

```
z_next = (1 + eta) * W z_current - eta * z_previous
```

Every inner iteration also draws a fresh seeded minibatch gradient at the fixed
local parameters, accumulating `gradient / R`. After all R compute/gossip
iterations, let `mixed_x` and `mixed_y` be the two accelerated gossip outputs:

```
G_current = (g_1(x) + ... + g_R(x)) / R
x_next    = mixed_x - learning_rate * mixed_y
y_next    = mixed_y + G_current - G_previous
G_previous = G_current
```

The model step uses the mixed tracker **before** the current phase's gradient
difference is injected. The new tracker, model and phase gradient are immutable
DAG outputs, used to start the next phase. From a common initialization and zero
tracker, the first phase keeps model parameters unchanged while it fills the
tracker; all gradient work and communication in that phase are charged.

This follows Algorithm 2's model-then-tracker update order. The `R=1, eta=0`
control keeps this pipeline and startup delay; its finite trajectories are not
identical to this repository's ATC GT-DSGD, which first computes and mixes the
current gradient correction and then updates/mixes the model. It should not be
treated as a byte-for-byte or iterate-for-iterate equivalence test between those
two published gradient-tracking update orders.

The ICML pseudocode writes a sum of phase gradients. The journal Algorithm 2
explicitly divides each contribution by R, matching its large-minibatch
interpretation. This implementation uses that **mean** convention, so increasing
R does not silently multiply the learning rate. Optional weight decay adds
`weight_decay * x` to every oracle gradient before division by R. Gradients are
evaluated at fixed parameters, not R successive local SGD iterates. Previous
sampled phase gradients are retained exactly, without redrawing old minibatches.
Model buffers follow each local forward pass and are not communicated.

The phase-end tracker mean equals the current phase-gradient mean. The model
mean takes a step using the preceding phase-gradient mean. Tests verify these
properties, in addition to each local numerical trajectory.

## Gossip and timing

Each inner iteration broadcasts a fused dense packet containing the current
model replica and tracker: two parameter-sized vectors per neighbor, plus one
header and serialized phase/step metadata. Received packets must match the
phase and inner-step index. Each client waits for its current gossip inputs
from every neighbor; early packets are buffered and duplicate packets cannot
rerun a step. No neighboring state is inspected or fetched without a transfer.

Gradient computation starts at the same simulated time as that gossip send.
Gossip may complete before the gradient, or the gradient before gossip; both
must complete before the next inner iteration starts. Thus a step's completion
time is the **maximum** of its gradient and gossip completion times, not their
sum. The final phase update waits for both parts of the final inner iteration.
Every gossip hop consumes the existing shared bandwidth and latency scheduler.
Intermediate replicas are not published as completed local training models.

Fixed-time snapshots use the last complete phase model, including initialization
when a client has not yet completed a phase. In this baseline `rounds`, individual
test intervals and the accuracy CSV's `round` column count **phases**; gradient
task counts count **minibatch oracle calls**. There are R calls per complete
phase. `stop=duration` keeps running to the common horizon; `stop=rounds` caps
local phases. `--synchronous` adds a global barrier between phases; by default
clients synchronize only through same-phase/step neighbor messages.

Ring, complete and seeded connected regular graphs use symmetric Metropolis
weights. By default, setup computes the non-consensus spectral radius
`lambda = ||W - 11^T/n||_2` and chooses
`eta = (1 - sqrt(1-lambda^2)) / (1 + sqrt(1-lambda^2))`, following Algorithm 3.
`--detag-eta` selects an explicit finite value in `[0,1)`, including zero for
unaccelerated gossip. Setup/topology discovery and spectrum computation have
zero simulated cost. One/two-client graph cases are supported.

`local_steps=1` and `momentum=0` are required; use the dedicated phase length
to control R. Churn, changing/directed graphs, LoRA and global evaluation are
rejected. Parameters must be fully trainable floating tensors with a common
dtype. Vector arithmetic has zero simulated cost; gradient tasks use the
existing scalar CPU/CUDA trainer, without GPU batching. Actual executor time
is separate from simulated protocol time.

Wire size is twice the actual dense parameter size. The generic
`model_size_bytes` record does not override this algorithm-specific packet
layout. Non-parameter buffers remain local, and gradients are not embedded in
network packets. Saved-DAG and worker execution use importable operation
contracts carrying explicit model, tracker, recurrence and gradient state.
`detag-mixing.json` records W, its spectrum bound, eta, R and packet bytes;
`detag-steps.csv` records both completion times for every completed inner step.

## Usage

In the `conflux-gpu` environment:

```sh
python main.py --local --algorithm detag --dataset synthetic --model mlp \
  --peers 8 --rounds 10 --duration 30 --local-steps 1 --momentum 0 \
  --learning-rate .05 --detag-phase-length 2 \
  --test-method snapshot --test-period 5 \
  --work-dir output/detag-validation/local
```

Options include `--detag-topology ring|complete|kreg`, `--detag-degree K`,
`--detag-phase-length R`, and `--detag-eta ETA`. The synthetic/digits, CIFAR
and Tiny ImageNet comparison runners all accept `detag` and these options.
Explicitly select one local step and zero shared momentum when their defaults
specify local SGD or momentum.

## Validation

`tests/test_detag.py` independently calculates four-phase logistic trajectories
for ring/complete graphs, R=1/2/3, eta=0/.2, skewed compute and optional global
phase barriers. It compares every accelerated gossip output, every accumulated
phase gradient, model and tracker; verifies fixed parameters during the phase,
recurrence resets, the charged startup phase and mean conservation; and checks
exact fused-packet bytes in both directions.

A known changing stochastic oracle independently checks averaging distinct
draws at fixed phase parameters and retaining the previous phase's exact mean.
Controlled fast/slow network tests assert both overlap directions, charged
gossip delay, and `completed=max(gradient_ready,gossip_ready)` for every step.
Other checks cover state/buffer immutability, early/duplicate/invalid messages,
topology/spectrum edge cases, snapshots, duration stopping, deterministic
metrics, saved-DAG replay and CPU/CUDA agreement across all states.

Fresh suite and screening results are recorded in
[the validation record](validation/detag.json). The short offline digits
screening is execution evidence, not a claim of paper reproduction or optimal
hyperparameters for any algorithm.

On 3 November 2026 the full maintained suite passed **537 tests**, including
52 DeTAG cases, in `conflux-gpu` on an RTX 5000 Ada GPU. The public CLI and
both large-dataset runners' option parsing were also checked.

## Short digits screening

The 48-run screening contains default DeTAG (`R=2`, spectrum-derived eta),
a predeclared `R=1, eta=0` DeTAG control, ATC GT-DSGD and plain DSGD. Eight
clients use a 64-32-10 ReLU MLP with batch 32, learning rate .05 fixed before
execution, zero momentum/decay and one minibatch per inner iteration. Both
DeTAG configurations are reported; test metrics were not used to select a
configuration or tune its learning rate.

Every local model is evaluated on all 445 held-out digits examples at 1, 2 and
3 simulated seconds. Final local-model accuracy, averaged over eight clients
and then reported as mean ± sample standard deviation across seeds 0/1/2
(percentage points):

| Capacity profile | Data | Plain DSGD | GT-DSGD | DeTAG R=1, eta=0 | DeTAG R=2, auto eta |
| --- | --- | ---: | ---: | ---: | ---: |
| Uniform | IID | 87.00 ± 3.57 | 83.61 ± 4.78 | 89.81 ± 2.88 | 75.99 ± 8.90 |
| Uniform | Dirichlet α=.3 | 79.29 ± 2.12 | 77.56 ± 5.37 | 86.18 ± 1.66 | 65.82 ± 3.38 |
| Heterogeneous | IID | 33.06 ± 18.00 | 30.51 ± 16.57 | 33.73 ± 19.30 | 16.59 ± 8.18 |
| Heterogeneous | Dirichlet α=.3 | 22.98 ± 12.29 | 19.85 ± 13.17 | 21.62 ± 13.45 | 15.72 ± 8.95 |

The unaccelerated R=1 control has higher mean accuracy than default R=2 in
all four conditions. It outperforms the other controls on uniform capacities
in this screening; heterogeneous results are noisy and show no consistent
gain over plain DSGD. Phase averaging, fewer model updates and the startup
pipeline matter over such a short horizon. These measurements cannot establish
a longer-run ranking or the benefit of tuned acceleration on larger sparse
networks.

Both DeTAG configurations schedule 2672 gradient tasks under uniform capacity,
versus 1798 for GT-DSGD and 2144 for DSGD. Heterogeneous means are 413, 368 and
394.33 respectively. Tasks include oracle calls in progress at the horizon.
Completed model-update counts differ because R=2 makes one parameter update
per two oracle calls. Uniform wire traffic is 103.56 MB for either DeTAG
configuration, versus 69.87 MB for GT-DSGD and 41.83 MB for DSGD, including
partial transfers. Overlap increases throughput while still consuming the
bandwidth of both dense channels.

The nominal network capacity is 100 MB/s and latency is 2 ms. Uniform gradients
take 9 ms. Heterogeneous capacities independently select gradient times from
6/15/36/90 ms and assign a quarter of clients 4× nominal bandwidth and the rest
.5×. Capacities/data settings are exactly paired across methods within each
seed; mean capacity differs across profiles. Measured scalar CUDA execution
times overlap validation work and are separate from simulated protocol time.

Reproduce the default and comparison matrix with:

```sh
python scripts/validate_algorithms.py --variants detag gt-dsgd dsgd --dataset digits \
  --peers 8 --local-steps 1 --duration 3 --test-period 1 --bandwidth 100000000 \
  --latency-ms 2 --learning-rate .05 --profiles uniform heterogeneous \
  --partitioners iid dirichlet --seeds 0 1 2 --detag-phase-length 2 \
  --device cuda:0 --output output/detag-validation/digits-r2
```

For the R=1 control, use `--variants detag --detag-phase-length 1 --detag-eta 0`
with the same common options and a separate output directory. The
[validation record](validation/detag.json),
[per-seed summaries](validation/detag-digits-summary.csv) and
[fixed-time curves](validation/detag-digits-curves.csv) preserve all configurations.
