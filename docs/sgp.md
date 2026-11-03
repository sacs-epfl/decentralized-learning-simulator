# Stochastic Gradient Push

`sgp` implements plain SGP, Algorithm 1 of
[Assran et al., ICML 2019](https://proceedings.mlr.press/v97/assran19a.html).
Its [supplement](https://proceedings.mlr.press/v97/assran19a/assran19a-supp.pdf)
specifies the rotating exponential graph. The authors'
[reference code](https://github.com/facebookresearch/stochastic_gradient_push)
was checked at revision `e682064d3f0a6216868f8528f804afe091af19b8`.
This is a fresh implementation; the repository's historical `pushsum` branch
is a separate experimental protocol and was not used as the SGP implementation.

## Numerical contract

Each client maintains a parameter numerator `x`, scalar mass `w`, and the
debiased model `z=x/w`. Initially `w=1` and `x=z` at a shared seeded model.
In iteration k:

```
g_i       = stochastic_gradient(F_i, z_i)
x_half_i  = x_i - learning_rate * g_i
x_next_i  = sum_j P[i,j] * x_half_j
w_next_i  = sum_j P[i,j] * w_j
z_next_i  = x_next_i / w_next_i
```

The fresh minibatch gradient is evaluated at **z** and applied to **x**.
Training z with ordinary SGD and then multiplying by w would incorrectly
multiply the gradient by w. Numerators persist as explicit numerical DAG
outputs; they are not replaced by debiased models between iterations.
Optional coupled weight decay adds `weight_decay*z` to the oracle gradient;
the pinned reference comparison uses zero decay. This convention regularizes
the objective evaluated at z, rather than applying optimizer decay to x.

For each sender j, every outgoing edge and its local retained share receive
weight `1/(out_degree(j)+1)`. Thus P is **column stochastic**. The same weights
apply to the numerator and scalar mass. Incoming weighted vectors are summed,
without receiver-side averaging or a second application of sender weights.
The mass at each client is updated only from its retained share and packets
that have actually arrived. No global mass calculation supplies protocol state.

At a common completed logical iteration, total scalar mass stays N and total
numerator changes by minus the learning rate times the sum of local gradients.
Individual masses generally differ from one on irregular directed graphs.
Without gradients, dividing by the mass recovers the unweighted origin average
despite a nonuniform stationary distribution. During optimization, debiased
local models need not be equal, and their arithmetic mean is not generally the
mean numerator. At a wall-clock snapshot, clients can have completed different
iterations, so summing their last published masses is not a conservation test.

Parameters must be fully trainable floating tensors with a common dtype.
Non-parameter buffers follow each local gradient forward pass and are not mixed.
Source models, gradients, incoming packets and previous numerators are immutable.

## Directed topology and timing

`--sgp-topology` selects:

| Value | Communication schedule, excluding local retention |
| --- | --- |
| `exponential` (default) | Every client sends to `(i+offset) mod N`; offset cycles through `1,2,4,...,2^floor(log2(N-1))`, following Appendix A's one-peer schedule. |
| `ring` | Directed cycle, sending only to `(i+1) mod N`. |
| `irregular` | Directed cycle plus the shortcut `0→2` when N≥3, creating unequal outgoing degrees and nontrivial masses. |
| `complete` | Every other client; dense parallel-SGD control. |

One/two-client cases are supported. For N=1 no network messages are sent and
SGP reduces to plain SGD. The irregular graph is an additional Algorithm 1
topology, not the paper's experimental topology. Exponential phases can be
disconnected individually (e.g. offset 4 on 8 clients); their periodic union
contains the directed cycle and is strongly connected. For powers of two,
one full schedule cycle exactly averages static values in exact arithmetic.
Arbitrary-size schedules remain jointly strongly connected, but do not have
that finite-step exact-averaging property in general.

Every iteration performs one local gradient and then sends its weighted
numerator and mass. A client waits for its gradient and every **incoming**
peer's same-iteration packet before publishing z and proceeding. It does not
wait for an outgoing recipient's gradient. Early messages are buffered by
iteration; expected senders follow that iteration's graph phase. Duplicates
cannot complete an iteration or send a packet twice. The default has no global
barrier; `--synchronous` adds one before each iteration. Blocking logical
iterations still have unequal wall-clock completion times under heterogeneity.

Each outgoing edge carries one actual dense parameter-sized numerator plus an
8-byte float64 mass, fused into one message, with one generic header and
serialized round metadata. Local retention uses no network. Scalars are sent
even on regular graphs, where the reference can optimize them away. The generic
`model_size_bytes` setting does not override this packet layout. Transfers use
the shared upload/download contention and latency scheduler. No remote state
is accessed without its corresponding transfer.

Graph/membership setup is known and has zero simulated cost. Vector operations
and scalar bookkeeping also have zero simulated cost. Gradient tasks use the
existing scalar CPU/CUDA oracle and explicit simulated gradient durations;
this custom operation does not use the executor's generic SGD batching path.
Numerical executor wall time does not feed back into simulated protocol time.

This baseline requires `local_steps=1`, `momentum=0`, stable membership, and
individual or snapshot evaluation. Overlap SGP (Algorithm 2), momentum/Adam,
multiple peers per exponential phase, arbitrary graph discovery, message loss,
churn and LoRA are excluded. Ordinary network latency is charged and waited
for; it is not the stale-message update rule of Overlap SGP.

`rounds` and individual test intervals count SGD/PushSum iterations.
`stop=duration` continues to the common horizon; `stop=rounds` caps iterations.
Snapshots evaluate the last completed debiased model, including initialization
before the first completed iteration. `sgp-mixing.json` records all graph
phases, sender-column matrices and packet bytes. `sgp-rounds.csv` records
gradient readiness, iteration completion and scalar mass.

## Usage

In `conflux-gpu`:

```sh
python main.py --local --algorithm sgp --dataset synthetic --model mlp \
  --peers 8 --rounds 10 --duration 30 --local-steps 1 --momentum 0 \
  --learning-rate .05 --sgp-topology exponential \
  --test-method snapshot --test-period 5 --work-dir output/sgp-validation/local
```

The synthetic/digits, CIFAR and Tiny ImageNet runners accept `sgp` and
`--sgp-topology exponential|ring|irregular|complete`. Explicitly select one local
step and zero momentum when shared runner defaults differ.

## Validation

`tests/test_sgp.py` constructs sender-column matrices independently and checks
every model, numerator, weighted packet and scalar mass for four schedules,
weight decay on/off, compute skew and optional global barriers. It checks total
numerator/mass conservation at common iterations and exact fused-packet traffic.
A known changing oracle tests evaluating at z and updating x without scaling
gradients by w. A static averaging test verifies correction of a nonuniform
stationary limit. Other tests cover complete-graph centralized-SGD equivalence,
one-worker SGD, powers-of-two exact averaging, arbitrary-size graph schedules,
directed causal waiting, early/duplicate/wrong-phase messages, immutable inputs,
local buffers, snapshots, duration stopping, seeded reproducibility, streaming
execution, saved-DAG replay and CPU/CUDA agreement across all numerical states.

`tests/check_sgp_reference.py` executes the pinned authors' `UniformMixing`,
`PushSum.mix`, `ps_numerator` and `unbias` arithmetic unchanged with mocked
broadcast transport, graph discovery and buffer allocation. Four clients run
eight steps for each of the four graph schedules, using non-IID quadratic
gradients and plain PyTorch SGD. It compares outgoing weighted numerator/mass
messages, resulting numerators, masses and debiased models. The reference's
regular-graph scalar-omission optimization is disabled to exercise its general
path. Float64 parameters and default weights give maximum parameter and mass
errors of `2.22e-16`. This validates arithmetic, not MPI/thread execution.

```sh
git clone https://github.com/facebookresearch/stochastic_gradient_push.git /tmp/sgp-reference
git -C /tmp/sgp-reference checkout e682064d3f0a6216868f8528f804afe091af19b8
python tests/check_sgp_reference.py /tmp/sgp-reference
python -m pytest -q
```

Suite, CLI and screening evidence is preserved in
[the validation record](validation/sgp.json),
[per-seed summaries](validation/sgp-digits-summary.csv) and
[fixed-time curves](validation/sgp-digits-curves.csv).

On 3 November 2026 the full maintained suite passed **656 tests**, including
63 SGP-specific cases, in `conflux-gpu` on an RTX 5000 Ada GPU. The public
CUDA CLI and option parsing in both large-dataset runners also passed.

## Short digits screening

The 48 runs use eight clients, a 64-32-10 ReLU MLP, batch 32, one gradient per
iteration, learning rate .05, zero momentum/decay, and three simulated seconds.
Every client's debiased model is evaluated on all 445 held-out digits examples
at 1, 2 and 3 seconds. Graph configurations, bandwidth conditions and the common
learning rate were fixed before execution; all results are reported without
using test metrics to select a configuration.

At ample nominal bandwidth (100 MB/s), final accuracy averages over eight local
models, then reports mean ± sample standard deviation across seeds 0/1/2, in
percentage points:

| Capacity | Data | Ring DSGD | Exponential SGP (default) | Irregular SGP |
| --- | --- | ---: | ---: | ---: |
| Uniform | IID | 87.00 ± 3.57 | 87.06 ± 3.53 | 86.95 ± 3.53 |
| Uniform | Dirichlet α=.3 | 79.29 ± 2.12 | 80.90 ± 2.96 | 78.64 ± 2.95 |
| Heterogeneous | IID | 33.06 ± 18.00 | 34.41 ± 19.10 | 33.76 ± 19.55 |
| Heterogeneous | Dirichlet α=.3 | 22.98 ± 12.29 | 22.55 ± 13.54 | 22.20 ± 13.31 |

Default SGP has similar mean accuracy to ring DSGD in this short screening,
with large seed variation under heterogeneity. Under uniform capacity it sends
21.03 MB versus DSGD's 41.83 MB, about half as much traffic. It completes 2160
local model updates versus 2136, scheduling 2168 versus 2144 gradients including
work in progress at the horizon. The irregular graph sends 23.66 MB and completes
2152 updates. Under heterogeneous capacity, default SGP sends 3.93 MB and
completes 399.67 updates on average, versus 7.62 MB and 386.33 for DSGD.

The predeclared congested condition lowers nominal bandwidth to 200 kB/s and
uses only Dirichlet α=.3 partitions:

| Capacity | Ring DSGD accuracy | Exponential SGP accuracy | DSGD completed updates | SGP completed updates |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 14.68 ± 3.92 | 24.36 ± 9.86 | 190 | 400 |
| Heterogeneous | 10.87 ± 0.46 | 13.99 ± 5.02 | 93 | 174 |

SGP completes more iterations and has higher mean accuracy in these congested
conditions, but all final accuracies remain low. Its uniform-capacity traffic
is 3.91 MB versus DSGD's 3.75 MB: less communication per iteration permits more
iterations, so total fixed-time traffic need not decrease. Heterogeneous means
are 1.73 MB and 1.88 MB respectively. Partial final-horizon transfers count.

Default exponential SGP sends eight directed packets per complete iteration,
irregular SGP nine, and ring DSGD sixteen. Dense parameters occupy 9640 bytes;
SGP adds an 8-byte mass, before the header/metadata. These comparisons change
graph schedules and message counts as well as the algorithm. They cannot
isolate scalar debiasing, which is nontrivial only in the irregular runs here.
No claim establishes a larger-model ranking, the benefit of Overlap SGP, or
reproduction of the paper's ImageNet/translation experiments.

Latency is 2 ms. Uniform gradients take 9 ms. Heterogeneous clients independently
receive gradient times from 6/15/36/90 ms; a quarter have 4× nominal bandwidth
and the remainder .5×. Data settings and capacities are exactly paired within
condition/profile/seed; mean capacity differs across profiles. Executor wall
times overlap other validation work and are separate from protocol time.

Reproduce the ample-bandwidth comparison with:

```sh
python scripts/validate_algorithms.py --variants sgp dsgd --dataset digits \
  --peers 8 --local-steps 1 --duration 3 --test-period 1 --bandwidth 100000000 \
  --latency-ms 2 --learning-rate .05 --profiles uniform heterogeneous \
  --partitioners iid dirichlet --seeds 0 1 2 --device cuda:0 \
  --output output/sgp-validation/digits-default
```

For the irregular sensitivity run use `--variants sgp --sgp-topology irregular`.
For congestion use `--variants sgp dsgd --partitioners dirichlet --bandwidth 200000`.
Use separate output directories. The validation record preserves all commands,
manifests and source hashes. Checks cover finite metrics, all local snapshots,
paired capacities, matching sent/received counters, column-stochastic matrices,
joint strong connectivity and independently predicted masses at every completed
SGP iteration.
