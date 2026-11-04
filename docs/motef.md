# Base MoTEF

`motef` implements Algorithm 1 of
[Islamov, Gao and Stich, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/39b77b5e422b4e070e2811b73ea9bcf7-Abstract-Conference.html),
combining momentum tracking with compressed model/tracker estimates and error
feedback. The [reference repository](https://github.com/mlolab/MoTEF) is pinned
at `10919bd38e093897329d1b096ba037701084d635`. MoTEF-VR is a separate [`motef-vr` algorithm](motef-vr.md).

## Update and initialization

Using one row per client, let x be the local model, m its EMA gradient momentum,
v the momentum tracker, h the compressed model estimate, and g the compressed
tracker estimate. With symmetric Metropolis matrix W:

```
x_next = x + gamma * (W h - h) - learning_rate * v
q_h    = Compress(x_next - h)
h_next = h + q_h
m_next = (1-lambda) * m + lambda * Gradient(x_next)
v_next = v + gamma * (W g - g) + m_next - m
q_g    = Compress(v_next - g)
g_next = g + q_g
```

Both consensus corrections read the **old** estimates. The model takes a step
with the old tracker; the fresh minibatch gradient is evaluated at x_next.
The exact previous momentum is retained, without redrawing its gradient.
`momentum_weight` is lambda, the weight of the fresh gradient, rather than
the old-momentum coefficient of an ordinary SGD optimizer.

Initialization computes one fresh local minibatch gradient at the shared seeded
model, with its normal simulated training cost. Set `v0=m0=that gradient` and
`h0=g0=0`. The initial weighted estimate memories are also zero and require no
network exchange. The authors' synthetic `beerM` solver uses this consistent
tracker/momentum initialization with a full local gradient; here a single seeded
minibatch is used. Algorithm 1 accepts initial estimates as inputs; this cold
estimate choice differs from the appendix's warm initialization used to simplify
its initial Lyapunov bound. No initial neighbor gradient or estimate is supplied
for free. A minibatch initialization is not a claim to reproduce a large initial
batch experiment or its simplified initial error bound.

The repository's main `BEER_Momentum.init` instead sets V to a gradient and M
to zero. We do not adopt that mismatch: the paper's tracking invariant requires
`V0=M0`. Our reference audit executes the main **update recurrence** with the
declared consistent initial state, and independently runs the synthetic solver
with its own consistent initialization. It also uses the synthetic helper's
largest-absolute-value Top-k; the main repository helper partitions by signed
value, rather than implementing the paper's magnitude-Top-k rule.

At every common completed iteration, `mean(v_next)=mean(m_next)` and the model
mean takes a step with the previous momentum mean. Compression residuals are
represented by `x-h` and `v-g`. Each client stores its own h/g and the weighted
neighbor sums `W h`/`W g`, updating those memories only through received deltas.
Thus there are six persistent parameter vectors (m, v, h, g, and two weighted
memories) in addition to the model, independent of node degree.

## Communication and supported scope

Top-k and unscaled random-k operate on the **whole flattened parameter vector**.
They retain `max(1, ceil(D*keep_ratio))` coordinates. Top-k selects largest
magnitudes; ties may use any valid selection. Random-k uses seeded sampling
without replacement and is not scaled by D/k. Its two channels have distinct
seed streams. `identity` sends the complete dense residual in each channel.
Even zero-valued selected coordinates consume their stated wire bytes.

A completed local gradient produces one fused packet containing q_h and q_g.
Each sparse channel carries k int64 coordinates and k parameter-dtype values;
the combined payload is `2*k*(8+element_size)` bytes. Identity uses
`2*parameter_bytes`. There is one generic control header and serialized round
metadata per neighbor packet. At keep_ratio=1, sparse modes still carry their
indices and can cost more than identity. The generic `model_size_bytes` setting
does not override this explicit layout.

The implementation deliberately batches both deltas into one send after the
gradient completes. It does not overlap the model-delta transfer with that
gradient. Clients wait for their local gradient and all same-iteration neighbor
packets before updating memories, publishing the completed model and starting
the next iteration. Early packets are buffered; duplicate packets/completions
cannot repeat work. The default has neighbor dependencies without a global
barrier; `--synchronous` additionally imposes an iteration-start global barrier.

Transfers use the shared upload/download contention and latency scheduler. No
neighbor model, tracker or gradient is inspected without its transfer. Topology
construction, vector operations, compression and aggregation have zero simulated
compute cost. Gradient tasks have explicit simulated duration and scalar CPU/CUDA
numerical execution, without generic SGD batching. Actual executor wall time is
separate from simulated protocol time. All models, estimates, packets and
momentum state are immutable DAG outputs with importable operation contracts.

Fixed ring, complete and seeded connected regular graphs use symmetric
Metropolis weights including local retention. One/two-client cases are supported.
For one client, MoTEF reduces to EMA-gradient SGD, with one charged initialization
gradient and one fresh gradient after each model step. Lambda=1 removes EMA
smoothing; it does not change the algorithm to this repository's ATC GT-DSGD
update ordering.

`local_steps=1`, shared `momentum=0`, stable membership and fully trainable,
nonempty floating parameters of one dtype are required. Use the dedicated
lambda to control MoTEF momentum. Coupled weight decay optionally adds
`weight_decay*x_next` to the fresh gradient. Non-parameter buffers stay local
through gradient forward passes. Directed/changing graphs, churn, loss/recovery,
LoRA, quantization compressors and multi-step local-SGD extensions are
excluded. Numerical ranges on gamma/lambda do not promise convergence for all
learning rates and compression ratios.

Snapshots evaluate the last completed iteration, or initialization before any
completion. `rounds` and individual test intervals count MoTEF iterations,
excluding the initial gradient. `stop=duration` runs to the common
horizon; `stop=rounds` caps local iterations. `motef-rounds.csv` records gradient
and communication completion; round 0 identifies the charged bootstrap, with
no model update. `motef-mixing.json` records topology, W, packet bytes, selected
parameters and initialization. Scheduled gradient counts include bootstrap and
work still in progress at the horizon.

## Usage

In `conflux-gpu`:

```sh
python main.py --local --algorithm motef --dataset synthetic --model mlp \
  --peers 8 --rounds 10 --duration 30 --local-steps 1 --momentum 0 \
  --learning-rate .05 --motef-compressor topk --motef-keep-ratio .01 \
  --motef-consensus-stepsize .1 --motef-momentum-weight .1 \
  --test-method snapshot --test-period 5 --work-dir output/motef-validation/local
```

The synthetic/digits, CIFAR and Tiny ImageNet runners accept `motef` and
`--motef-topology ring|complete|kreg`, `--motef-degree K`,
`--motef-compressor topk|randomk|identity`, `--motef-keep-ratio`,
`--motef-consensus-stepsize` and `--motef-momentum-weight`. Defaults are starting
settings, not paper-tuned hyperparameters. Explicitly select one local step and
zero shared momentum when runner defaults differ.

## Validation

`tests/test_motef.py` independently calculates five-iteration logistic trajectories
for three compressors, two graphs, two lambda values, skewed compute and optional
global barriers. It checks every model, m/v, h/g, weighted estimate memory and
decoded packet, plus both mean invariants and exact wire traffic. A known
changing oracle verifies gradients at the next model and persistence of the exact
previous momentum. Other checks cover compression layout/contractivity, charged
startup, causal delivery, duplicate/early messages, one-client EMA-SGD, small
graphs, local buffers, immutable inputs, duration stopping, snapshots,
reproducibility, streaming, saved-DAG replay and CPU/CUDA agreement.

`tests/check_motef_reference.py` executes the pinned main optimizer's
`BEER_Momentum.update` unchanged with the declared initial state and actual
synthetic magnitude-Top-k helper. Four clients run eight iterations on ring and
complete graphs, identity/Top-k and lambda=.2/1. It compares every state and
compressed delta. It also executes `synthetic/solvers.py:beerM` unchanged on the
same non-IID quadratic objectives and compares the final model. Maximum error
is `2.00e-15` across states/packets and `3.33e-16` against the synthetic solver's
final model in float64. This excludes the main initializer, signed-order
compressor, training-data pipeline and deployment.

```sh
git clone https://github.com/mlolab/MoTEF.git /tmp/motef-reference
git -C /tmp/motef-reference checkout 10919bd38e093897329d1b096ba037701084d635
python tests/check_motef_reference.py /tmp/motef-reference
python -m pytest -q
```

Suite, reference, CLI and screening evidence is preserved in
[the validation record](validation/motef.json),
[per-seed summaries](validation/motef-digits-summary.csv) and
[fixed-time curves](validation/motef-digits-curves.csv).

On 3 November 2026 the full maintained suite passed **724 tests**, including
65 MoTEF-specific cases, in `conflux-gpu` on an RTX 5000 Ada GPU. The public
CUDA CLI and both large-dataset runners' option parsing also passed.

## Short digits screening

The 54-run screening uses eight clients on the same ring/Metropolis matrix,
a 64-32-10 ReLU MLP, batch 32, one fresh gradient per iteration, learning rate
.05, zero shared momentum/decay, and three simulated seconds. MoTEF uses its
default Top-k keep ratio .01, gamma=.1 and lambda=.1. CHOCO uses its default
per-tensor Top-k .01 and gamma=.1; GT-DSGD uses dense tracking. All settings
were fixed before execution and test metrics did not select configurations.

Every local model is evaluated on all 445 held-out digits examples at 1, 2 and
3 seconds. Final accuracy averages over eight local models, then reports mean
± sample standard deviation across seeds 0/1/2, in percentage points:

| Nominal bandwidth | Capacity | Data | MoTEF | CHOCO-SGD | GT-DSGD |
| --- | --- | --- | ---: | ---: | ---: |
| 100 MB/s | Uniform | IID | 85.93 ± 4.01 | 85.29 ± 4.40 | 83.60 ± 4.78 |
| 100 MB/s | Uniform | Dirichlet α=.3 | 78.53 ± 1.28 | 51.57 ± 3.29 | 77.56 ± 5.37 |
| 100 MB/s | Heterogeneous | IID | 31.83 ± 15.65 | 31.93 ± 14.46 | 30.51 ± 16.57 |
| 100 MB/s | Heterogeneous | Dirichlet α=.3 | 17.04 ± 4.80 | 15.18 ± 5.16 | 19.85 ± 13.17 |
| 200 kB/s | Uniform | Dirichlet α=.3 | 54.80 ± 2.46 | 39.27 ± 6.62 | 10.41 ± 0.72 |
| 200 kB/s | Heterogeneous | Dirichlet α=.3 | 16.10 ± 3.11 | 14.85 ± 4.61 | 8.87 ± 2.21 |

MoTEF has higher mean accuracy than CHOCO in the uniform non-IID conditions
here, and comparable means on IID data. Under heterogeneous capacities the
ample-bandwidth results are noisy and show no consistent advantage over
GT-DSGD. The congested non-IID means favor MoTEF, but heterogeneous final
accuracies remain low. This short, untuned screening does not establish a
larger-model ranking or reproduce the paper's experiments.

Under uniform ample capacity, MoTEF sends 2.99 MB and completes 2168 model
updates, CHOCO sends 1.79 MB and completes 2176, and GT-DSGD sends 69.87 MB
and completes 1790. Scheduled gradient tasks are 2184, 2184 and 1798; MoTEF's
count includes eight startup gradients as well as work in progress. Under
uniform congestion, completed updates are 1330, 1584 and 96, with 1.84 MB,
1.31 MB and 3.85 MB sent respectively. All traffic includes partial transfers
at the horizon. Per-seed records preserve the heterogeneous counts as well.

The model has D=2410 float32 parameters (9640 bytes). MoTEF keeps 25 coordinates
in each channel and fuses them into a 600-byte payload. CHOCO happens to keep
25 coordinates across its separate tensors, for a 300-byte payload. Both add
one header/metadata per neighbor; GT-DSGD sends two sequential 9640-byte dense
vectors with separate headers/phase metadata. Global versus per-tensor
selection, two versus one channels, initialization and update order differ.
This is not an isolated ablation of momentum tracking or compression, and a
dense MoTEF control was not benchmarked in this screening.

Latency is 2 ms. Uniform gradients take 9 ms. Heterogeneous clients independently
receive gradient times from 6/15/36/90 ms; a quarter have 4× nominal bandwidth
and the rest .5×. Data settings and capacities are exactly paired within
condition/profile/seed; mean capacity differs across profiles. Actual scalar
CUDA execution times overlap other validation work and are separate from
simulated protocol time.

Reproduce the ample-bandwidth matrix with:

```sh
python scripts/validate_algorithms.py --variants motef choco gt-dsgd --dataset digits \
  --peers 8 --local-steps 1 --duration 3 --test-period 1 --bandwidth 100000000 \
  --latency-ms 2 --learning-rate .05 --profiles uniform heterogeneous \
  --partitioners iid dirichlet --seeds 0 1 2 --device cuda:0 \
  --output output/motef-validation/digits-ample
```

For congestion, use `--partitioners dirichlet --bandwidth 200000` and a separate
output directory. The validation record preserves both commands, manifests,
source hashes and paired capacities. Every run was checked for finite metrics,
all local snapshots, matching sent/received bytes, symmetric doubly stochastic
MoTEF matrices, charged bootstrap and contiguous completed iterations within
the horizon.
