# GT-DSGD

`gt-dsgd` implements the adapt-then-combine (ATC) version of stochastic
gradient tracking in Algorithm 1 / Eq. (1) of
[Xin, Khan and Kar, IEEE Transactions on Signal Processing 2021](https://arxiv.org/abs/2008.04195),
arXiv v2. This baseline uses stable membership, an undirected fixed topology,
and constant learning rate. It is not DeTAG, a compressed method, or a
reproduction of the paper's complete experimental recipe.

## Exact update

Using node-row notation and a symmetric doubly stochastic matrix `W`, initialize
`y_0 = 0` and `g_{-1} = 0`. For logical iteration `k`:

```
g_k       = stochastic_gradient(x_k, minibatch_k)
z_k       = y_k + g_k - g_{k-1}
y_{k+1}   = W z_k
proposal  = x_k - learning_rate * y_{k+1}
x_{k+1}   = W proposal
```

Exactly one new local minibatch gradient is evaluated per iteration. The
previous **sampled gradient** is preserved as an explicit DAG output; it is
never recomputed with a new minibatch. Optional coupled weight decay adds
`weight_decay * x_k` to `g_k`, and that complete gradient is retained for the
next difference. Shared SGD momentum must be zero.

This specific GT-DSGD variant mixes both the corrected tracker and the model
proposal. Other gradient-tracking update orders have different finite-iteration
trajectories. In particular, from a common initialization its first step is
`x_1 = x_0 - learning_rate * W^2 g_0`. The implementation does not initialize
`y_0` using an additional uncharged gradient.

The reference property `mean(y_{k+1}) = mean(g_k)` holds at each completed
logical iteration, and the model mean takes the corresponding mean-gradient
step. These are numerical invariants, not claims that a local tracker equals
the exact gradient of a common model or that noisy tracking is exact at every
individual client.

## Protocol and timing

Each client computes its local gradient, broadcasts `z_k`, and waits for its
own gradient and all neighbors' **same-iteration** tracker messages. It forms
`y_{k+1}` and its proposal, broadcasts the proposal, then waits for all
neighbors' same-iteration proposals before publishing `x_{k+1}`. Separate
phase inboxes accept early packets and ignore duplicates; a completed phase
cannot run twice. Unexpected senders or phase identifiers are rejected.

Each phase sends one dense parameter-sized vector per neighbor. Both transfers
are charged separately, including a header, serialized round/phase metadata,
bandwidth contention and latency. The second phase depends on the first phase's
result. A client cannot begin its next gradient before finishing both phases.
There are no stale-cache substitutions or free tracker exchanges.

The default permits local progress once the required neighbor messages arrive;
it preserves synchronous iteration-index arithmetic without imposing a global
round barrier. `--synchronous` additionally waits for every participant before
starting the next logical iteration. Both modes produce the same per-iteration
numerical trajectories with the same seeded minibatches, but can have different
simulated completion times.

Ring, complete and seeded connected regular graphs are supported through
symmetric Metropolis weights, including positive self-weight. One client reduces
to ordinary SGD and incurs no network traffic. Two-client rings have one distinct
neighbor per client. Directed graphs, topology changes and churn are excluded.
The current implementation requires fully trainable floating parameters with
a common dtype; LoRA is rejected. Models' non-parameter buffers, including
BatchNorm statistics, remain local and follow the local gradient pass.

Gradient computation uses the existing trainer's seeded minibatches and
configured simulated training duration. Forming/mixing vectors has zero
simulated compute cost, consistent with the other maintained baselines.
Numerical gradient tasks use scalar CPU/CUDA execution; GPU training batching
does not handle this custom operation. Measured executor time is separate from
simulated protocol time.

Wire vectors have the actual parameter count times element size. As with QGM
and SWIFT, a generic `model_size_bytes` override does not replace that vector
size. The tracker and previous gradient remain explicit private DAG state;
neither is hidden in a process-local optimizer. Importable operation contracts
support worker execution and saved-DAG replay. `gt-dsgd-rounds.csv` records each
completed iteration's gradient, tracker and final model availability times.

`stop=rounds` caps each client's updates, while `stop=duration` keeps running
to the common time horizon. Fixed-time snapshots evaluate the most recently
completed local model, even when a client is in the middle of either exchange.
Individual round-based evaluation is also supported; global evaluation is
rejected. Snapshot accuracy averages clients uniformly. At a fixed logical
iteration, the doubly stochastic update targets the equal-client objective;
unequal partition sizes do not automatically imply equal-example weighting.

## Usage

In the `conflux-gpu` environment:

```sh
python main.py --local --algorithm gt-dsgd --dataset synthetic --model mlp \
  --peers 8 --rounds 10 --duration 30 --local-steps 1 --momentum 0 \
  --learning-rate .05 --gt-dsgd-topology ring \
  --test-method snapshot --test-period 5 \
  --work-dir output/gt-dsgd-validation/local
```

Use `--gt-dsgd-topology complete` or `--gt-dsgd-topology kreg --gt-dsgd-degree K`
for the other graphs. The synthetic/digits, CIFAR and Tiny ImageNet comparison
runners accept `gt-dsgd` and these same options. Explicitly select
`--local-steps 1 --momentum 0` where a runner defaults to a local-SGD block or
nonzero momentum. Settings serialize through the shared algorithm registry.

## Validation

`tests/test_gt_dsgd.py` compares five full-batch logistic-regression iterations
with independently computed gradients, tracker corrections, two matrix mixes
and model updates. The matrix is constructed independently for ring/complete
graphs. Tests cover uniform/skewed compute, optional global barriers and weight
decay, including exact wire bytes and received-message counts in both phases.
A separate known noisy oracle changes its gradient draw every iteration and
checks that the exact previous draw is retained. Both references verify the
tracker-mean conservation property.

Additional checks cover one-client SGD, latency and input availability, early
packets in both phases before local computation, duplicates after completion,
topology edge cases, duration stopping, streaming fixed-time snapshots, seeded
metrics, saved-DAG replay, immutable inputs and local buffers, CPU/CUDA model
and tracker equivalence, unsupported settings, CLI mapping and baseline aliases.
The public CLI also executes a real local CUDA smoke run. The
[validation record](validation/gt-dsgd.json) records the fresh checks and the
small digits screening matrix separately from implementation correctness.

On 3 November 2026 the full maintained suite passed **482 tests**, including
40 GT-DSGD cases, in `conflux-gpu` on an RTX 5000 Ada GPU.

## Short digits screening

The 24-run matrix compares GT-DSGD with the existing plain DSGD control at
three simulated seconds: eight clients, three seeds, uniform/heterogeneous
capacities and IID/Dirichlet α=.3 data. A 64-32-10 ReLU MLP uses batch 32 and
a predeclared learning rate .05, without momentum or weight decay. All eight
stored local models are evaluated on all 445 held-out test examples at 1, 2
and 3 seconds. These test results were not used to select hyperparameters.

Final local-model accuracy, averaged over clients and then reported as mean ±
sample standard deviation across the three seeds (percentage points):

| Capacity profile | Data | Plain DSGD | GT-DSGD |
| --- | --- | ---: | ---: |
| Uniform | IID | 87.00 ± 3.57 | 83.61 ± 4.78 |
| Uniform | Dirichlet α=.3 | 79.29 ± 2.12 | 77.56 ± 5.37 |
| Heterogeneous | IID | 33.06 ± 18.00 | 30.51 ± 16.57 |
| Heterogeneous | Dirichlet α=.3 | 22.98 ± 12.29 | 19.85 ± 13.17 |

GT-DSGD has lower mean accuracy in all four conditions in this short, untuned
screening. It schedules fewer gradient tasks because of the second exchange:
1798 versus 2144 for uniform capacities, and 368 versus 394.33 on average for
heterogeneous capacities. These counts include gradients in progress at the
horizon; completed iteration counts are in the per-run CSV. Its mean wire
traffic is 69.87 versus 41.83 MB for uniform capacities, and 14.19 versus 7.62 MB
for heterogeneous capacities, including partial final-horizon transfers.
This is execution evidence, not a conclusion about longer or tuned training.

Nominal bandwidth is 100 MB/s and latency is 2 ms. Uniform gradients take 9 ms.
The heterogeneous profile independently selects gradient times from 6, 15, 36
and 90 ms and assigns a quarter of nodes 4× nominal bandwidth and the rest
.5×. Exact capacities/data settings are paired across methods for every seed.
Mean capacity differs between profiles, so compare methods within a profile.
Measured scalar CUDA executor time is separate from simulated protocol time;
these jobs overlapped other validation work.

```sh
python scripts/validate_algorithms.py --variants gt-dsgd dsgd --dataset digits \
  --peers 8 --local-steps 1 --duration 3 --test-period 1 --bandwidth 100000000 \
  --latency-ms 2 --learning-rate .05 --profiles uniform heterogeneous \
  --partitioners iid dirichlet --seeds 0 1 2 --device cuda:0 \
  --output output/gt-dsgd-validation/digits
```

The [source/configuration record](validation/gt-dsgd.json),
[per-seed summary](validation/gt-dsgd-digits-summary.csv), and
[fixed-time curves](validation/gt-dsgd-digits-curves.csv) retain all runs.
