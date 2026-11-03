# SWIFT

`swift` implements the one-minibatch update in Algorithm 1 of
[Bornstein et al., ICLR 2023](https://arxiv.org/abs/2210.14026).
The [authors' implementation](https://github.com/umd-huang-lab/SWIFT) supplies
the persistent-cache convention and the uniform-CIS regular-graph CCS weights.
This is a stable-membership, regular-topology specialization, not a reproduction
of their MPI experiments or training recipe.

## Update and message ordering

Each client begins at the common seeded model and has a private cache slot per
neighbor, initially containing that same model. With local counter `c=1`:

1. Broadcast the current **pre-update** model to every neighbor.
2. Compute one minibatch gradient `g` at those local parameters, while network
   transfers continue independently.
3. If `c % averaging_interval == 0`, read the latest **delivered** model from
   each neighbor's cache and average their parameters with the local parameters.
   Otherwise retain the local parameters.
4. Subtract `learning_rate * g` from the result, publish the model, increment
   the counter, and immediately begin the next step.

Thus the numerical rule is `x_next = sum_j w_ij * cached_x_j - lr * g(x_i)`
on averaging steps, and ordinary SGD otherwise. Coupled weight decay, when
requested, is evaluated at the pre-average local parameters as part of `g`.
The gradient is not recomputed at the mixed parameters and neighbors' gradients
are not averaged. Model buffers, including BatchNorm statistics, stay local and
follow the client's gradient pass. Numerical operations preserve input models.

The averaging interval `H` corresponds to the paper's communication set `C_s`
with `s=H-1`; the first average occurs at step `H`. Broadcasts occur at **every**
step even outside that set, following Algorithm 1 line 7 and the authors'
`communicate` method. The simulator follows the paper's broadcast-before-gradient
ordering; the authors' training loop issues that broadcast after backward and
before the optimizer step. This affects which messages arrive during a step.

Receivers retain the greatest model version delivered by each neighbor. Repeated
or out-of-order versions cannot overwrite a newer cache entry. Incoming models
only change the mailbox; they never interrupt a gradient pass or change a
published local model. There are no inbox barriers, pairwise handshakes, or
network completion dependencies in the training loop. Messages queued before a
client reaches its round limit continue to consume network capacity, and stopped
clients can still receive models without taking further optimization steps.

## Communication weights and influence

Ring (including one/two-client edge cases), complete, and seeded connected
`kreg` graphs are supported. CCS is precomputed as a zero-cost setup operation;
its neighbor discovery and setup exchanges are not simulated. Arbitrary irregular
graphs and changing membership/topology are excluded.

All supported graphs are regular. For uniform client influence scores (CIS),
the weights match the authors' implementation: `1 / (degree + 1)` for self and
every neighbor. For positive nonuniform scores `p`, normalize them and reserve
`1/n` self-weight, as in CCS. Let `m_i` be the sum of CIS over the closed
neighborhood (including self). The equal-degree CCS specialization is
`w_ij = (1 - 1/n) * p_j / max(m_i, m_j)` on an edge, with the diagonal set to
the remaining mass. Using receiver-row notation, it obeys
`p_i * w_ij = p_j * w_ji`. Consequently the expected active-client mixing matrix
is symmetric and doubly stochastic when activation probabilities are `p`.
Tests verify these invariants and known closed-form weights. The closed
neighborhood includes self, consistent with the authors' uniform implementation
and the paper's remaining-mass assignment to neighbors **and self**.

Default CIS are **uniform**, matching the paper's experiments. Supply
`--swift-influence p0 p1 ...` for an explicitly chosen vector. The event simulator
lets every client train continuously at its configured speed; it does not sample
clients from CIS or estimate actual update rates. Under unequal speeds, uniform
CIS may differ from empirical activation frequencies. The expected-matrix result
then does not directly apply to the realized schedule. Matching CIS to expected
update rates still gives a rate-weighted gradient objective; it does **not**
establish equal-client or equal-example optimization. There is no inverse-rate
gradient correction hidden in this baseline.

## Timing, payloads and supported settings

`local_steps=1` and shared SGD `momentum=0` are required. Use the dedicated
averaging interval for periodic mixing. `--synchronous`, churn, LoRA and global
evaluation are rejected; use individual or fixed-time snapshot evaluation.
Parameters must be unfrozen floating tensors sharing one dtype.
`rounds` caps each client's local updates under `stop=rounds`; `stop=duration`
allows continuous updates to the common simulated horizon. Computation and
bandwidth must be positive. Gradient tasks use the existing seeded trainer and
scalar CPU/CUDA execution. GPU training batching currently does not handle
gradient-only operations.

Each send contains a dense parameter vector, as in the authors' implementation
and the plain DSGD control. Its wire size is the actual parameter count times
element size, plus a control header and serialized version metadata. Buffers
and gradients are never transmitted. `model_size_bytes` remains the generic
simulator model-size record; it does not override this vector's size. All
broadcasts share the bandwidth/latency scheduler and count
partial wire traffic at the final horizon. There is no free latest-model lookup,
coalescing, or cancellation: insufficient bandwidth can accumulate in-flight
messages and leave caches stale while training continues. Averaging and applying
the already computed gradient have zero simulated cost, as in the other baselines.
Measured executor time is separate from simulated training time.

`swift-weights.json` records the normalized CIS and receiver-row matrix.
`swift-rounds.csv` records completed local counters/times, averaging steps, and
maximum positive **counter-version lag**. This counter difference is diagnostic;
it is not a wall-clock age or a bounded-delay claim across unequal client rates.
All gradient/model dependencies are explicit DAG references, and the numerical
update operation carries its importable contract for saved-DAG and worker execution.

## Usage

```sh
conda run -p /mnt/data/mdevos/conda/envs/conflux-gpu python main.py --local \
  --algorithm swift --dataset synthetic --model mlp --peers 8 \
  --local-steps 1 --momentum 0 --learning-rate .05 \
  --swift-averaging-interval 2 --stop duration --duration 10 \
  --test-method snapshot --test-period 2 --work-dir output/swift-validation/local
```

The maintained synthetic/digits, CIFAR and Tiny ImageNet comparison runners
accept the `swift` baseline and `--swift-*` options. Their shared optimizer
defaults still apply: explicitly select one local step and zero momentum.
JSON/YAML experiment files accept `swift_influence: [1, 2, ...]` as a list;
the CLI accepts `--swift-influence 1 2 ...`.

## Validation

`tests/test_swift.py` checks independent multistep full-batch logistic gradients
and updates against causal delivery traces, including periodic averaging and
unequal compute speeds; the expected mixing matrix; exact completed-transfer
traffic; duplicated, stale and non-neighbor messages; training with no completed
network transfer; topology edge cases; fixed-time local-model snapshots;
input/buffer immutability; reproducible DAGs/metrics; CLI registration; and
CPU/CUDA agreement with the existing training backend enabled.
Saved-DAG replay and list-valued experiment configuration are also tested.

The [validation record](validation/swift.json) records the fresh test and small
offline digits screening results. Those results compare stored local models at
fixed simulated times using identical capacities/data settings. The digits
workload exposes only a held-out test split, so the predeclared learning rate
is not selected using these results. This short check establishes execution
and timing behavior; it does not establish CIFAR/ImageNet performance or a
general ranking of SWIFT against the other baselines.

## Offline digits screening, 3 November 2026

The fixed matrix contains 48 GPU runs: two methods, two bandwidth regimes,
uniform/heterogeneous capacities, IID/Dirichlet data, and seeds 0, 1 and 2.
Eight clients use a 64-32-10 ReLU MLP, batch 32, learning rate .05 fixed in
advance, zero momentum/decay, one gradient per step, and a ring. SWIFT averages
every step with uniform CIS. Every stored local model is evaluated on all 445
test examples at 1, 2 and 3 simulated seconds. No test-based tuning is performed.

Final accuracy is the mean of eight local-model accuracies, then mean ± sample
standard deviation across the three seeds (percentage points):

| Capacity profile | Data | Ample: DSGD | Ample: SWIFT | Congested: DSGD | Congested: SWIFT |
| --- | --- | ---: | ---: | ---: | ---: |
| Uniform | IID | 87.00 ± 3.57 | 89.69 ± 2.80 | 63.47 ± 1.59 | 8.75 ± 1.91 |
| Uniform | Dirichlet α=.3 | 79.29 ± 2.12 | 83.45 ± 0.59 | 42.12 ± 8.42 | 9.29 ± 1.21 |
| Heterogeneous | IID | 33.06 ± 18.00 | 70.26 ± 9.27 | 24.09 ± 4.41 | 10.00 ± 0.34 |
| Heterogeneous | Dirichlet α=.3 | 22.98 ± 12.29 | 46.14 ± 8.30 | 18.86 ± 7.14 | 10.25 ± 0.42 |

Ample bandwidth is 100 MB/s nominal per client; congested bandwidth is 1 MB/s
(decimal units). Uniform training takes 9 ms per gradient. The heterogeneous
profile independently draws gradient times from 6, 15, 36 and 90 ms, and assigns
one quarter of clients 4× nominal bandwidth and the remainder .5×. Capacities
are exactly paired between methods for each seed. Mean capacity differs between
the uniform and heterogeneous profiles, so compare methods within a profile.

With ample bandwidth SWIFT schedules 2672 gradients in each uniform run,
versus 2144 for DSGD; heterogeneous means are 1485.33 versus 394.33.
Counts include gradients in progress at the horizon. Completed counts are in
the [per-run summary](validation/swift-digits-summary.csv). SWIFT maintains
the same gradient count when bandwidth is congested. However, training
progress alone does not imply accuracy progress: queued broadcasts leave cache
entries stale and repeatedly pull local parameters toward old values. The
congested results demonstrate that failure mode. Wire totals include partial
transfers; they do not imply those models have reached the mailbox.

This short, untuned screening supports the wait-free timing contract and
shows strong sensitivity to bandwidth and stale caches. It cannot establish
long-run time-to-accuracy, an equal-client objective under unequal activation
rates, or performance on larger models. Scalar CUDA executor wall time is
recorded separately and is not a protocol speed measurement; these jobs ran
concurrently with validation work.

Reproduce the ample-bandwidth matrix with:

```sh
python scripts/validate_algorithms.py --variants swift dsgd --dataset digits \
  --peers 8 --local-steps 1 --duration 3 --test-period 1 --bandwidth 100000000 \
  --latency-ms 2 --learning-rate .05 --profiles uniform heterogeneous \
  --partitioners iid dirichlet --seeds 0 1 2 --device cuda:0 \
  --output output/swift-validation/digits-ample
```

Use `--bandwidth 1000000` and a separate output directory for congestion.
The [source/configuration record](validation/swift.json),
[per-seed results](validation/swift-digits-summary.csv) and
[fixed-time curves](validation/swift-digits-curves.csv) preserve both regimes.
