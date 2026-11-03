# FSPDA-SA

`fspda` implements the stochastic-approximation variant of the **Fully Stochastic
Primal Dual Algorithm** from [Yau, Liu and Wai, arXiv:2410.18774v2](https://arxiv.org/abs/2410.18774v2),
*A Stochastic Approximation Approach for Efficient Decentralized Optimization
on Random Networks* (28 May 2025). The [authors' repository](https://github.com/OscarYau525/FSPDA)
is pinned at `0808887c6e9f713ca203df3138fb8995094be694`. FSPDA-STORM is excluded.

## Recurrence and source discrepancy

Let x_i be client i's model, u_i its node dual, and C_ij the diagonal binary
coordinate mask shared by the endpoints of an active undirected edge. Each
iteration uses one fresh local minibatch gradient at the **old** model:

```
L_i       = sum_active_j C_ij (x_i - x_j)
x_i_next  = x_i - learning_rate * gradient_i - dual_weight * u_i - consensus_stepsize * L_i
u_i_next  = u_i + dual_stepsize * L_i
```

All right-hand sides use old states. `consensus_stepsize` is gamma,
`dual_weight` is eta, and `dual_stepsize` is beta. They are independent constant
coefficients; eta and gamma already absorb the primal learning rate in the
paper's notation. There is no degree normalization, Metropolis weight,
inverse-probability scaling, gradient tracking, or compression error feedback.
Optional shared weight decay adds `weight_decay*x_i` to the minibatch gradient.

This is the saddle-point descent/ascent in equations (5) and (7), using the
positive graph Laplacian, and agrees with Appendix A, equations (35)-(36) when
local counters agree and a gradient is ready. Equations (6) and (8b) instead
print the opposite sign for the Laplacian/dual ascent; we interpret those signs
as typographical errors because they conflict with the incidence-matrix
definition, equation (7), the eliminated recurrence (11), and Appendix A (36).

The pinned authors' `pcode/optim/fspda.py` has a separate update-order
discrepancy: its primal uses old self and old neighbor models, but its dual
uses **new self minus old neighbors**. The simulator uses old models in both
updates, following the paper formulation. The executable
[reference audit](../tests/check_fspda_reference.py) runs the unchanged reference
step and helpers with known old-model packets. It verifies primal agreement at
matched inputs and measures exactly the dual difference
`beta * sum_active_j C_ij (x_i_next - x_i)`. It does not assert full-trajectory
reference equivalence or reproduce the reference distributed deployment.

All clients start with the shared seeded model and **zero dual**. Symmetric
edges and masks preserve `sum_i u_i=0` at every common completed logical
iteration, so the mean model follows the uniform mean of the local gradients.
Unequal client data sizes do not change those client weights. In heterogeneous
execution a wall-clock snapshot may contain different logical iterations,
so the common-iteration invariant need not hold across that snapshot.

An isolated client still takes `x_next=x-lr*gradient-eta*u`; its persistent
dual is retained. Omitting the dual force on inactive rounds changes FSPDA.

## Random network, communication and supported scope

The connected underlying graph can be a ring, complete graph, or seeded
connected regular graph. In `bernoulli` mode every undirected edge is sampled
independently each logical iteration with `edge_probability` in (0,1].
In `one-edge` mode the initiator rotates across clients and selects one of its
neighbors uniformly; exactly that edge is active (`edge_probability` is unused).
Individual rounds may be disconnected or have no active edges.
The rotating initiator gives edge support over a client cycle rather than an
iid draw over all edges at every iteration; it follows the reference experiment
schedule and is not a claim that every theorem assumption holds per round.

For each active edge, choose `max(1,ceil(D*keep_ratio))` distinct coordinates
uniformly over the whole flattened parameter vector. The two endpoints use
the same mask. Values are **unscaled** projections. Distinct edges and rounds
have separate deterministic seed streams. `keep_ratio=1` sends dense values
and omits indices; otherwise each payload contains k int64 indices and k
parameter-dtype values, costing `k*(8+element_size)` bytes. Even selected zeros
consume wire bytes. Every endpoint sends its old model coordinates once;
there is no dual, gradient or acknowledgement packet. A generic control header
and serialized round metadata are charged per directed send. The generic
`model_size_bytes` override does not replace this explicit tensor layout.

The logical activation schedule and mask seeds are known to all clients at
setup; this is a shared-schedule assumption with zero simulated discovery or
mask-negotiation traffic. The implementation does not copy the reference's
distributed mask negotiation. Explicit indices are still charged on the wire.
No neighbor model or gradient is read without its corresponding transfer.

Old-model packets leave at iteration start and overlap one charged gradient
computation at the old model. Updating waits for that local gradient and all
same-iteration active neighbors. Early packets are buffered; duplicate packets
and completion callbacks cannot repeat an update. Default execution has local
neighbor dependencies; `--synchronous` adds a global iteration-start barrier.
This is a logical-round implementation, **not** Appendix A's interruptible,
stale-message asynchronous counter/gradient-rescaling algorithm.

Transfers use shared upload/download contention and latency. Vector updates,
mask selection and topology setup have zero simulated compute cost; gradient
tasks have explicit positive durations. Actual numerical CPU/CUDA execution
time is separate from simulated time. Publication and snapshots expose only
completed updates. The saved `fspda-network.json` records the contract and
`fspda-rounds.csv` records started/gradient-ready/completed times and neighbors.

Supported settings are stable membership, one local gradient per iteration,
shared momentum zero, individual or snapshot evaluation, fully trainable
nonempty floating parameters with a common dtype. Model buffers remain local
and follow the local gradient forward pass. Unsupported churn, LoRA,
local-SGD blocks and shared momentum fail early. STORM, Adam, extra dual
momentum, compression noise, packet loss/recovery and learning-rate schedules
are excluded. Positive finite coefficients do not by themselves guarantee
stability: gamma multiplies an unnormalized Laplacian, so topology and gains
must be tuned together. Defaults are usable screening values, not paper tuning.

## Usage

```sh
python main.py --algorithm fspda --local --dataset digits --model mlp \
  --peers 8 --local-steps 1 --momentum 0 --learning-rate .05 \
  --fspda-topology ring --fspda-edge-probability .5 --fspda-keep-ratio .1
```

Use `--fspda-sampling one-edge` for rotating-initiator single-edge activation.
Use `--fspda-edge-probability 1 --fspda-keep-ratio 1` for an all-edge dense
control. The same options work in the synthetic/digits, CIFAR and Tiny ImageNet
comparison runners; only synthetic/digits execution is validated here.

## Validation

[Independent tests](../tests/test_fspda.py) compare complete logistic-regression
model/dual trajectories against an incidence-matrix oracle, including two
topologies, both activation modes, sparse/dense coordinates, heterogeneous
capacities and optional barriers. Checks also cover conservation, convergence
to the known optimum of unequal local quadratics, isolated persistent dual
forces, old-state ascent, actual coordinate/value payload bytes, overlap and
readiness, early/duplicate messages, small graphs, local buffers and immutable
inputs, duration stopping, streaming snapshots, seeded serialized replay,
CPU/CUDA agreement and CLI/broker registration.

The [validation record](validation/fspda.json) preserves test results, source
hashes, the executed reference discrepancy audit, benchmark settings and
per-seed evidence. The README labels this **Numerical · Protocol · Integration**;
the reference audit establishes a documented difference, not full reference
equivalence. No multi-process FSPDA broker deployment, large-model performance
or paper experiment reproduction is claimed.

The `conflux-gpu` full suite passed **781 tests** (five existing warnings,
121.00 s), including **54 FSPDA cases**. The actual CLI also completed a
four-client synthetic CUDA run. The pinned reference audit executed 24
matched-input node steps: maximum primal error `5.55e-17`, maximum error in
the predicted dual-order difference `3.11e-16`, and a measured dual discrepancy
of `0.3522`. This is evidence for the documented difference, not equivalence.

## Digits screening

The frozen [per-seed summary](validation/fspda-digits-summary.csv) and
[curves](validation/fspda-digits-curves.csv) cover **66 CUDA runs**: 36 default
FSPDA/DSGD/GT-DSGD ample-bandwidth runs, 18 congested non-IID runs, six all-edge
dense FSPDA controls and six rotating single-edge controls. Each condition uses
eight clients, a 2410-parameter float32 MLP, three simulated seconds, one-second
snapshots, learning rate .05, batch size 32, no momentum/decay, seeds 0/1/2,
and fixed gains gamma=.1, eta=.01, beta=.1. Default FSPDA uses edge probability
.5 and keep ratio .1. No test-set tuning or selection was performed.

Final mean local test accuracy (%), mean ± sample standard deviation:

| Bandwidth | Profile | Data | FSPDA-SA | DSGD | GT-DSGD |
| --- | --- | --- | ---: | ---: | ---: |
| 100 MB/s | Uniform | IID | 88.16 ± 2.47 | 87.00 ± 3.57 | 83.60 ± 4.78 |
| 100 MB/s | Uniform | Dirichlet α=.3 | 58.18 ± 3.66 | 79.29 ± 2.12 | 77.56 ± 5.37 |
| 100 MB/s | Heterogeneous | IID | 30.74 ± 12.15 | 33.06 ± 18.00 | 30.51 ± 16.57 |
| 100 MB/s | Heterogeneous | Dirichlet α=.3 | 16.50 ± 7.61 | 22.98 ± 12.29 | 19.85 ± 13.17 |
| 200 kB/s | Uniform | Dirichlet α=.3 | 25.53 ± 5.76 | 14.68 ± 3.92 | 10.41 ± 0.72 |
| 200 kB/s | Heterogeneous | Dirichlet α=.3 | 14.84 ± 4.72 | 10.87 ± .46 | 8.87 ± 2.21 |

Sparse FSPDA saves traffic and performs comparably on uniform IID data here,
but falls behind the dense baselines on ample-bandwidth non-IID data. Its
congested means are higher, while all congested and heterogeneous runs have
low accuracy at this short horizon. There is no consistent overall winner.

With all edges active and dense coordinates, FSPDA reaches **85.77 ± .64%**
on uniform non-IID data and **20.91 ± 8.29%** on heterogeneous non-IID data.
The rotating single-edge sparse control reaches **45.61 ± 2.49%** and
**18.15 ± 9.62%**, respectively, at 100 MB/s. Those controls change both graph
activation and coordinate coverage; they do not isolate the value of the dual
correction. Gains were held fixed across all controls.

At uniform ample capacity, default FSPDA sends 7.98 MB on average and completes
2664 local updates, versus DSGD's 41.83 MB/2136 and GT-DSGD's 69.87 MB/1790.
Dense FSPDA sends 51.99 MB/2664 updates; single-edge sparse FSPDA sends
1.99 MB/2664. Under uniform congestion, default FSPDA completes about 910
updates with 2.76 MB sent, versus DSGD's 190/3.75 MB and GT-DSGD's 96/3.85 MB.
Counts include only completed model updates; scheduled gradients also include
work in progress. Traffic includes partial transfers at the horizon.

Sparse FSPDA packets contain 241 int64 coordinates and 241 float32 values
(2892 bytes); dense packets contain 9640 value bytes. Each send also pays its
header and round metadata. DSGD uses one and GT-DSGD two sequential dense
neighbor exchanges per iteration; FSPDA overlaps old-model exchange with its
gradient. Different update orders, coefficients and exchange patterns mean
this is a method screening rather than an isolated communication ablation.

Latency is 2 ms. Uniform gradients take 9 ms; heterogeneous gradient times are
drawn independently from 6/15/36/90 ms, while a quarter of clients get 4×
nominal bandwidth and the rest .5×. Capacities are paired across methods within
profile/partition/seed. Profile means differ. Actual scalar CUDA execution
times overlap other validation work and are separate from simulated time.

Reproduce the ample-bandwidth comparison with:

```sh
python scripts/validate_algorithms.py --variants fspda dsgd gt-dsgd --dataset digits \
  --peers 8 --local-steps 1 --duration 3 --test-period 1 --bandwidth 100000000 \
  --latency-ms 2 --learning-rate .05 --profiles uniform heterogeneous \
  --partitioners iid dirichlet --seeds 0 1 2 --device cuda:0 \
  --output output/fspda-validation/digits-ample
```

For congestion, change to `--bandwidth 200000 --partitioners dirichlet` and a
separate output directory. For controls use `--variants fspda --partitioners
dirichlet`, adding either `--fspda-edge-probability 1 --fspda-keep-ratio 1` or
`--fspda-sampling one-edge`, again with distinct output directories. The
validation record contains all four manifests, source hashes, capacities and
audit results. Every run has finite metrics for all eight clients at all three
snapshots, matching sent/received bytes, and charged completed updates within
the horizon. CIFAR, Tiny ImageNet, larger models and live broker execution
remain unbenchmarked.
