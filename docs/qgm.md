# Quasi-Global Momentum

`qgm` implements QG-DSGDm from [Lin et al., ICML 2021](https://proceedings.mlr.press/v139/lin21c.html),
with optional Nesterov momentum and two controls using the same transport:
persistent local-momentum DSGD (`dsgdm`) and plain DSGD (`dsgd`). These runner
aliases all use registered `QGMSettings`/`QGMSimulation`; saved settings record
`algorithm=qgm` plus the optimizer mode.

For each node, gradient `g` includes coupled SGD weight decay. Starting with
private buffer `h=0`, the ordinary update is:

```
v_i       = beta * h_i + g_i
u_i       = v_i                         # ordinary
u_i       = g_i + beta * v_i            # Nesterov option instead
proposal  = x_i - lr * u_i
x_i_next  = sum_j W_ij * proposal_j
h_i_next  = mu * h_i + (1-mu) * (x_i-x_i_next)/lr
```

Crucially, the buffer uses the displacement **after neighbor mixing**, including
its consensus component. The local-momentum control instead retains `v_i` as
its next buffer. Plain DSGD uses beta zero and retains no buffer. Defaults are
beta .9 and mu=beta, following the paper's numerical setting. Nesterov is off by
default and can be enabled explicitly. The original paper and reference define
both variants.

Only **one gradient step per exchange** is supported. `local_steps!=1` is
rejected rather than silently inventing a block extension. The common trainer's
`learning.momentum` must be zero: use `--qgm-beta` for the explicit persistent
buffer, which is never reset at task boundaries. Gradient computation uses the
existing trainer, data transforms and seeded minibatches; exact gradients are
used rather than reconstructing them from a rounded parameter difference.

The private buffer is an immutable DAG output. Migration across workers and
saved-DAG replay preserve it without hidden process-local optimizer state.
Ring, complete and connected regular graphs use symmetric Metropolis weights.
Nodes wait for all neighbors' same-round proposals; early packets are buffered
by round. `--synchronous` on the main CLI adds a global round barrier. Stable
membership is required; churn is rejected.

Each neighbor transfer contains only a dense parameter vector. The private
momentum buffer stays local and causes no additional network messages. All
three controls use the same payload layout, phase metadata, bandwidth/latency
scheduler and simulated gradient-step cost. Buffers such as BatchNorm running
statistics follow the local trainer and are not mixed or transmitted. Models
with frozen parameters are currently rejected. One common floating parameter
dtype is required.

Gradient tasks currently use the scalar execution path, including on CUDA;
GPU batching of these custom gradient tasks is not yet implemented. This affects
simulator execution time, not simulated protocol time. Keep those two measures
separate in comparisons.

## Usage

```sh
python scripts/validate_algorithms.py --variants qgm dsgdm dsgd \
  --peers 8 --local-steps 1 --duration 30 --test-period 5 \
  --qgm-beta .9 --qgm-nesterov --output output/qgm-validation/example
```

The main broker CLI uses `--algorithm qgm --momentum 0 --local-steps 1` and
`--qgm-mode quasi|local|plain`. The aliases in comparison runners select the
corresponding mode. `--qgm-mu` defaults to beta. Momentum state is not sent with
models. The existing legacy `dpsgd` implementation remains unchanged.

## Validation

`tests/test_qgm.py` compares five-round full-batch logistic-regression trajectories
against independently calculated gradients, mixing and buffers, for all three
modes, ordinary/Nesterov momentum, and local/global round barriers. Tests cover
skewed arrivals, exact traffic, state persistence, model buffers, immutable
inputs, graph edge cases, deterministic metrics and CPU/CUDA equivalence.

`tests/check_qgm_reference.py` optionally executes the numerical `step` method
from the [official implementation](https://github.com/epfml/quasi-global-momentum/tree/d1d2f758c549db6093c52be71bcdbdf1a3976549),
pinned at that revision. Only tensor packing, timing and weighted transport are
substituted; the reference's SGD, Nesterov, virtual-sequence and momentum-buffer
arithmetic is executed unchanged over eight rounds on four workers. This does
not claim to reproduce its MPI deployment or paper benchmark recipe.

The [validation record](validation/qgm.json) pins the source and checks. The
eight-round reference comparison has maximum parameter error
`2.22e-16`. The main-branch suite passed 227 tests on macOS CPU (13 CUDA skips);
the common research execution snapshot passed all 265 tests on Linux CUDA.
Real replay through one broker/two workers and two brokers/one worker matched
all 48 validation rows and all eight final full-test rows against CPU local
execution. Those execution checks establish numerical/state consistency, not
comparative algorithm performance.

CIFAR evaluation uses disjoint training/validation/test splits. Balanced fixed
validation subsets accelerate curves; final test tasks explicitly select the
full held-out test set. Their metric-cache key includes the evaluation split,
so a validation evaluation cannot be reused as a test evaluation. Independent
tests check both split isolation and balanced subset membership.

`benchmark_qgm.py` supports separate tuning, LR selection and evaluation phases
with resumable per-job manifests and disjoint worker slots. The implementation
and experiments are recorded separately. Historical experiment directories are
not modified.

## CIFAR-10 screening benchmark

The [audited benchmark record](validation/qgm-cifar64.json) contains all 36
baseline runs: 64 nodes, three seeds, uniform/heterogeneous capacities and
IID/Dirichlet-alpha-.1 partitions, GN-LeNet without augmentation, batch 32,
one gradient step per exchange, and 900 simulated seconds. The fixed ring uses
symmetric Metropolis mixing. Independent compute/network heterogeneity preserves
the uniform profile's mean capacity (20 ms per gradient step and 100000 B/s).
These are synthetic capacities, not FedScale traces. Membership is stable.

Each variant selected its learning rate from `.001,.003,.01,.03,.1` using the
mean final full-validation accuracy across uniform IID/non-IID seed-0 runs.
The selected rates are `.003` for QG/local momentum and `.01` for plain DSGD;
beta=mu=.9, Nesterov enabled for both momentum variants, zero weight decay.
Training uses 45000 examples, selection uses a disjoint 5000, and final test
uses the separate full 10000-image set. Six curves use a fixed balanced
1000-image validation subset; those curves are distinct from final test metrics.

Final full-test accuracy, mean of all 64 stored local models, then mean ± sample
SD over the three seeds:

| Capacities | Data | Plain DSGD | Local momentum | QG momentum |
|---|---|---:|---:|---:|
| Uniform | IID | 35.27 ± 0.17 | 44.10 ± 0.50 | 44.24 ± 0.21 |
| Uniform | Dirichlet α=.1 | 19.29 ± 0.43 | 20.86 ± 1.21 | 25.23 ± 1.09 |
| Heterogeneous | IID | 27.45 ± 0.41 | 34.12 ± 0.52 | 32.85 ± 0.41 |
| Heterogeneous | Dirichlet α=.1 | 15.66 ± 0.67 | 17.13 ± 0.51 | 19.14 ± 1.05 |

QG's correction improves non-IID accuracy over persistent local momentum by
4.37/2.01 percentage points, with lower cross-entropy (2.577/2.853 versus
3.310/3.497). IID performance is tied under uniform capacities and worse than
local momentum under heterogeneous capacities. This is screening evidence, not
a universal ranking or reproduction of the paper's architecture/training recipe.

All three controls have exactly identical simulated traffic/work within each
profile/seed. Uniform runs send 4.713 decimal GB and schedule 6937 gradient
tasks; heterogeneous means are 1.809 GB and 2678.67 tasks. Traffic includes
in-flight bytes at the final snapshot. Every node at all six snapshots and
all final test rows is finite; capacities match exactly across controls.
Real per-run GPU execution is approximately 58–61 seconds (uniform) and 27–29
seconds (heterogeneous), separately from the 900-second simulated horizon.
Custom gradient tasks use scalar CUDA execution. These execution timings do not
establish algorithm speed. Broader tuning, longer runs, other models/topologies
and more seeds could change the result.
