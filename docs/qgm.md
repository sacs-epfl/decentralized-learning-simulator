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

CIFAR evaluation uses disjoint training/validation/test splits. Balanced fixed
validation subsets accelerate curves; final test tasks explicitly select the
full held-out test set. Their metric-cache key includes the evaluation split,
so a validation evaluation cannot be reused as a test evaluation. Independent
tests check both split isolation and balanced subset membership.

`benchmark_qgm.py` supports separate tuning, LR selection and evaluation phases
with resumable per-job manifests and disjoint worker slots. The implementation
and experiments are recorded separately; see the evaluation report added after
completion for the actual configuration, source hashes, tuning choices and
results. Historical experiment directories are not modified.
