# DSpodFL comparison

This implements [DSpodFL, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html)
from [Equation (2) and Algorithm 1](https://arxiv.org/html/2402.03448#A2), checked
against the authors' [reference agent](https://github.com/ShahryarBQ/DSpodFL/blob/main/src/Agent.py).

## Update semantics

For each logical iteration, both endpoints of a physical edge use the same seeded
Bernoulli draw. Active edges exchange **old** models from that iteration. Each
node independently chooses whether to perform a local SGD update. Its result is:

```text
next_model_i = local_trained_model_i
             + Σ_j r_ij × (old_model_j − old_model_i)
```

If local training is skipped, `local_trained_model_i = old_model_i`.
`r_ij = 1 / (1 + max(degree_i, degree_j))` uses degrees of the physical graph.
Inactive weights return to the diagonal; surviving edges are not renormalized.
The induced mixing matrix is symmetric, nonnegative and doubly stochastic.
This differs from averaging the newly trained neighbor models, which would
incorrectly dilute/mix the local gradient in Equation (2).

With `--local-steps 1`, this is the paper's one-step SGD equation. With
`--local-steps 5`, the local update is a five-step local-SGD delta, matching the
previous comparison's local block length. That is an explicit extension, rather
than a claim that the paper analyzes five-step deltas as one stochastic gradient.
Both configurations are reported separately.

Old-model transfers overlap computation. A node waits for its own selected
training and all active same-iteration neighbors before publishing its update.
Arrivals before local startup/training completion and future-iteration arrivals
are buffered separately. Data dependencies enforce causality without an extra
global iteration barrier. Empty iterations have a minimum 20 ms opportunity
duration, preventing infinitely many free iterations; this is an explicit
simulation convention, not an inferred computation cost from the paper.
Control headers and full model transfers consume the shared bandwidth/latency
scheduler. Local training has the existing explicit simulated duration; aggregation
is instantaneous in simulated time for every method.

## Predeclared policy

The main comparison uses a seeded connected degree-4 random regular graph.
This is a valid physical topology for the framework, and differs from the
paper's random geometric graph. Nominal SGD and link probabilities are both 0.5.
They are reduced for below-average capacity:

```text
p_sgd(i)  = 0.5 × min(1, mean_training_duration / training_duration_i)
p_link(ij)= 0.5 × min(1, min(bandwidth_i, bandwidth_j) / mean_bandwidth)
```

The policy is independent of client data, accuracy and test results. Capacities,
topology, mixing weights and probabilities are fixed; event masks change each
iteration. Probabilities and edges are saved in `dspodfl_probabilities.csv`.
`constant` and `beta` probability modes are also implemented; `beta` draws fixed
node/edge probabilities from Beta(0.8, 0.8), as in the paper's CIFAR experiment.
They are not part of this fixed matrix, and the capacity-derived policy is not
claimed to be an author-tuned default.

Stable membership and plain SGD are required. Sporadic computation/communication
does not implement offline-node failure detection, reconnects or recovery.
Nonzero momentum and weight decay are explicitly rejected for this implementation.

## Running

Use `--algorithm dspodfl` in the coordinator or `--variants dspodfl` with
`scripts/validate_algorithms.py` / `scripts/run_cifar.py`.
`scripts/benchmark_dspodfl.py` runs the fixed CIFAR matrix for H=1 and H=5.
Flags: `--dspod-degree`, `--dspod-sgd-probability`,
`--dspod-communication-probability`, `--dspod-probability-mode`.

Mathematical and causal execution checks are in `tests/test_dspodfl.py`.
The implementation is a specialization under our workloads, not a reproduction
of the paper's VGG11/random-geometric-graph tuning. Historical comparative
results and raw-result provenance are preserved on the research branch.
