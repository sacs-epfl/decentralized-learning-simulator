# CHOCO-SGD

Registered as `choco` in the CLI, distributed broker settings registry, and the
synthetic/digits, CIFAR-10 and Tiny ImageNet comparison runners.

Sources:

- Koloskova, Stich and Jaggi, [Decentralized Stochastic Optimization and Gossip
  Algorithms with Compressed Communication](https://proceedings.mlr.press/v97/koloskova19a.html),
  ICML 2019, Algorithm 2.
- Koloskova, Lin, Stich and Jaggi, [Decentralized Deep Learning with Arbitrary
  Communication Compression](https://arxiv.org/abs/1907.09356), ICLR 2020.
- [Official epfml/ChocoSGD implementation](https://github.com/epfml/ChocoSGD/tree/3cb976a54499c2cdc8ea68e001217b42b58870ae),
  pinned for the optional numerical reference audit.

## Update contract

This implementation uses the **ICML Algorithm 2 ordering**. Each node maintains
its private model `x`, its own reconstructed public estimate `h`, and a weighted
sum `m = sum_j W_ij h_j` of public estimates (including its own). Both estimate
vectors start at zero; the common initial model is not preloaded into them.

For each logical round:

```
y_i      = LocalSGD(x_i, H steps)
x_i_next = y_i + gamma * (m_i - h_i)           # previous round's estimates
q_i      = Q(x_i_next - h_i)
            exchange q_i with graph neighbors
h_i_next = h_i + q_i
m_i_next = m_i + sum_j W_ij * q_j             # includes q_i locally
```

`q` in the equations denotes the decompressed packet. Only the compressed packet
crosses the network. The two vectors retain reconstruction error across rounds;
compressing each freshly trained model and discarding this state is not CHOCO.

For `H=1`, zero momentum and zero weight decay this is the paper's plain-SGD
update. `H>1` is an explicitly supported **local-SGD block extension**, not H
independent CHOCO steps. The ICLR presentation reindexes communication and
training; this code does not silently substitute a freshly updated estimate in
the old-estimate correction. With identity compression and `gamma=1`, the dense
limit is old-model mixing minus each worker's own gradient, rather than averaging
all freshly trained models. This leaves the toolbox's existing train-then-average
D-PSGD variant unchanged.

The default topology is a ring. Complete and seeded connected random regular
graphs are supported with symmetric, doubly stochastic Metropolis weights:
`W_ij = 1/(1 + max(deg(i),deg(j)))`, and the diagonal completes each row to one.
Self updates are local and cause no network transfer. One- and two-node rings
are supported without duplicate/self transfers.

Nodes wait for all neighbors' packets from the same round. Early packets are
buffered by round, so heterogeneous speeds do not mix future estimates into an
older correction. The default permits local round progress; `main.py --synchronous`
adds a global simulated-time round barrier. These are the same logical updates,
with different execution schedules. Snapshots use each node's latest completed
round model; CHOCO has no single shared aggregate.

## Compression and communication accounting

- `topk` retains the largest absolute coordinates **per trainable parameter
  tensor**, following the reference compressor. Top-k ties may select any valid
  tied coordinates, as in PyTorch's reference operator.
- `randomk` samples coordinates without replacement, per tensor, using a
  deterministic per-node/per-round seed. Values are **not** rescaled by `d/k`:
  this is the biased contraction compressor used by CHOCO, not an unbiased
  gradient estimator. The random sequence need not match the reference's NumPy
  sequence.
- `identity` sends every trainable parameter coordinate, with no indices.

For sparse modes, `k = max(1, floor(numel * keep_ratio))` for each tensor. The
option specifies the fraction **retained**, whereas the reference's
`compress_ratio` specifies the fraction discarded. Nonempty tensors therefore
retain at least one coordinate. Coordinates are global flat int64 indices;
values retain the parameter dtype. A sparse FP32 coordinate costs **12 bytes**
(8-byte index plus 4-byte value), so retaining 1% generally costs about 3% of the
dense parameter payload before headers and per-tensor rounding. Sparse mode at
100% still pays for indices; use `identity` for the dense control.

Every directed neighbor transmission uses the shared bandwidth/latency scheduler
and is charged value/index bytes, serialized round/phase metadata, and the
configured control header. The packet is sent separately to each neighbor;
there is no free network broadcast. Packet size comes from the actual model's
trainable parameters, not a scaled full-model pickle size. `--model-bytes` does
not override this compressed-payload calculation.

The graph, common model layout and tensor boundaries are known to all nodes.
No per-round topology discovery is modeled. `choco-communication.csv` records
weights and packet sizes; its self-loop entries describe local state updates,
not additional traffic. Actual network counters are in `client_statistics.csv`.
`choco-rounds.csv` records completion times. Compression and state-update CPU
execution time is measured by the executor, but is not assigned an additional
simulated compute delay. Keep GPU/CPU execution time separate from protocol time.

## Supported model and optimizer state

Compression covers trainable floating parameters of a common dtype. Frozen
parameters and model buffers remain local. For example, BatchNorm running means,
variances and counters follow the local trainer; they are not secretly averaged
or sent uncompressed. This matches parameter-only optimizer communication but
should be stated when comparing models with BatchNorm. GroupNorm LeNet in the
CIFAR runner does not have those running-statistic buffers.

Both public-estimate vectors are explicit, immutable DAG outputs. They survive
streaming execution, saved-DAG replay and migration between worker processes;
there is no hidden worker-local compression state. Ordinary local training uses
the existing guarded GPU residency/batching backend.

Momentum is rejected because the shared trainer resets momentum between tasks;
that would not implement persistent CHOCO momentum. Plain SGD and the shared
SGD weight-decay convention are supported. Stable membership is required and
churn is rejected. There is no automatic consensus-step or learning-rate tuning.
The default `gamma=.1` and `keep_ratio=.01` are starting settings, not a claim
that they satisfy every theoretical bound or are optimal for a workload.

## Usage and validation

Small offline execution:

```sh
python scripts/validate_algorithms.py --variants choco --peers 8 \
  --local-steps 1 --duration 30 --test-period 5 \
  --choco-compressor topk --choco-keep-ratio .01 \
  --choco-consensus-stepsize .1 --output output/choco-validation/example
```

In the main broker-based CLI use `--algorithm choco --momentum 0 --local-steps 1`.
CHOCO options are also available in `scripts/run_cifar.py` and `scripts/run_tiny.py`;
set `--momentum 0` explicitly where a runner has a nonzero default.

```sh
python -m pytest tests/test_choco.py -q
python tests/check_choco_reference.py /path/to/pinned/ChocoSGD
```

The independent five-round logistic-regression oracle checks models and public
estimates for Top-k, random-k and identity compression, `H=1/3`, global/local
barriers and skewed compute. Additional tests cover mean preservation, the dense
limit, compression-memory persistence, immutable inputs, scalar/remainder tensor
sizes, convolution/frozen parameters/local buffers, graph edge cases, exact
traffic, deterministic DAGs, finite snapshots and CPU/CUDA optimized execution.

The optional audit extracts only the official source's pure numerical helper
functions, avoiding its old MPI and bit2byte environment dependencies. It checks
eight rounds on four workers against the reference SGD, Top-k and old-estimate
consensus operations. It does not claim to reproduce that repository's full MPI
training pipeline or its communication overlap performance.

The maintained main suite passed **195 CPU tests** (ten CUDA skips) on macOS
and **205 tests with CUDA** on sacs006. One broker/two workers and two
brokers/one worker matched serial metrics on the same CHOCO DAG. Eight small
synthetic cases and both CIFAR capacity profiles completed with finite metrics
and all nodes present at every snapshot. Recorded suite, reference, broker-replay
and smoke-run evidence is in [choco.json](validation/choco.json). Small synthetic and CIFAR runs are execution
checks with untuned settings; they are not performance rankings or evidence that
CHOCO or Conflux outperforms the other. The CIFAR checks use one seed, eight nodes,
Dirichlet alpha .1, 30 simulated seconds and three fixed snapshots.
