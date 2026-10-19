# DeMo baseline

DeMo (Decoupled Momentum Optimization, ICLR 2026) is implemented as a stable-membership, synchronous sparse collective. Primary sources: [conference paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17790fc9b4828206f99a0dea35da3657-Abstract-Conference.html), [paper v2](https://arxiv.org/html/2411.19870v2), and [official standalone optimizer](https://github.com/bloc97/DeMo/blob/0e48145d702b25f83055e11d9585a9c11bcfe154/demo.py) (SHA256 `578068af51d543477d79af06caa6e3c01d544db0967c8d13b6ba2972d209a8ce`).

The implementation follows the released Signum optimizer, including details where the paper's prose/pseudocode differ: `delta = .999 * residual + lr * gradient`; orthonormal block DCT; top-32 coefficients per block with target chunk 64; subtract the worker's own decoded transmission; gather sparse indices and values; average collisions among **transmitted** values (`scatter_reduce(mean, include_self=False)`); inverse transform; decoupled weight decay and `parameter -= lr * sign(reconstruction)`. Missing coefficients are not treated as extra zero-valued participants in that mean. Residuals remain private to each worker and persist as immutable DAG outputs across rounds. Ordinary local SGD momentum is rejected.

The reference transform handles vectors and matrices. Convolution kernels are unfolded into `(out_channels, -1)` matrices, then restored after inverse transformation. This explicit CNN adaptation is not a reproduction of the paper's language-model experiments. Models with buffers (including BatchNorm running statistics) or frozen parameters are currently rejected instead of silently dropping their state. CPU and CUDA share the same Torch implementation; deterministic CPU coefficient accumulation avoids CUDA scatter atomic ordering differences. No new runtime dependency is needed.

A decentralized ring all-gather forwards each worker's packet over `N-1` hops. Every transfer charges int64 indices and parameter-dtype values, metadata, latency, and bandwidth contention. GNLeNet has 85,354 parameters and its default packet is 20,472 bytes (the comparison's pinned full-model payload remains 346,251 bytes). No dense model broadcast is added after the collective: all recipients compute identical updates. The local executor memoizes that identical pure update, but each node publishes it only after its own complete gather. Workers can enter their next logical iteration at different physical times; future packets are buffered. Stable membership is required; churn recovery is not implemented. As for the other baselines, compression/aggregation compute has zero simulated duration; real execution time is measured separately.

`--algorithm demo` and `scripts/run_cifar.py --variants demo` expose `--demo-decay`, `--demo-chunk`, `--demo-topk`. `learning.local_steps` is the number of microbatch gradients accumulated at **fixed** parameters before a DeMo optimizer step. Thus H=1 is the canonical one-batch algorithm; H=5 matches the other algorithms' five-batch compute block but does not mean five uncommunicated sign updates. Mean microbatch gradients are used. Batch size and simulated per-microbatch compute cost retain their existing meanings. GPU residency/batching optimizations for ordinary SGD tasks do not apply to these custom tasks; this affects execution speed only.

`tests/test_demo.py` checks independent SciPy DCT values, collision means, residual conservation/persistence, the dense SignSGD limit, fixed-weight gradient accumulation, exact wire-byte accounting, single-node and multi-node ring causality, skewed arrivals, and retained/streaming execution equivalence. `tests/check_demo_reference.py PATH_TO_OFFICIAL_DEMO_PY` additionally runs seven steps with three workers against the pinned official optimizer: zero parameter difference and maximum residual difference `4.19e-9` on CPU.

## CIFAR evaluation

`scripts/benchmark_demo.py --mode tune` selects learning rates on the separate
5,000-image validation split. `--mode evaluate --profile uniform` (or
`heterogeneous`) evaluates both H=1 and H=5 over IID/non-IID partitions and three
seeds. The generic driver runs DeMo only; combine results with other algorithms
using the shared `scripts/run_cifar.py` settings. The research comparison and
its frozen provenance remain on the `conflux-v2` branch.
