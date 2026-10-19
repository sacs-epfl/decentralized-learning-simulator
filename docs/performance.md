# Shared execution optimizations

These changes affect the cost of executing a protocol's training DAG, not its
learning rule or simulated communication schedule.

- Model state transfers copy complete parameters and buffers directly, without
  serializing through pickle. Cached model structures avoid redundant random
  initialization, preserve RNG state and always return independent values.
- DAG tasks index references once; streaming execution releases values after
  their final consumer. Tensor edge copies clone only the required slice.
- Exact duplicate chunk reconstructions can share a bounded cache in streaming
  mode. Keys are DAG references, never approximate model hashes. Tests compare
  every saved checkpoint with uncached execution and exercise eviction/release.
- Native CIFAR/Tiny ImageNet tensors and deterministic per-peer data loaders
  avoid repeated transforms and dataset construction. Local steps continue
  across epoch boundaries without silently dropping requested work.
- CUDA training reuses private workspaces and a bounded immutable input-state
  cache. Independent ready tasks can execute in guarded CUDA stream batches.
  Startup numerical/performance checks can disable batching; unsupported shapes,
  models or memory pressure use scalar/reference execution. Optimizer momentum
  and model buffers are reset from the correct input for each local-SGD task.
- Inputs are immutable DAG values. Training cannot mutate a predecessor that
  other tasks or evaluation snapshots still need.

Local runners default to residency and batching on supported CUDA workloads.
Disable them with `--no-gpu-residency` / `--no-training-batching`; configure width
and cache bounds with `--training-batch-size` and `--gpu-state-cache-mb`.
CPU/MPS use the reference path. Custom DeMo gradient tasks currently use their
own reference execution path. The distributed worker path benefits from shared
state-copy/data improvements; the guarded resident/batched backend is currently
used by the local DAG executor.

The shared simulator also contains correctness fixes for byte accounting,
immutable control messages, device compute serialization, local chunk delivery,
state/buffer aggregation and evaluation timestamps. These corrections must not
be disabled to obtain a speedup.

Run `python -m pytest` for maintained regression tests. CUDA-specific numerical
checks skip on machines without CUDA. `scripts/benchmark_training.py` measures
execution cost with numerical checks and separate warmup/data-loading accounting.
Historical Tiny ImageNet and CIFAR optimization audits remain on the research
and archive branches; their hardware-dependent speedups are not universal claims.
