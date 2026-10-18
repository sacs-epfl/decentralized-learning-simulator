## Decentralized Learning Simulator

This repository contains the code related to a Decentralized Learning simulator.
The main idea of this simulator is that first, the timestamps all events in the system (training, model transfers, aggregation etc.) are determined using a discrete-event simulator.
Meanwhile, the simulator devises a compute graph, containing all compute tasks (train, aggregate or test).
Afterwards, the compute graph is solved, possibly using different machines and multiple workers.

There are three different process types: the coordinator, a broker and a worker.
The coordinator node runs the discrete-event simulation and then pushes the compute graph to the brokers.
A broker process starts one or more workers, and workers are responsible for carrying out model training, aggregation or testing.

A main advantage of this simulator over other ones is that it natively supports reporting total time duration of the learning process.
It also supports the integration of traces that include the training and network capacity of each node.
FedScale bandwidth and compute traces use `data/fedscale_traces`.

Protocols include Shatter, the original Conflux, and `conflux-v2`. The latter uses
sampled training, one agreed contributor set per round, and bandwidth-aware chunk
reducers. Each next-sample recipient reconstructs the same sample-wide average.

Reported learning time is **simulated time**, distinct from the actual time needed
to execute the PyTorch workflow. Small offline validation uses a serial executor;
distributed broker/worker execution is also available.

### Conda environment and offline validation

```sh
conda env create -f environment.yml
conda activate conflux-validation
python -m pytest tests/test_bw_scheduler.py tests/test_conflux_comparison.py tests/test_comparison_edge_cases.py -q
python scripts/validate_conflux.py
python scripts/validate_conflux.py --sample-size 4 --chunks 8 --output output/conflux-validation-s4-k8
python scripts/validate_conflux.py --dataset digits --bandwidth 12000 --duration 180 --test-period 10 --output output/conflux-validation-digits
```

The environment uses Python 3.12, PyTorch 2.14.1 and TorchVision 0.29.1. Offline
validation needs no GPU, broker, account or dataset download. It executes the real
protocol-generated DAGs, including PyTorch training, aggregation and testing.

Each sweep uses 24 clients, three paired seeds, IID and Dirichlet partitions, and
uniform and heterogeneous capacities. Shatter is tested at degrees 2 and 4.
The synthetic workload is a 16-32-4 MLP. The handwritten-digits workload is a
64-32-10 MLP on scikit-learn's bundled dataset, with disjoint client training data
and a stratified held-out test set. Success fraction defaults to 1 for both Conflux
variants. These are small mechanism checks, not evidence of ImageNet performance.

Results include `summary.csv`, `curves.csv`, `accuracy.png` and a manifest recording
parameters, runtime versions and the source hash. Per-run DAGs and CSVs are in the
ignored `runs/` subdirectory. Reruns replace those named runs. Time-to-accuracy
uses fixed evaluation points; empty fields indicate targets not reached within the
common time budget. The heterogeneous profile has six nodes at four times nominal
bandwidth and eighteen at half nominal bandwidth, with independently assigned
compute speeds. These are controlled profiles, not measured WAN traces.

The metric is mean test accuracy of each node's latest fully reconstructed model
at the same simulated times. Nodes retain the common initial model until their
first reconstruction. Thus idle sampled nodes' stale models count, and neither
protocol receives a free global parameter average. Use `--test-method snapshot`
and `--test-period` for this metric in the main simulator as well.

### Comparison corrections

- Both protocols use the same work-conserving bandwidth scheduler. Conflux keeps
  bounded outstanding pulls without reserving idle bandwidth for them.
- Control advertisements, pull requests, contributor plans and attached metadata
  consume bandwidth. Nonlocal transfers use the same one-way latency; same-host
  chunk delivery is local for both protocols.
- With churn disabled, original Conflux skips redundant population-view exchanges;
  all nodes already know membership at initialization.
- Cancelled bytes and bytes sent before the experiment cutoff count toward volume.
  Slot-scheduler accounting respects the slower endpoint.
- Training is serialized per device and cannot mutate shared predecessor models.
  Sent control views and evaluation inputs are immutable snapshots.
- Accuracy CSV headers include learning rate; plots read timestamps as seconds.
- Conflux handles disabled interval tests, local contribution completion,
  overlapping pulls and round limits, and records training compute time.
- Legacy chunk coalescing only combines disjoint contributor sets: overlapping
  averages cannot be deduplicated using contributor IDs alone.
- Local steps continue across epoch boundaries. Full-model averaging includes
  floating-point buffers; integer buffers use maximum counters.

The comparison uses the repository's direct real-node Shatter implementation. It
does not simulate remote virtual-node hosting or establish equivalent privacy
guarantees. Both protocols abstract chunking and aggregation compute as instantaneous
in simulated time. Payload size is the serialized model divided among chunks,
plus modeled metadata; this is not a packet-level network model.

The checks cover actual DAG execution for all three protocols, exact Conflux v2
sample averages at full and partial success fractions, byte accounting, fixed-time
evaluation, CSV parsing, immutable inputs and device compute serialization. Other
historical protocol tests use obsolete settings and task-reference APIs; they are
not a current compatibility suite.

### Conflux v2

A temporary coordinator freezes the first
`ceil(sample_size * success_fraction)` ready contributors. Every reducer uses that
same set. Contributors upload each chunk to its reducer once; reducers average the
tensors and send aggregate chunks to every next-sample recipient. Chunk assignment
minimizes assigned load relative to public bandwidth capacity. Reducers may be
outside the training sample. More chunks can spread work over more relays, while
`--relay-count` caps potential reducers. Capacity ties rotate between rounds.

This version requires stable membership and explicitly rejects churn. Capacities
are public and fixed; coordinator/reducer recovery and changing membership need
further work. Success fraction below 1 introduces completion-dependent participation,
which can bias learning when speed correlates with client data. The default avoids
that additional selection effect. Fewer bytes do not guarantee faster convergence;
uniform networks and small samples can favor sparse Shatter.

### Running the Simulator

Use the Conda environment above. For CIFAR-10, FEMNIST, speech or Tiny ImageNet,
additionally install `flwr-datasets`; speech transforms also require `librosa`.
Those larger workloads have not been validated by the offline checks.
You can run the coordinator node using:

```
python main.py --algorithm conflux-v2 --dataset synthetic --model mlp --peers 24 --sample-size 8 --chunks-in-sample 4 --duration 90 --test-method snapshot --test-period 5
```

You can inspect the possible parameters by running `python3 main.py -h`.
The coordinator node will then run the discrete-event simulator and generate a compute graph.
By default, the coordinator expects only a single broker to be started but this number can be changed with the `--brokers` flag.
You can start a broker as follows:

```
python3 broker.py --coordinator tcp://localhost:5555
```

The coordinator will by default listen on port `5555`.
After the broker started, it will spawn a worker process, receive the compute graph from the coordinator and starts to solve the compute graph.
The resulting models (i.e., the outcome of the leaf nodes in the compute graph) will be communicated back to the coordinator.

### Validation outcomes (18 October 2026)

Final code: 25 focused tests passed. A four-client, one-worker coordinator/broker
smoke test also completed successfully. The 144 training runs above cover three
configurations, four protocol variants, four data/network conditions and three seeds.
All result manifests identify the same final source hash.

Handwritten digits, sample size 8, four chunks. Times below are median simulated
seconds to 60% mean available-node test accuracy, measured on a 10-second grid.
The Shatter column uses the faster of the two tested degrees in that condition.
Speedup is the median of the three paired seed ratios.

| Network | Data | Shatter | Conflux v2 | Paired speedup | Fewer bytes over 180 s |
| --- | --- | ---: | ---: | ---: | ---: |
| heterogeneous | dirichlet | 130 s (shatter-r2) | 80 s | 2.12x | 42% |
| heterogeneous | iid | 70 s (shatter-r2) | 40 s | 1.75x | 42% |
| uniform | dirichlet | 70 s (shatter-r2) | 100 s | 0.82x | 78% |
| uniform | iid | 40 s (shatter-r2) | 60 s | 0.67x | 78% |

These results support a conditional gain on the heterogeneous profiles, not a
universal speed claim. Uniform networks can favor sparse Shatter. Stable membership,
small models and simulated capacities remain important limits; larger-model WAN
and churn experiments are still needed.

See [the digit curves](output/conflux-validation-digits/accuracy.png),
[raw digit results](output/conflux-validation-digits/summary.csv), and
[all condition summaries](output/conflux-validation/comparison.json).

### Larger CIFAR-10 experiments

`scripts/run_cifar.py` runs resumable, deterministic GNLeNet sweeps on one GPU.
Run independent sweeps with `CUDA_VISIBLE_DEVICES=0` and `=1` to use two GPUs.
Keep the Conda environment, `TMPDIR`, caches, dataset and output on a volume with
adequate free space. The server setup uses `/mnt/data/mdevos/conflux`.

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0 python scripts/run_cifar.py \
  --dataset-base-path /mnt/data/mdevos/conflux/datasets \
  --output /mnt/data/mdevos/conflux/results/cifar128-uniform \
  --profiles uniform --peers 128 --sample-size 32 --chunks 16
```

The native loader uses 45,000 training images, a fixed stratified 5,000-image
validation split for tuning, and the official 10,000-image test split for final
measurement. Normalization matches the original pipeline; no augmentation is used.
Client partitions are disjoint, IID or Dirichlet, with at least eight examples per
client (small partitions receive examples moved from larger ones).
If the Toronto archive is slow, `scripts/prepare_cifar.py --data <dataset-path>`
prepares the canonical Hugging Face mirror with verified SHA-256 checksums. This
preparation step additionally needs `pyarrow`; its dataset provenance is recorded.

The larger sweeps equalize average bandwidth and compute step time between uniform
and heterogeneous profiles. One quarter of nodes have eight times the bandwidth of
the rest. Compute rankings are independent. `--step-ms` is the effective simulated
time per SGD step, including the legacy simulator augmentation factor; it is
separate from measured GPU execution time. `scripts/calibrate_cifar.py` measures
actual training and full-test throughput before choosing experiment budgets.

The streaming local executor frees values after their last consumer and reuses
evaluation metrics only for identical model references or reconstructions from
exactly identical chunk references. Every node still receives its own snapshot
measurement. Each sweep writes an immutable configuration/source manifest, run
progress, summary, curves, plot and a `COMPLETE` marker after all requested runs.
An incomplete run restarts; completed runs are skipped. Use a fresh output folder
when changing settings or source. This executor avoids the broker/GPU transfer
overhead while preserving the protocol-generated training DAG.

The two-GPU dispatcher (`scripts/dispatch_cifar.py --root <server-root>`) selects
each protocol's learning rate from three 64-node validation-only pilots, using the
mean of their final two snapshots. It then executes 36 paired test runs on 128
nodes: Shatter at degrees 2 and 4 and Conflux V2, two capacity profiles, two data
partitions and three seeds. Original Conflux V1 is excluded at the user's request.
The current main configuration uses sample size 32, 16 chunks, 900 simulated
seconds and snapshots every 90 seconds. All six protocol/profile combinations
selected learning rate 0.01 from {0.002, 0.01, 0.03}. This short single-seed tuning
budget is shared equally; it is not an exhaustive hyperparameter search.
On `sacs006`, independent jobs share the two 32-GB RTX 5000 Ada GPUs after initial
throughput and memory checks; CPU-heavy V1 simulations run alongside GPU work.
Actual execution times reflect this concurrency and are not protocol speedups.
Protocol comparisons use simulated time with the fixed capacities in each manifest.

#### CIFAR-10 results on 128 nodes

All 36 main runs completed on `sacs006` (two RTX 5000 Ada GPUs, 32 GB each).
The execution source is commit `a7a6881`, SHA-256
`d028c75f3bf3f1735b8861e8c8a5931aa549a13627ac1f60cb8ead70f2e416b1`.
The dispatcher defaults subsequently exclude V1; the protocol implementations are
unchanged. The audit verifies the common source hash, 36 distinct runs, three
seeds per condition, and all 128 nodes at every one of ten snapshots.

Mean test accuracy at 900 simulated seconds, GNLeNet (85,354 parameters), without
augmentation. Shatter degree 2 was stronger than degree 4 in all four conditions;
the table uses that stronger tested baseline.

| Network | Data | Shatter degree 2 | Conflux V2 |
| --- | --- | ---: | ---: |
| heterogeneous | Dirichlet | 26.11% | 39.99% |
| heterogeneous | IID | 38.74% | 45.43% |
| uniform | Dirichlet | 38.08% | 42.46% |
| uniform | IID | 50.66% | 47.83% |

On heterogeneous/Dirichlet data, V2 reaches 30% at a median of 360 seconds;
Shatter never reaches 30% within 900 seconds in any seed. This gives a median
paired speedup lower bound greater than 2.5x at that target. On heterogeneous/IID
data, V2 reaches 40% at 540 seconds in every seed; Shatter never reaches 40%
within the budget. Uniform/IID favors Shatter: 40% takes 360 seconds versus V2's
450 seconds. Targets use the 90-second observation grid; unreached targets remain
explicitly censored rather than being assigned fabricated completion times.

V2 sends about 58% fewer bytes in heterogeneous profiles and 84% fewer in uniform
profiles over the same budget. Its lower traffic does not guarantee faster
learning, as the uniform/IID result shows. These are communication-bound,
stable-membership simulations at 100,000 bytes/s mean capacity, 20 ms nominal
compute per SGD step and 20 ms one-way latency; they do not establish performance
under compute-bound workloads or on a real distributed network.

See the [curves](output/cifar-gpu/main128/accuracy.png),
[all protocols with seed variation](output/cifar-gpu/main128/table.md),
[per-seed time-to-accuracy results](output/cifar-gpu/main128/comparison.json),
and [audit](output/cifar-gpu/main128/audit.json).
Raw DAGs, individual-node CSVs, logs, dataset and Conda environment remain under
`/mnt/data/mdevos/conflux` and `/mnt/data/mdevos/conda/envs/conflux-gpu` on the server.

### Tiny ImageNet / ResNet18 comparison

`scripts/prepare_tiny.py --data <dataset-path>` downloads the Tiny ImageNet
Parquet mirror, verifies pinned SHA-256 checksums, and prepares native 64×64 RGB
tensors. Preparation requires `pyarrow`. The loader reserves 25 training images
per class for tuning: 95,000 images remain for training, 5,000 for tuning, and the
10,000 labeled official validation images provide heldout evaluation. Client
partitions are disjoint Dirichlet splits with alpha 0.1 and at least eight images
per client. All partitions share the normalized image tensors.

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0 python scripts/run_tiny.py \
  --data /mnt/data/mdevos/conflux/tiny-datasets \
  --output /mnt/data/mdevos/conflux/results/tiny128/shatter-r2 \
  --variant shatter-r2
CUDA_VISIBLE_DEVICES=1 python scripts/run_tiny.py \
  --data /mnt/data/mdevos/conflux/tiny-datasets \
  --output /mnt/data/mdevos/conflux/results/tiny128/conflux-v2 \
  --variant conflux-v2
```

The defaults compare Shatter degree 2 and Conflux V2 on 128 nodes, seed 0, with
16 chunks and a V2 sample of 32. Both train torchvision ResNet18 from scratch
with 200 outputs, the standard stem and BatchNorm, native 64×64 inputs, and no
augmentation. Local SGD uses five steps, batch size 32, learning rate 0.01,
momentum 0.9 and no weight decay. These are shared settings rather than the
result of a Tiny ImageNet hyperparameter search.

The heterogeneous network has an eightfold bandwidth spread, a quarter of nodes
in the fast group, and mean bandwidth 10,000,000 bytes/s. Compute speeds are
independently assigned and normalized to a mean simulated step time of 50 ms;
one-way latency is 20 ms. The higher bandwidth than the CIFAR experiments suits
the 45,195,718-byte model. Both runs use the same capacity assignment and data
partition. They run for 1,800 simulated seconds, with measurements every 300 s.
Curves use the same fixed, class-balanced 2,000-image heldout subset; the final
snapshot additionally evaluates every node on all 10,000 heldout images. The
result is the mean of individual node accuracies, without free model averaging.

`--calibrate` measures actual GPU training and evaluation throughput. Actual GPU
execution time is separate from simulated protocol time. This is an exploratory
one-seed comparison with stable membership. The executor clones tensor slices
without copying their entire backing model storage, which keeps large-model
chunk processing within memory limits. Chunk reconstruction includes BatchNorm
buffers; the focused tests check an exact full-state ResNet18 round trip.

#### Execution optimizations

Task outputs copy the complete state directly into an independent CPU model,
avoiding an unnecessary pickle round trip. Reconstruction deep-copies a private
CPU prototype instead of randomly initializing weights that it immediately
overwrites. The prototype cache is bounded to four dataset/architecture pairs
per process. Initial training models still use the original seeded initializer.
The local executor shares inputs only with built-ins that read them or clone
them before mutation; evaluation and unknown task functions keep defensive
copies. Models, chunk values and BatchNorm buffers remain independent DAG values.

PyTorch's CUDA allocator cache is retained between tasks by default; set
`SessionSettings.release_cuda_cache=True` (or `run_tiny.py --release-cuda-cache`)
when releasing unused GPU memory between tasks is needed. The changes use normal
CPU/PyTorch operations and require no GPU-specific kernel or extra dependency.
`run_tiny.py --device auto` selects CUDA when available and otherwise CPU; an
explicit device can also be supplied. No mixed precision, changed aggregation
order, fewer SGD steps, or reduced evaluation set is involved.

Tiny ImageNet runs pin the simulated payload to the original measured
45,195,718 bytes (`--model-bytes`), because pickle storage metadata can vary
slightly between processes. Other simulators retain automatic measurement unless
`SessionSettings.model_size_bytes` is explicitly supplied. This setting changes
only the modeled transfer size, not the model's state or precision.

Regression checks compare the reference wire-copy executor and optimized
executor bit for bit for both protocols, including every intermediate model,
chunk and snapshot. GPU checks additionally compare ResNet18 training state with
the experiment's deterministic algorithms enabled. The eight-node GPU smoke
runs produced identical recorded curves, full-validation accuracies and losses;
execution took 10.84 s versus 18.07 s for Shatter, and 6.82 s versus 13.38 s for
V2. These short-run timings include dataset loading and evaluation and are not
claims about simulated protocol speed or other machines.

Streaming execution also reuses reconstructions whose complete chunk references
are identical. This cache expires entries after their final duplicate task and
has both a 64 MiB tensor-payload budget and a four-entry limit. Oversized models
are recomputed normally. Set `execute_local(..., reconstruction_cache_bytes=0)`
to disable it. Retained results are never memoized, and every mutating consumer
still receives a private copy. Sink evaluation tasks avoid constructing an
unused output model; cached sink metrics additionally avoid copying a model
that will not be evaluated. Regression tests cover eviction, budget bypass,
last-use release and exact snapshot equality for both protocols.

Additional peer-reviewed comparison methods are available: Moshpit SGD
(NeurIPS 2021), repaired Teleportation (ICLR 2025; ring and hypercube), and a
full-participation D-Local-SGD ring reference. See the
[literature selection, protocol contracts and benchmark design](docs/peer-baselines.md).
All new baselines support the shared local executor and require zero momentum
and stable membership. Their validation includes exact averaging, token
handoff/causality, bandwidth-aware allocation and network-byte conservation.

The completed 84-run, three-seed CIFAR-10/GNLeNet comparison uses 64 clients,
IID/Dirichlet-alpha-0.1 data, uniform/heterogeneous networks and a common
900-second simulated budget. In the heterogeneous non-IID case, V2 reaches
34.63% ± 1.37% all-node accuracy, versus 31.92% ± 0.61% for two-phase Moshpit,
30.66% ± 0.84% for one-phase Moshpit, and 20.27% ± 0.34% for Shatter. V2 reaches
30% at 600 seconds in every seed, compared with 750–900 seconds for two-phase
Moshpit, and sends 0.565 GB versus 1.771 GB at the full budget. Moshpit wins
the uniform-network cases; for uniform IID it reaches 56.33% ± 0.67% versus
V2's 47.68% ± 0.62%, using about eight times the traffic. The
[full results and limitations](docs/peer-baselines.md#completed-results) include
Teleportation, the ring reference, loss, paired-seed differences and censored
targets. These common-parameter screenings are not individually tuned paper
reproductions or a universal ranking.

The local executor additionally enables GPU residency and training batching by
default on CUDA. Residency reuses private GPU training workspaces and a bounded
cache of repeated input-model versions (256 MiB by default). Independent ready
tasks train concurrently on separate CUDA streams, with two models per group.
Each model retains its own parameters, SGD momentum, data batches and BatchNorm
buffers. A new optimizer is created for every local training task. Data sampling
happens serially with the original per-task seeds; stochastic forwards and
non-native data pipelines retain scalar execution. DAG values, chunk averaging
and snapshot semantics stay on CPU with their original ordering.

Before enabling concurrent batches, startup calibration warms both paths,
compares three complete-task timings, and requires exact CUDA state equality
and at least a 5% median improvement over resident serial training. Unsupported
batch operations and batch memory exhaustion fall back to serial execution;
unrelated errors propagate. CPU and other devices use the reference executor.
The actual decision, fallback reasons, input-cache counters and peak prefetch
count are recorded in `training-executor.json` and progress logs. Prefetching
never publishes a task before its original heap position and holds at most one
group of extra output models. Resident resources are released on completion or
failure.

Use these options with `run_tiny.py` or `run_cifar.py`:

```sh
--no-training-batching              # Residency with serial local training
--no-gpu-residency                  # Fresh GPU workspaces for each task/group
--no-gpu-residency --no-training-batching  # Reference execution
--training-batch-size 2             # Maximum independent clients per group
--gpu-state-cache-mb 256            # Zero disables the input-version cache
```

`scripts/benchmark_training.py --data <tiny-dataset-path> --output <new-path>`
compares reference training, residency and batch widths 2, 4 and 8 using the
same client partitions and seeds. It times complete SGD tasks, validates model
states, and reports dataset loading and warmup separately. CUDA batching uses
ordinary per-model kernels on independent streams; direct model-axis
vectorization is not selected on CUDA because it failed the ResNet18 numerical
guard. No BatchNorm statistics are disabled or shared between clients.

On `sacs006` (RTX 5000 Ada, Torch 2.14.1), 32 independent ResNet18 client tasks
with five SGD steps and batch size 32 gave these three-repeat median times:

| Training executor | Seconds | Speedup over reference |
| --- | ---: | ---: |
| Reference | 2.452 | 1.00× |
| GPU residency, serial | 1.694 | 1.45× |
| GPU residency, two-client batches | 1.380 | 1.78× |
| GPU residency, four-client batches | 1.570 | 1.56× |
| GPU residency, eight-client batches | 1.614 | 1.52× |

Every output parameter and BatchNorm buffer matched the reference bit for bit.
Two-client batches were fastest and are the default. This isolated training
speedup does not include DAG construction, chunk aggregation or evaluation.
Batching can only use independent tasks already ready in the DAG; Shatter's
later training frontier often contains only one such task. The production
calibration can therefore select serial training on another workload/platform.
The [benchmark and execution plan](output/tiny-imagenet-validation/gpu-training/plan.json)
and [raw timing results](output/tiny-imagenet-validation/gpu-training/benchmark.json)
record the exact source and commands.

#### Tiny ImageNet results on 128 nodes

The one-seed heterogeneous runs completed on `sacs006`. At 1,800 simulated
seconds, mean individual-node heldout accuracy on all 10,000 images was:

| Protocol | Full heldout accuracy | Simulated bytes sent | Local training tasks |
| --- | ---: | ---: | ---: |
| Shatter degree 2 | 1.42% | 505.42 GB | 5,660 |
| Conflux V2 | 4.55% | 204.85 GB | 2,304 |

V2 sends 59.5% fewer bytes and reaches 2% on the fixed curve subset at the
900-second checkpoint; Shatter remains below 2% at every checkpoint in this
budget. Accuracy is still low for both methods. These early learning results
need a longer training budget and additional seeds to assess convergence.

Full reference and optimized executions produce exactly the same complete DAGs,
all 128 per-node metrics at all six checkpoints, final full-validation metrics,
training-task counts and modeled byte totals. Original execution took 34.49 min
for Shatter and 14.02 min for V2. The first optimization pass reduced those times
to 13.95 min and 5.35 min respectively (observed 2.47× and 2.62× improvements).
These actual execution times include evaluation and reflect concurrent jobs on
the server; they are distinct from simulated learning/protocol speed.

The second pass (`ec3778f`) completes the same V2 workload in 4.02 min, an
observed 3.49× improvement over the original and another 25% reduction from the
first pass. It reuses 2,201 of 2,272 reconstructions (71 unique input sets),
peaks at 45,155,008 cached tensor bytes, and releases the cache by completion.
All 768 individual-node curve rows, 128 full-validation rows, the complete DAG,
traffic and training-task counts match both earlier V2 executions exactly.
That pass also preserves the original eight-node Shatter full-evaluation
comparison; the Shatter timing above measures the first pass.
Its focused validation passed 45 tests on Linux/CUDA and 43 on macOS/CPU, with two
GPU-only checks skipped on the CPU host. The saved
[optimization audit](output/tiny-imagenet-validation/tiny128-memoized/equivalence.json)
contains the measured timings and final cache counters.

The GPU training pass (`03da38a`) further reduces full execution time, including
dataset loading, startup calibration and evaluation:

| Protocol | Previous optimized execution | GPU residency + batching | Time saved |
| --- | ---: | ---: | ---: |
| Shatter degree 2 | 13.95 min | 11.25 min | 19.4% |
| Conflux V2 | 4.02 min | 2.61 min | 35.0% |

Both use the same new source, with the opt-out flags at their defaults. For
each protocol, all 768 per-node curve rows, 128 final full-validation rows,
the complete DAG, traffic and training counts exactly match the original and
first optimized executions. V2 also exactly matches the memoized execution.
These are single observed runs on GPU 1 while the existing long Shatter job
continues on GPU 0, sharing server CPU/memory; they are not controlled timing
replicates. The one-seed learning results above are unchanged.

Shatter batches 142 of 5,660 tasks, whereas V2 batches 1,034 of 2,304; most later
Shatter training tasks become ready individually. Both input-version caches
peak at 45,155,008 bytes and release their contents by completion; prefetching
holds at most two output models. Neither run required a fallback. Validation
passes 62 tests on Linux/CUDA and 56 on macOS/CPU, with six CUDA checks skipped
on the CPU host. The [complete GPU execution audit](output/tiny-imagenet-validation/gpu-training/equivalence.json)
records every comparison and final execution counter. Raw DAGs and per-node
files remain under `/mnt/data/mdevos/conflux/results/tiny128-gpu-training`.
The existing long jobs retain their original frozen execution source.

The original execution source is `d14b771`; the paired first-pass optimized
source is `ce1c7a4`. Both protocols within each pair share the same execution
source. The source audit verifies every module against its commit and records
macOS transfer sidecars that affected the optimized directory's raw hash.

See the [accuracy curves](output/tiny-imagenet-validation/tiny128-optimized/accuracy.png),
[results table](output/tiny-imagenet-validation/tiny128-optimized/table.md),
[configuration and snapshot audit](output/tiny-imagenet-validation/tiny128-optimized/audit.json),
[complete reference equivalence check](output/tiny-imagenet-validation/tiny128-optimized/equivalence.json),
and [source audit](output/tiny-imagenet-validation/source-audit.json).
Full DAGs and per-node snapshot CSVs remain on the server in
`/mnt/data/mdevos/conflux/results/tiny128` and `tiny128-optimized`.

A longer follow-up was launched on `sacs006` on 18 October 2026: the same
128-node, seed-0, heterogeneous Dirichlet-alpha-0.1 comparison, extended to
18,000 simulated seconds. Shatter degree 2 and Conflux V2 each use one GPU.
Learning, data, model and capacity settings are unchanged; snapshots use the
same balanced 2,000-image subset every 900 seconds, followed by all 10,000
heldout images at the final snapshot. The
[launch plan](output/tiny-imagenet-validation/tiny128-long18000/plan.json)
records the source hash, exact commands, logs and server output directory.
Both runs completed and passed the common-source/configuration audit. Every
one of the 20 checkpoints contains all 128 nodes, all metrics are finite, and
all 256 per-node rows at 900/1,800 seconds exactly match the earlier short run
for each protocol. The final full-validation files each contain 128 rows.

| Protocol/readout after 18,000 simulated seconds | Full 10,000-image accuracy | Full-validation loss | Total traffic |
| --- | ---: | ---: | ---: |
| Shatter degree 2, all-node checkpoint mean | 11.09% | 4.2013 | 4.795 TB |
| Conflux V2, all-node checkpoint mean | 22.20% | 3.5099 | 2.045 TB |
| Conflux V2, latest completed sample aggregate | 22.70% | 3.4854 | Same V2 traffic |

V2 uses 57.4% less traffic at this common time budget. On the fixed 2,000-image
curve subset, the all-node mean first reaches 10% at 4,500 simulated seconds
for V2 and 16,200 seconds for Shatter (3.6× earlier at the saved checkpoint
resolution). V2 reaches 20% at 11,700 seconds; Shatter never reaches 15% or 20%
within the budget. Neither protocol reaches 25%. Unreached targets remain
explicitly censored in the saved time-to-accuracy table.

The final curve-subset means are 11.41% for Shatter and 22.98% for V2, with
23.10% for V2's latest aggregate. These curve metrics are distinct from the
10,000-image final metrics above. V2's freshest saved model is identified by
maximum recorded round (712), and all 32 recipients have identical accuracy
and loss on both evaluation sets. At the final snapshot, V2's stored node
checkpoints are on average 2.625 rounds behind this aggregate, with a maximum
age of 15 rounds. The additional aggregate curve measures a different quantity
from the all-node mean. Shatter retains distinct mixed models and has no single
shared aggregate. These records evaluate fixed snapshots, not every all-reduce;
the saved DAG contains references and protocol history, not trained weights.

This remains a **one-seed** result, not evidence of a universal advantage or
final convergence. Actual execution took 103.17 min for Shatter and 35.47 min
for V2 on the original frozen executor; those hardware times are separate from
the five simulated protocol hours and from the newer GPU-backend validation.
Intermediate cumulative traffic was not recorded in these original jobs, so
the traffic comparison uses measured totals at the common final budget.

See the [combined scientific figure](output/tiny-imagenet-validation/tiny128-long18000/comparison.png),
[final summaries](output/tiny-imagenet-validation/tiny128-long18000/summary.csv),
[audit](output/tiny-imagenet-validation/tiny128-long18000/audit.json),
[aggregate and staleness checkpoints](output/tiny-imagenet-validation/tiny128-long18000/latest-v2-aggregate.csv),
and [censored time to accuracy](output/tiny-imagenet-validation/tiny128-long18000/time-to-accuracy.csv).
Raw DAGs and per-node files remain in the planned server results directory.
