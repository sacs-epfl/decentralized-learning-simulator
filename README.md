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
