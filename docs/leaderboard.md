# Algorithm leaderboard

Run a paired comparison of algorithms against one fixed learning and network
scenario. Budgets are **simulated protocol seconds**, independent of the wall
time needed to execute training. There is no special 900-second limit.

```sh
# Inspect the full resolved plan without loading a model or dataset.
python scripts/run_benchmark.py experiments/leaderboard.yaml --plan

# Offline screening: every registered algorithm, three paired seeds.
python scripts/run_benchmark.py experiments/leaderboard.yaml

# In another terminal, view live results or an already completed benchmark.
python scripts/serve_leaderboard.py output/leaderboard
```

Open <http://127.0.0.1:8765>. For execution on a remote host, forward the port:
`ssh -L 8765:127.0.0.1:8765 HOST`. The server binds to loopback by default and
serves only the dashboard, its script, and the two leaderboard exports.
It can start before the benchmark creates its output directory.

To select between experiment settings on one dashboard:

```sh
python scripts/serve_leaderboard.py --catalog experiments/leaderboard-settings.json --port 8766
```

The setting picker keeps each scenario's runs, rankings, curves, and CSV separate.
The catalog currently includes the completed IID/homogeneous 900-second run and
the Dirichlet alpha 0.1 / heterogeneous run. The latter is the default setting.
The budget picker defaults to **900 seconds** when available (otherwise the
largest available budget), and preserves a user's choice during live refreshes.
Setting links can be bookmarked, e.g. `?setting=dirichlet01-heterogeneous`.
To add future IID/heterogeneous or non-IID/homogeneous combinations, add their
result directories to this catalog and restart the server; no data is pooled
across settings. A catalog may include a setting before its output exists.

Use the Python environment containing the simulator requirements. No additional
dashboard framework, database, external service, or JavaScript build is needed.

## Budgets and scenarios

```sh
# Short smoke run with two algorithms and a single seed.
python scripts/run_benchmark.py experiments/leaderboard.yaml \
  --variants dpsgd-ring qgm --budgets 2 4 --seeds 0 \
  --evaluation-period 1 --output output/leaderboard-smoke

# Longer horizons, derived from the same continuous training trajectories.
python scripts/run_benchmark.py experiments/leaderboard.yaml \
  --budgets 60 300 1800 --seeds 0 1 2 3 4 \
  --evaluation-period 30 --output output/leaderboard-long
```

The bundled synthetic workload is an offline mechanism check, not evidence of
performance on a real dataset. Its 20/60-second budgets are inexpensive starting
points. Copy the file and set the shared dataset, model, partitioner, alpha,
dataset paths/revisions, device, and client count for a substantive comparison.
Choose `evaluation_split: test` for datasets such as offline digits that do not
provide validation data; the dashboard marks this as exploratory.

An experiment uses the normal settings/algorithm option names:

```yaml
version: 1
name: my-comparison
output: ../output/leaderboard-my-comparison
benchmark:
  budgets: [60, 300, 1800]
  seeds: [0, 1, 2]
  evaluation_period: 30
  accuracy_targets: [50, 70, 80]
  network:
    profile: uniform
    bandwidth_bytes_per_second: 1000000
    compute_step_ms: 200
settings:
  dataset: synthetic
  model: mlp
  peers: 8
  device: cpu
  partitioner: iid
  evaluation_split: validation
  learning_rate: 0.05
  momentum: 0
  local_steps: 1
runs:
  - name: dpsgd-ring
    settings: {algorithm: dpsgd, topology: ring}
  - name: choco
    settings: {algorithm: choco, choco_keep_ratio: 0.1}
```

Budgets, seeds, and evaluation period accept CLI overrides. `--variants` selects
named variants. Budgets/seeds must be unique nonempty lists of positive and
nonnegative integers, respectively. The runner automatically includes time zero,
each regular evaluation time, and **every exact budget**, even when the period
does not divide a budget. All budgets use one trajectory up to the largest
horizon, avoiding duplicate training. Changing that horizon may affect protocols
whose schedules depend on the total duration; use the same full budget list for
paired comparisons.

Each named variant expands across the same seeds. Scenario fields cannot change
inside a variant: only algorithm, graph, and optimizer/protocol options may vary.
For another partitioner, model, network profile, or client count, use another
benchmark output directory. A single leaderboard never pools different scenarios.

`compute_step_ms` is the simulated duration of one local SGD step. A block of
`local_steps` costs that multiple; algorithms account for their own extra work
and communication. Uniform profiles give all clients the configured capacities.
Heterogeneous profiles draw compute and bandwidth factors independently from
`[0.5, 1, 2, 4]`, normalize each set to mean one, and reuse them for each variant
at the same seed. The shared `network_latency_ms` setting is also supported.
FedAvg retains its implementation's unconstrained server bandwidth; client
traffic is still charged and server transmissions are included in sent bytes.
Version 1 supports stable membership with these explicit capacities; traces and
churn are rejected rather than silently changing which clients are measured.

Graph choices remain protocol-specific and are visible in resolved settings.
A fixed physical capacity scenario does not imply identical logical communication
graphs. Use named graph variants where an algorithm supports them. The example
includes legacy algorithms and research adaptations listed in the README; a
leaderboard entry does not strengthen their validation status or establish paper
equivalence.

## Measurements and interpretation

`benchmark.evaluation_model` selects the readout (default `local`):

- `local`: evaluate every physical participant's latest completed local model
  on the same split/subset, then average client accuracies.
- `global`: take the **uniform arithmetic parameter average of all participants'
  latest completed local models**, then evaluate that single global model once
  on the selected split/subset. This is parameter averaging, not prediction
  ensembling or averaging accuracies. FedAvg's server is not an extra input.

Both modes include the initial model for clients that have not updated.
Aggregation/evaluation is an **offline diagnostic**: its execution cost contributes
to wall time, while collection traffic, aggregation latency, and evaluation do
not consume modeled protocol resources. The global average never feeds back into
client training. Events at identical simulated times follow the simulator's
deterministic event order. GNLeNet has no BatchNorm buffers; the shared aggregation
routine averages floating state and uses the existing maximum rule for integer
buffers on other supported models.

The collector publishes only a consecutive prefix of complete, unique checkpoints:
all client scores for local readout, or one global score for global readout.
Missing or invalid final checkpoints fail the run. A completed checkpoint may
appear while its worker is still processing later checkpoints.

For each selected budget the dashboard and exports contain:

| Measurement | Definition |
|---|---|
| Accuracy | Mean client accuracy (`local`) or single averaged-model accuracy (`global`) at the exact budget; primary sort key |
| Accuracy SD | Sample standard deviation across seeds; blank for one seed |
| Average accuracy | Area under the checkpoint curve from zero to the budget, divided by the budget; left-constant integration, in percentage points |
| Worst 10% accuracy | Local readout only: mean of the lowest `ceil(clients / 10)` client accuracies, then mean across seeds; omitted for global readout |
| Traffic | Total sent bytes, counted once, including bytes sent by unfinished transfers at the checkpoint |
| Best observed accuracy | Maximum checkpoint mean accuracy within the budget, calculated per seed then averaged |
| Time to accuracy | First observed checkpoint reaching the target; no interpolation |
| Full-run wall seconds | Worker startup, simulation, training, evaluation, and output through the largest configured horizon |

Time-to-target JSON includes reached/evaluated seed counts and the censoring
horizon. Its displayed mean covers successful seeds only; never compare that mean
without the reach rate. Full-run wall seconds is repeated across budget views
and must not be interpreted as execution time to the selected smaller budget.

Results are provisional until **every variant** has the full paired seed set at
that budget. Until then, means and curves can cover different numbers of seeds;
official numeric ranks are withheld. Failed/interrupted runs retain their status
and logs, are excluded from aggregate scores, and keep the ranking provisional.
Exact ties share a rank. Close means do not imply statistically distinct winners.

The default synthetic example uses common hyperparameters and is explicitly an
**untuned screening**. Named runs support hand-selected configurations. The
optional tuning pipeline below selects LRs on validation data before running
fresh seeds on test data. Model-state checkpoint selection/resume is not
implemented. Best observed accuracy is a finite-budget observation, not an
estimate of the unlimited-time optimum or a saved deployable checkpoint.

## Global-model rerun with frozen LRs and a longer horizon

The follow-up configuration reuses the previously selected LRs without another
search, measures a single averaged global model, and runs each trajectory through
900 simulated seconds:

```sh
python scripts/run_benchmark.py experiments/leaderboard-cifar10-32-global900.yaml --plan
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_benchmark.py experiments/leaderboard-cifar10-32-global900.yaml
python scripts/serve_leaderboard.py output/leaderboard-cifar10-32-global900
```

It preserves CIFAR-10/GNLeNet, 32 IID clients, the original capacities and network
latency, and evaluation seeds 10/11/12. Frozen LRs are FedAvg/D-PSGD/QGM `0.01`,
CHOCO `0.1`, and DeMo `0.001`, selected by the earlier 300-second **local-model**
validation search. They are reused as requested, rather than retuned for this
global readout or longer horizon. There are **15 runs**, with four workers across
two GPUs and two workers sharing each GPU. Budget views are 60/300/900 seconds;
one global model is evaluated on all 10,000 test images every 60 seconds.
The previous 300-second run remains in its original output directory.

Each global checkpoint uses equal weight for every client's latest available
model, including initially idle clients. The mean model is an evaluation branch
only. Its collection/aggregation overhead is excluded from simulated protocol
time and reported network bytes; wall time includes the actual averaging and
evaluation. The dashboard labels this readout and omits worst-client accuracy.
Best observed accuracy at 900 seconds is a finite-budget result, not evidence of
an unlimited-time optimum.

## Dirichlet alpha 0.1 and heterogeneous hardware

The extreme follow-up uses the same five algorithms and frozen LRs as the global
900-second experiment, with 32 clients and seeds 10/11/12:

```sh
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_benchmark.py experiments/leaderboard-cifar10-32-dirichlet01-heterogeneous.yaml
```

Only two scenario fields change: `settings.partitioner: dirichlet` with
`alpha: 0.1`, and `benchmark.network.profile: heterogeneous`. Compute-step times
and bandwidths have independent client assignments, drawn from the existing
`[0.5, 1, 2, 4]` factors and normalized to mean one. Client capacities are paired
across algorithms for each seed. Base mean step time remains 20 ms, base mean
bandwidth 1 MB/s, and latency 10 ms. The native CIFAR partitioner enforces at
least eight examples per client. Learning rates are reused without tuning.

There are 15 runs, four concurrent workers across two GPUs, global parameter
averaging every 60 seconds, and budget views of 60/300/900 seconds. Select
**Non-IID (Dirichlet α = 0.1) · heterogeneous hardware** in the catalog dashboard
to monitor it. Experiments and the dashboard run on the remote host independently
of the browser or SSH tunnel. Closing a laptop stops its tunnel but does not stop
the remote workers; reconnect the tunnel to view progress again.

## Automated equal-budget LR tuning

The first real-data configuration compares **FedAvg, D-PSGD, QGM, CHOCO-SGD, and
DeMo**, using CIFAR-10, GNLeNet (85,354 parameters), and 32 clients:

```sh
python scripts/prepare_dataset.py uoft-cs/cifar10 --data data/cifar10
python scripts/run_benchmark.py experiments/leaderboard-cifar10-32.yaml --plan
python scripts/run_benchmark.py experiments/leaderboard-cifar10-32.yaml
python scripts/serve_leaderboard.py output/leaderboard-cifar10-32
```

Skip preparation if `data/cifar10` already contains a prepared DatasetDict.
The configuration uses CUDA device 0, IID clients, full participation for FedAvg,
ring communication for D-PSGD/QGM/CHOCO, and DeMo's ring all-gather. All clients
have 1 MB/s bandwidth, a modeled 20 ms local step, and 10 ms network latency.
The measured GPU execution speed does not change these modeled capacities.
The native CIFAR pipeline normalizes images and uses no data augmentation.
FedAvg's server retains the implementation's unconstrained bandwidth.

Append a top-level `tuning` section to a benchmark configuration:

```yaml
tuning:
  budget: 300
  seeds: [0]
  evaluation_period: 300
  evaluation_examples: 1000
  metric: accuracy
  learning_rates:
    fedavg: [0.003, 0.01, 0.03, 0.1, 0.3]
    dpsgd-ring: [0.003, 0.01, 0.03, 0.1, 0.3]
    qgm: [0.003, 0.01, 0.03, 0.1, 0.3]
    choco: [0.003, 0.01, 0.03, 0.1, 0.3]
    demo: [0.0001, 0.0003, 0.001, 0.003, 0.01]
```

The LR keys must exactly match named variants. Every algorithm must have the
**same number of unique, positive candidates**. Trial count, seed set, simulated
duration, evaluation cadence, and validation subset size are identical across
algorithms. Equal budget means equal allocated trials and simulated seconds;
actual CPU/GPU wall time and completed optimizer updates can differ. There is no
early elimination, extra refinement, or transfer of trials between algorithms.
LR ranges differ to account for different update rules. These ranges are a
declared search space, not a guarantee that each algorithm's optimum is inside it.
Boundary winners are recorded without silently adding trials.

The pipeline forces every search trial to use `validation`. It selects the LR
with the highest mean accuracy at the exact tuning budget, averaged across all
tuning seeds. `metric: average_accuracy` instead selects the mean curve integral
and requires intermediate checkpoints. Lower LR breaks an exact score tie.
Every candidate must complete on every tuning seed; failed or incomplete
searches preserve their artifacts and **block final evaluation**, so a partially
evaluated grid cannot silently select a winner. No automatic retries are added.

Final benchmark settings must use `evaluation_split: test`. Final seeds must be
disjoint from tuning seeds. After selection, LR is frozen per algorithm for all
final seeds and budgets. Other algorithm options stay fixed throughout both
phases. The checked-in first experiment runs **25 tuning trials** (five LRs × one
seed × five algorithms), then **15 held-out runs** (three seeds × five algorithms).

CIFAR's fixed stratified holdout reserves 5,000 of the original 50,000 training
images before client partitioning, leaving 45,000 for training. Tuning uses the
same deterministic, class-balanced 1,000-image subset of that validation split.
Final evaluation uses all 10,000 original test images, on fresh seeds 10/11/12,
at time zero and every 60 seconds through 300 seconds. Budget views are 60 and
300 seconds; the single selected LR optimizes the **300-second** criterion.
One tuning seed keeps this first search manageable; additional tuning seeds are
supported and multiply every algorithm's allowance equally. This short, IID
experiment is a first comparison, not a long-horizon or heterogeneous-data claim.

The root dashboard automatically follows the active phase. During tuning, it
shows validation candidates and the declared search budget. During evaluation,
it shows the five algorithms, frozen LRs, and held-out test curves. Selecting
settings or time horizons based on these test results requires a new independent
evaluation before making an unbiased final claim.

```text
OUTPUT/
  pipeline.json               # phase, status, search budget, selected rates
  resolved.json               # complete resolved pipeline specification
  source.yaml
  dataset-source.json         # prepared dataset provenance, when available
  selection.json              # validation scores per candidate/seed and winners
  leaderboard.json            # live mirror of active phase plus pipeline status
  leaderboard.csv
  tuning/                    # ordinary benchmark artifacts for every candidate
  evaluation/                # ordinary benchmark artifacts with frozen LRs
```

The same `run_benchmark.py` command handles both phases. The normal `--output`,
`--variants`, and final `--budgets`/`--seeds` overrides still work. Changing final
budgets does not change `tuning.budget`; keep the distinction explicit in reports.
Use a new output directory for every search. Phase manifests retain separate
source/package provenance, raw curves, exact trial settings, logs, and failures.

## Artifacts and failures

```text
OUTPUT/
  source.yaml                 # original input; CLI overrides are in the manifest
  manifest.json               # resolved benchmark, run states, source provenance
  leaderboard.json            # atomic live snapshot, per-seed curves and metrics
  leaderboard.csv             # one row per budget and variant
  runs/VARIANT-sSEED/
    resolved.json             # executable settings and exact checkpoint schedule
    manifest.json             # terminal run state, wall time, errors, curve
    run.log
    network.json              # passive traffic measurements by simulated time
    execution.json            # actual SessionSettings and artifact path
    training-executor.json
    data/.../                 # workflow DAG, settings, per-peer and merged CSVs
```

Each run has its own process to isolate model, dataset, RNG, and executor state.
Runs execute sequentially unless `benchmark.execution_devices` declares worker
slots. For example `[cuda:0, cuda:1, cuda:0, cuda:1]` runs at most four workers,
two per GPU, and assigns each new run to the next free slot. Devices are recorded
in each run's resolved settings; repeated devices intentionally share a GPU.
CPU thread counts remain per worker. Actual wall times include resource
contention, while modeled protocol time and network capacities remain fixed.
Ctrl-C/SIGTERM stops every active worker process group and leaves queued runs
planned. The parent polls metrics once per second; the dashboard refreshes every
two seconds.
Outputs are read-only from the dashboard. Raw CSVs remain available for audit.

Existing output directories are refused. Failed workers do not stop the remaining
variants; the benchmark exits nonzero. Ctrl-C or SIGTERM stops the worker process
group, saves interrupted status, and leaves unstarted runs planned. No checkpoint
resume is implied. Source/package provenance is recorded using the shared
experiment provenance collector.
