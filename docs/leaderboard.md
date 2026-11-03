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

At each checkpoint, evaluate the latest published, completed local model for
**every physical participant**, with the initial model as fallback for clients
that have not updated. FedAvg's server is not an extra participant. All clients
use the same evaluation split/subset. No global parameter average is created.
Evaluation is passive: its real execution cost contributes to wall time, but it
does not consume modeled training/network resources. Events at identical
simulated times follow the simulator's deterministic event order.

The collector publishes only a consecutive prefix of complete, unique all-client
checkpoints. Missing or invalid final checkpoints fail the run. A completed
checkpoint may appear while its worker is still processing later checkpoints.

For each selected budget the dashboard and exports contain:

| Measurement | Definition |
|---|---|
| Accuracy | Mean client accuracy at the exact budget; primary sort key |
| Accuracy SD | Sample standard deviation across seeds; blank for one seed |
| Average accuracy | Area under the checkpoint curve from zero to the budget, divided by the budget; left-constant integration, in percentage points |
| Worst 10% accuracy | Mean of the lowest `ceil(clients / 10)` client accuracies, then mean across seeds |
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

The default example uses common hyperparameters and is explicitly an **untuned
screening**. Named runs support hand-selected configurations, but automated
equal-budget tuning, independent held-out-seed selection, model-state checkpoint
selection/resume, and final held-out testing are not implemented here. Keep tuning
on validation data and run an independent final evaluation before making research
claims. Best observed accuracy is a finite-budget observation, not an estimate of
the unlimited-time optimum or a saved deployable checkpoint.

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
Runs execute sequentially to avoid competition for the same CPU/GPU. The parent
polls metrics once per second; the dashboard refreshes every two seconds.
Outputs are read-only from the dashboard. Raw CSVs remain available for audit.

Existing output directories are refused. Failed workers do not stop the remaining
variants; the benchmark exits nonzero. Ctrl-C or SIGTERM stops the worker process
group, saves interrupted status, and leaves unstarted runs planned. No checkpoint
resume is implied. Source/package provenance is recorded using the shared
experiment provenance collector.
