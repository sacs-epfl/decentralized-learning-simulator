# Experiment configuration

Run a single experiment or a sweep through the existing simulator and DAG
executor with one YAML or JSON file:

```sh
python scripts/run_experiment.py experiments/smoke.yaml
```

The bundled smoke example runs four offline CPU experiments: IID and Dirichlet
data, each with seeds 0 and 1. No dataset/model download or brokers are needed.
Each run uses a separate process and output directory, isolating model, dataset,
RNG and task-counter state between configurations.

Preview resolved settings before launching anything:

```sh
python scripts/run_experiment.py experiments/food101.yaml --plan
```

Planning validates the file, expands its combinations, and prints the effective
CLI options. It never loads models/datasets or creates output directories.
The Food-101 example describes the 16-node, 3-regular D-PSGD LoRA comparison with
IID and Dirichlet alpha 0.1, pinned source/model revisions and a common seed.
Without `--plan`, it uses the full dataset and performs training.

## File format

```yaml
version: 1
name: comparison
execution: local
output: ../output/comparison
settings:
  algorithm: dpsgd
  dataset: synthetic
  model: mlp
  device: cpu
  peers: 4
  topology: ring
  rounds: 3
  local_steps: 2
  batch_size: 16
  learning_rate: 0.1
  momentum: 0
  alpha: 0.1
  test_interval: 1
matrix:
  partitioner: [iid, dirichlet]
  seed: [0, 1]
```

- `version`: required, currently `1`.
- `name`: optional experiment name; defaults to the file's stem.
- `execution`: `local` (default), `simulate`, or `distributed`.
- `output`: output root, relative to the configuration file unless absolute.
  Defaults to `output/NAME` beside the configuration file.
- `settings`: shared settings using CLI destination names, without leading
  `--`. Underscores and hyphens are interchangeable: `learning_rate` and
  `learning-rate` are equivalent. Use positive destination names for opt-outs,
  e.g. `gpu_residency: false` and `training_batching: false`.
- `matrix`: optional Cartesian product of settings and their candidate values.
- `runs`: optional named configurations, as an alternative to `matrix`.

Settings stay flat, matching the CLI; for example `batch_size: 32`, rather than
a nested `learning` object. Numeric/boolean values must have their actual YAML/
JSON types. Unknown keys, duplicate keys, invalid choices, nonfinite numbers,
empty axes and duplicate matrix values are rejected. Only safe YAML types are
accepted; Python constructors and expressions are not evaluated. Matrices are
limited to 10,000 runs per file.

## Comparing algorithms with different options

Use named runs for combinations that are not a Cartesian product. Shared
settings are combined with each run's settings:

```yaml
version: 1
name: algorithms
output: ../output/algorithms
settings:
  dataset: synthetic
  model: mlp
  device: cpu
  peers: 4
  rounds: 2
  momentum: 0
  test_interval: 1
runs:
  - name: dpsgd-ring
    settings:
      algorithm: dpsgd
      topology: ring
  - name: moshpit
    settings:
      algorithm: moshpit
      group_size: 2
```

Names must start with a letter/digit and use only letters, digits, `.`, `_`, or
`-`. Named runs must be unique. A file can use `matrix` or `runs`, not both.
See [`experiments/algorithms.json`](../experiments/algorithms.json) for the JSON
equivalent. Algorithm-specific options keep their CLI prefixes, such as
`qgm_topology`, `choco_keep_ratio`, and `dspod_sgd_probability`.

## Overrides and paths

Normal CLI options override settings in the file:

```sh
python scripts/run_experiment.py experiments/smoke.yaml \
  --seed 42 --rounds 5 --output /tmp/my-comparison
```

Precedence is CLI overrides, then matrix/named-run settings, then shared
settings, then CLI defaults. A CLI override replaces the corresponding matrix
axis; `--seed 42` above produces two runs rather than four identical-seed runs.
Explicit learning settings also take precedence over the FEMNIST defaults.

Paths in the file are relative to the configuration file. This includes
`dataset_base_path`, `dataset_cache_dir`, `feedback_replay`, and local entries in
`dataset_data_files`. URLs remain URLs. Model/tokenizer/dataset identifiers stay
unchanged unless they are explicit local paths (`./`, `/`, `~`) or identify an
existing path beside the file. CLI paths are relative to the invocation directory.

`--output` selects a fresh output root, and `--execution` overrides the execution
mode. `work_dir`, `local`, `dry_run`, `from_dir`, and `config` are reserved inside
`settings`; use the corresponding top-level controls. Saved workflow replay
continues to use the existing `main.py --from-dir` interface.

## Execution modes

- **Local:** generates and validates the normal protocol DAG, then executes it
  using `execute_local()`. This supports both ordinary models and LoRA.
- **Simulate:** generates and validates the DAG without executing its numerical
  tasks. Simulation may still need model initialization to calculate payload
  sizes; `model_size_bytes` supplies an explicit modeled size. Use `--plan` for
  a preview that never accesses models/datasets.
- **Distributed:** uses the normal coordinator/broker flow. Start brokers as
  usual, using the configured coordinator port. Runs execute sequentially;
  brokers shut down after each run, so start/reconnect workers for each run.

The runner uses the current Python interpreter and inherits its environment,
including `CUDA_VISIBLE_DEVICES`, `HF_HOME`, `TMPDIR`, and other cache locations.
For example, on a server with caches already configured:

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py experiments/food101.yaml \
  --device cuda:0 --output /mnt/data/mdevos/food101-config-run
```

The two partitioners run sequentially on that visible GPU. `--continue-on-error`
allows later runs to proceed after a failure; the command still exits nonzero
if any run failed. By default, execution stops after the first failure.

A single-run file also works with existing entry points:

```sh
python main.py --config single-run.yaml
python main.py --config single-run.yaml --dry-run
python main.py --local --algorithm dpsgd --dataset synthetic --model mlp --rounds 2
python scripts/run_lora.py --config single-lora-run.yaml
```

`main.py` honors the file's execution mode; `run_lora.py` remains a local LoRA
compatibility wrapper. Sweeps must use `run_experiment.py`; other entry points
reject them rather than silently selecting one combination. CLI `duration`,
`test_period`, and gossip `gl_period` values are seconds and are converted to
simulator microseconds in the shared settings factory.

## Outputs and stopping

The sweep runner refuses to reuse an existing output directory. Choose a new
`--output` for another invocation. It writes:

```text
OUTPUT/
  source.yaml                 # original configuration (or source.json)
  manifest.json               # suite status, source and package provenance
  summary.csv                 # one row per attempted run
  runs/0001-matrix-dpsgd-iid-s0/
    resolved.json             # all effective CLI settings for this run
    manifest.json             # status, actual SessionSettings, timing and metrics
    run.log                   # child stdout/stderr
    data/.../                 # normal settings.json, workflow_dag, accuracy CSVs
```

Provenance includes the Git commit, working-tree dirty status, source digest,
Python version, and relevant package versions. Per-run settings record the
resolved device and algorithm-specific parameters. `summary.csv` reports the
mean of each evaluated peer's latest recorded accuracy/loss; it does not create
or evaluate a global averaged model. Runs without evaluation have blank metric
columns. Simulated timestamps and execution wall seconds are reported separately.

Failures preserve their logs and any artifacts produced before the failure.
Ctrl-C/SIGTERM stops the active child process and marks the run interrupted.
This is experiment orchestration, not training checkpoint/resume: an interrupted
run may have partial metrics and does not imply saved final models. Output
protection and manifests are managed by `run_experiment.py`; direct legacy entry
points retain their usual run-directory behavior.
