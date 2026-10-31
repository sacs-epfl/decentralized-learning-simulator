# Expanded Conflux V2 study (31 October 2026)

The local `conflux-v2` branch is rebased onto upstream main `bda202a`, including
YAML/JSON configurations, Hugging Face/Flower data loading, LoRA, and the new
protocol/compute APIs. The pre-rebase tip `401c3db` is retained by
`codex/conflux-v2-before-rebase` and the unchanged remote research branch. Because
main previously imported shared changes through curated history, the research
branch's net changes against merge base `4f2d611` were consolidated into one
commit before rebasing; its old individual commits are available in the backup.
The archived original experiment branch and historical server directories were
not modified. Publishing this rewritten history would require a force-with-lease
push; no remote branch has been rewritten.

Conflux V2 now uses explicit `TaskRef` dependencies, `create_compute_task`,
`send`, and `send_control`. Training compute is charged once by the shared
completion handler. Tests check common contributor sets, exact reconstruction,
partial-success fallback, overlap, serialization, and wire-byte accounting.

The protocol runs in the same shared configuration/local executor as baselines.
`scripts/benchmark_convergence.py` writes a normal version-1 `config.json` per
run and executes `main.py --config ...` in a fresh process. Its study description
is `experiments/conflux-v2/study.json`; this outer study file is consumed by the
benchmark script, while generated individual configurations also work directly
with `scripts/run_experiment.py` and `main.py`.

## Design

- Four workloads: digits/MLP (2,410 parameters), CIFAR-10/GNLeNet (85,354),
  Tiny ImageNet/MobileNet V3 Large, Tiny ImageNet/ResNet18. The two Tiny ImageNet
  workloads hold the dataset/input resolution fixed while varying architecture
  and model size. Native images are 32x32 for CIFAR and 64x64 for Tiny ImageNet;
  models train from scratch, without augmentation or pretrained weights.
- Six algorithms: Conflux V2, Shatter r=2, D-PSGD, Moshpit, Teleportation ring,
  and quasi-global momentum. Sixteen nodes; sampled methods use four trainers;
  Moshpit uses groups of four; V2/Shatter use sixteen chunks; V2 can place
  reducers across all sixteen nodes and uses immediate uploads with a full
  contributor set. D-PSGD and QGM use degree-three regular graphs.
- IID and Flower Dirichlet alpha 0.1 partitions. Seeds 0, 1, 2 are paired across
  algorithms. All methods see the same disjoint client data per seed.
- Four capacity profiles: uniform, network only, compute only, and both.
  Network capacity has an 8:1 ratio, with a quarter of nodes fast. Compute step
  times use independently drawn multipliers from 0.5, 1, 2, 4. Normalize both
  factors to preserve their arithmetic means across profiles. Independently
  seeded streams keep each assignment fixed when toggling the other factor.
- Common network latency is 20 ms. Bandwidth is 4 KB/s (digits), 100 KB/s
  (CIFAR), 10 MB/s (Tiny). Mean modeled SGD step time is 20 ms for small models,
  50 ms for Tiny models. These are controlled edge-device assumptions, **not
  measurements of the RTX GPU**. Calibration reports actual GPU costs separately.
  Serialized payload sizes are pinned once from calibration for exact pairing.
- Each workload/algorithm/partitioner gets two learning-rate candidates and the
  same simulated tuning horizon, with seed 10001 and joint heterogeneity.
  Select by area under validation accuracy between the first and last common
  checkpoints. Freeze that rate across all four confirmation capacity profiles.
  These two-point searches are a limited, equal tuning budget; they do not
  establish globally optimal baseline hyperparameters.
- Ordinary protocols use five local SGD steps and zero momentum. QGM keeps
  its published one-step update with beta=mu=0.9; its actual compute per step
  is charged, so its block is shorter. No extra free local steps are added.
- 96 tuning runs, followed by 576 confirmation runs. Each GPU executes at most
  one run at a time. A run records the exact resolved config, source SHA-256,
  GPU index, wall time, model bytes, training tasks, compute totals, peak CUDA
  allocation, raw DAG, and per-node metrics. Resume rejects changed source or
  settings and archives failed attempts instead of overwriting their evidence.

## Evaluation

The primary curve is **mean validation accuracy of all sixteen nodes' latest
completed stored checkpoints at fixed simulated times**, including idle sampled
nodes. There is no uncharged global average. V2's newest completed aggregate and
node staleness are recorded separately as a diagnostic; this is not substituted
for the common metric when ranking algorithms.

Validation is reserved before client partitioning. Digits uses fixed stratified,
disjoint training/validation/test pools across seeds; this changes the older
benchmark's two-way split, so historical numbers are not pooled with this study.
CIFAR/Tiny holdouts follow main's fixed-seed loader. CIFAR and Tiny curves use
fixed class-balanced validation subsets of 1,000 examples. Every confirmation
run also evaluates all final node checkpoints on the **full final test split**;
heldout tasks are persisted in the executed DAG and are never used for tuning.

Time to accuracy uses the first scheduled threshold crossing. Unreached targets
are right-censored at the time budget. Report the preceding/following checkpoint
bounds: a strict speed advantage requires nonoverlapping intervals. Paired
speedup estimates and 95% seed-bootstrap intervals are emitted only when all
three seeds reached the target for both algorithms. Three-seed intervals are
coarse. Thresholds within a run are correlated, so win counts are descriptive.

The horizons are 60 s (digits), 600 s (CIFAR), and 1,800 s (Tiny), with twelve
checkpoints. These finite budgets measure early convergence to prespecified
accuracy targets, not convergence to an optimization limit. Results apply to
stable membership, these bandwidth/compute assumptions and the implemented
baseline specializations. They cannot establish that V2 is universally fastest.
Simulated seconds are the protocol's elapsed time; GPU execution wall seconds
measure the cost of evaluating the simulator's DAG, not deployed latency.

## Server execution

All new code, data, caches, temporary files, logs and results are under
`/mnt/data/mdevos/conflux-v2-20261031` on `sacs006`, using `conflux-gpu`.
Existing cached parquet files are verified against their saved SHA-256 values.
The prepared datasets retain split fingerprints and source provenance. Tiny's
raw parquet image structs are explicitly cast to the Hugging Face Image feature
before saving so the current loaders receive decoded PIL images.

```sh
source /mnt/data/mdevos/conflux-v2-20261031/conflux-v2-env.sh
python scripts/benchmark_convergence.py --stage plan \
  --study experiments/conflux-v2/study.json --data ../datasets --output ../results/full
python scripts/benchmark_convergence.py \
  --study experiments/conflux-v2/study.json --data ../datasets --output ../results/full \
  --gpus 0 1
# Safe to repeat the same command to resume an identical interrupted study.
python scripts/benchmark_convergence.py --stage report \
  --data ../datasets --output ../results/full
```

`REPORT.md`, `summary.csv`, `time-to-accuracy.csv`, `paired-comparisons.csv`,
`speedup-summary.csv`, and per-workload PNGs are generated from completed runs.
A `COMPLETE` marker is written only after every requested run succeeds.

For a small ordinary YAML sweep without the outer study driver:

```sh
python scripts/run_experiment.py experiments/conflux-v2/smoke.yaml --plan
python scripts/run_experiment.py experiments/conflux-v2/smoke.yaml
```

For a compact live progress snapshot, use `--stage status` with the same output
and data paths. Read the currently running run's `execution-progress.json` or
`run.log` for DAG completion counts. The output directory is exclusively locked
by the study driver, so starting a second driver cannot corrupt active runs.

## Launch validation

The integrated branch passed the complete 422-test Linux/CUDA regression suite.
After adding final capacity/heldout guards, progress artifacts, and the exclusive
study lock, the focused configuration/scheduler/resume suite passed 53 tests.
Eight short GPU pilots completed: Conflux V2 and Moshpit on all four workloads.
These pilots validate execution and estimate cost; they are not the confirmatory
comparison. They use the separate `code` snapshot; the full study uses frozen
`code-study` and the calibrated, pinned model sizes in the checked-in study file.
