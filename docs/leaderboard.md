# Algorithm leaderboard

## Non-IID TPE methodology campaign (4 November 2026)

### Dataset/model combinations (6 November 2026)

The current CIFAR-100 held-out evaluations run for **five simulated hours
(18,000 seconds)** with the original tuning winner, **H=20 and
LR=0.028495662538780223**. This replaces the earlier one-hour held-out runs;
their evaluation directories and duplicated root score files are removed.
The complete 40-configuration tuning history, winning parameters and original
3,600-second validation-selected checkpoint remain intact. The five-hour primary
budget is explicitly fixed by the experiment request. The leaderboard reports
each seed's highest accuracy within that budget and the loss at that checkpoint,
then averages across seeds. All four data/hardware settings use
seeds 110–112, hourly budget views and 300-second checkpoints. The extension
follows inspection of prior test curves and remains exploratory.

```sh
python scripts/run_methodology.py --catalog experiments/leaderboard-cifar100-settings.json --algorithm dpsgd-kreg-log2 --evaluation-only --replace-evaluations
```

This mode requires the original `resolved.json` and `selection.json`, verifies
that the tuning declaration and every numerical setting except duration are
unchanged, and uses the ordinary campaign lock. It replaces only the four
held-out evaluation directories and root score copies; it never reruns tuning.
The sections below describe the original one-hour tuning and pilot setup.

Serve `experiments/leaderboard-workloads.json` to expose the **Dataset / model**
selector. It switches between the existing CIFAR-10/GNLeNet comparison and the
new CIFAR-100/ResNet-18-GN D-PSGD campaign. Settings, algorithm details, JSON/CSV
exports, rankings and deep links retain the selected workload. Each workload
has its own trusted catalog and results; different datasets are never combined
in one ranking. The original single-workload catalogs remain supported.

```sh
python scripts/serve_leaderboard.py --catalog experiments/leaderboard-workloads.json --port 8768
python scripts/run_methodology.py --catalog experiments/leaderboard-cifar100-settings.json --algorithm dpsgd-kreg-log2 --plan
python scripts/run_methodology.py --catalog experiments/leaderboard-cifar100-settings.json --algorithm dpsgd-kreg-log2
```

The CIFAR-100 preset uses the 100 fine labels and a fixed stratified 10% training
holdout before client partitioning: 45,000 train / 5,000 validation / 10,000 test.
The downloaded source is `uoft-cs/cifar100`, saved under `data/cifar100` with source
metadata. RGB channel mean/std are declared in `dasklearn/datasets/cifar100.py`.
Images stay at 32×32 with no augmentation. `resnet18_gn` uses a 3×3 stride-1 stem,
no initial max pool, 32-group GroupNorm throughout, 100 output logits and
11,220,132 parameters. It is randomly initialized without pretrained weights.
No running-statistic buffers are present, so the model also satisfies DeMo's
existing model constraint; only D-PSGD is launched for this workload.

Single-worker warm training calibration on an RTX 5000 Ada measured **5.514 ms
per minibatch of 32**, rounded up to **6 ms** in the simulated compute model.
The calibration artifact is
`experiments/methodology/calibration/cifar100-resnet18-gn.json`. The new workload
declares **100 Mbit/s = 12,500,000 bytes/s** per physical endpoint and retains
10 ms delivery latency. This is 12.5× the old workload's bandwidth. It is a
declared network scenario, not a measured deployment connection. The original
CIFAR-10 capacities and results remain unchanged.

The validation-only duration pilot in
`experiments/methodology/cifar100-resnet18gn-h10-duration-pilot.json` fixes
**H=10 and LR=0.03**, seed 100, and runs through 3,600 simulated seconds before
the full campaign starts. This fixed-H pilot reached **6.04%, 10.62% and 15.40%**
validation accuracy at 900, 1,800 and 3,600 seconds; its curve and provenance are
saved in `experiments/methodology/calibration/cifar100-resnet18-gn-duration.json`.
An initial H=20 pilot through 1,800 seconds is retained
separately at `cifar100-resnet18gn-horizon-pilot.json`. Neither uses test data or
contributes scores to TPE. In that initial pilot, validation accuracy was still
improving at 900 seconds (10.12% at LR=0.03) and reached 14.76% at 1,800 seconds,
motivating a **3,600-second initial comparison horizon**, rather than
automatically copying 900 seconds. The
benchmark reports 600 / 1,800 / 3,600 seconds with full validation/test checkpoints
every 300 seconds; it does not claim that 3,600 seconds reaches convergence.

D-PSGD keeps 32 clients, fixed seeded 5-regular graphs, non-IID alpha 0.1 /
homogeneous validation tuning, H ∈ {1,5,10,15,20}, log-uniform LR ∈ [0.001,1],
40 TPE configurations (12 startup), three paired tuning seeds 100–102, zero
momentum/weight decay and batch size 32. Every seed executes the full declared
budget. All selected parameters and one shared validation-selected checkpoint
are frozen for seeds 110–112 in all four IID/non-IID and hardware settings.
Two configurations per batch use six worker slots, three per GPU. Resident
scalar training is enabled; multi-model training batching is disabled for this
larger model. Tuning allocates 432,000 simulated seconds across 120 seed runs;
the 12 evaluations allocate another 43,200 simulated seconds.

The campaign declaration is
`experiments/methodology/dpsgd-cifar100-resnet18gn-tpe-dirichlet01-h20.json`;
results are isolated under
`output/leaderboard-v3/cifar100-resnet18-gn/dpsgd-kreg-log2-tpe-dirichlet01-h20/`.
Pilot/calibration data are in separate sibling artifacts. No previous winners
or scores are reused, and all comparisons within the new workload share the
same model, capacities and budget.

The active catalog, `experiments/leaderboard-settings.json`, includes
**D-PSGD, k = log2(n)**, **Shatter, 16 virtual nodes per real node,
r = log2(n)**, **Epidemic Learning (EL-Oracle)** and **DeMo**.
DeMo is queued after the full Epidemic Learning campaign and jointly tunes
its three compression parameters alongside LR and gradient accumulation H.
The Shatter degree uses the 32 real nodes: r = 5 on a graph
of 512 virtual nodes. Previous results and their directories are preserved;
`experiments/leaderboard-legacy-settings.json` serves the previous dashboard,
including its centralized reference. No previous tuning scores or test results
are imported into this campaign.

Click the algorithm name to open `/algorithm/dpsgd-kreg-log2`. The page includes
a plain-language description, peer-reviewed paper link, authors' code availability,
the local simulator source, experiment parameters, fixed parameter rationale,
the joint tuning table and all per-seed trials. Only verified authors' repositories
may be labeled as authors' code. The official NeurIPS supplementary archive is
linked separately; its code contents have not been verified.

The executable declaration is `experiments/methodology/dpsgd-kreg-tpe-dirichlet01-h20.json` and the
page metadata is `experiments/leaderboard-algorithms.json`. The campaign uses:

| Choice | Declaration |
| --- | --- |
| Dataset / model | CIFAR-10 / GNLeNet; 45,000 training, 5,000 validation, 10,000 test images |
| Clients / graph | 32 clients; fixed seeded random 5-regular graph, k = log2(32) |
| Update rule | Train H local minibatches, then average the updated self/neighbor models |
| Mixing / synchronization | Uniform weights 1/6; wait for same-round neighbors, no global barrier |
| Joint search | Optuna TPE; H in {1, 5, 10, 15, 20}; LR sampled logarithmically in [.001, 1] |
| Search policy | 40 configurations; 12 startup configurations; sampler seed 20261104; batches of 3 configurations; no pruning |
| Fixed optimizer settings | Batch size 32 per client, momentum 0, weight decay 0, constant LR |
| Tuning setting | Non-IID training data, Dirichlet alpha 0.1; homogeneous hardware |
| Tuning seeds | 100, 101, 102; paired across all candidates |
| Per-trial budget | 900 simulated seconds; full validation split every 60 seconds |
| Selection | Highest validation accuracy averaged across seeds at one shared checkpoint in 60–900 s; earliest checkpoint breaks within-candidate ties; lower LR, then variant name and candidate ID, break between-candidate ties |
| Search size | 40 configurations, 120 seed runs; 108,000 total allocated simulated seconds |
| Evaluation | Freeze the winning pair AND checkpoint time for all four data/hardware settings; fresh seeds 110, 111, 112 |
| Evaluation readout | Offline uniform parameter average of all clients; full test split, checkpoints every 60 seconds |
| Capacities | Mean 20 ms per SGD step, 1,000,000 bytes/s per client, 10 ms latency per transfer |

The runner checks connectivity for every declared graph seed and records the
actual edges and spectral gaps in `output/leaderboard-v2/dpsgd-kreg-log2-tpe-dirichlet01-h20-launch/plan.json`.
One graph is reused for every candidate at a given seed. Every trial must finish
its budget before selection. Failed/incomplete searches block evaluation; non-finite
endpoint candidates are retained as divergent and are ineligible. A variant with
no finite candidate remains in the record; a finite joint candidate from another variant
can still win. LR winners exactly on a range bound are visibly flagged without silently extending
the range. This is a finite declared search, not a claim of a global optimum.

```sh
# Resolve the search and graph checks without training.
python scripts/run_methodology.py --plan
# Launch the complete search, then frozen-pair evaluation across all four settings.
python scripts/run_methodology.py
# Serve the active dashboard.
python scripts/serve_leaderboard.py --catalog experiments/leaderboard-settings.json --port 8766
# Inspect previous results separately.
python scripts/serve_leaderboard.py --catalog experiments/leaderboard-legacy-settings.json --port 8767
```

The fresh search uses eight worker slots, four on each GPU. Preliminary tuning
results were deleted at the user's request before this fresh run. The selection
metric is `best_mean_accuracy`: maximize the mean across seeds at each shared
checkpoint, then maximize across checkpoints and candidate pairs. This differs
from averaging each seed's independent maximum. Initialization is displayed as a
reference and is excluded from trained-checkpoint selection. All candidate seeds
must have the complete declared checkpoint schedule. Divergent candidates remain
ineligible, even if an earlier checkpoint was finite.

The original 21-pair grid declaration remains at
`experiments/methodology/dpsgd-kreg.json`, and its output under
`output/leaderboard-v2/dpsgd-kreg-log2-anytime/` is preserved. TPE writes to the
fresh `output/leaderboard-v2/dpsgd-kreg-log2-tpe-dirichlet01-h20/` directory and imports no prior
scores. Changing the catalog does not launch training. Install the updated
requirements before running the campaign.

The completed IID TPE campaign is archived through
`experiments/archive/iid-tpe-h10-20261104/leaderboard-settings.json`; its
artifacts remain in `output/leaderboard-v2/dpsgd-kreg-log2-tpe/`. The active
catalog exposes only the new non-IID campaign and defaults to
`dirichlet01-uniform`. Learning-rate range [.001, 1], three paired tuning
seeds, 900-second budgets, batch size 32, zero momentum and zero weight
decay are unchanged. No prior validation observations are imported.

### Adaptive search configuration and progress

New launches through `scripts/run_methodology.py` cap concurrency at **three
workers per GPU** by default (six workers across the two configured GPUs).
Existing campaigns keep their original allocation. The cap preserves smaller
configured worker counts and CPU slots; `--workers-per-gpu` explicitly overrides
it. The resolved worker list is recorded in the launch plan. Search batches,
paired seeds, simulated budgets and validation selection rules are unchanged.

Set `tuning.method: tpe` and replace `learning_rates` with `search_spaces`, keyed
by the named runs. For example:

```yaml
tuning:
  method: tpe
  budget: 900
  seeds: [100, 101, 102]
  evaluation_period: 60
  evaluation_examples: 0
  metric: best_mean_accuracy
  select_variant: true
  n_trials: 40
  n_startup_trials: 12
  sampler_seed: 20261104
  batch_size: 3
  search_spaces:
    dpsgd-kreg-log2:
      learning_rate: {type: float, low: 0.001, high: 1.0, log: true}
      local_steps: {type: categorical, choices: [1, 5, 10, 15, 20]}
```

Distributions support `float` and `int` bounds, optional `step` or logarithmic
sampling, and categorical choices. Only algorithm parameters can vary; dataset,
seed, duration, evaluation split, and other scenario settings stay fixed.
Different named algorithms can declare different spaces. With
`select_variant: false`, each named run gets its own study and the same declared
configuration/seed/time allowance. With `select_variant: true`, variants of one
algorithm share a study, and TPE also samples the variant. Algorithm-specific
spaces are conditional on that variant.

`batch_size` is configurations **per study**, each expanded over every tuning
seed. Batches across studies share `benchmark.execution_devices`, so several
algorithms and seed evaluations can execute concurrently without exceeding the
declared worker slots. Repeated devices allow multiple workers on one GPU. There
is a barrier between batches. The seeded sampler receives observations in fixed
candidate order after every seed in the batch has finished; completion timing
cannot change the suggestion order. A batch that crosses the startup threshold
may contain extra random suggestions; the default 12/3 policy uses exactly 12.

Every configuration runs the full duration, including divergent configurations
that are eligible for the runner's existing full-budget divergence handling.
Incomplete or failed seed groups block held-out evaluation. Full-budget
divergence is recorded as an ineligible candidate and a worst-direction Optuna
observation. There are no pruning calls or shortened training budgets.

The algorithm detail page refreshes every two seconds and shows startup/adaptive
sampling, suggested/completed configurations, running/completed seed evaluations,
completed simulated-time allocations, the best provisional parameters and shared
checkpoint, and a score-history chart. Dots are completed configuration scores;
the line is the best score so far. Per-configuration tables expose all sampled
parameters and per-seed results. No observations are synthesized while a search
is planned.

`tuning/search.json` records the sampler version and seed, policy, candidate
parameters, sampling phases, paired observations and selected checkpoint times.
`tuning/leaderboard.json` aggregates every proposed candidate; individual worker
artifacts remain under `tuning/batch-*/runs/`. The final `selection.json` freezes
every sampled parameter and checkpoint for fresh-seed evaluation in every scenario.
New searches require fresh output directories. To recover a failed or interrupted
TPE tuning phase, use `scripts/run_methodology.py --algorithm ALGORITHM --resume-tuning`.
Recovery requires the original benchmark declaration and Optuna version. It
snapshots the prior control files, replays and verifies saved proposals and scores
in their original batch order, retains completed seed runs, and retries incomplete
seeds in fresh recovery directories. Original batch artifacts remain unchanged.
Selection and held-out evaluation still wait for the full declared search.
Grid reuse is separate from TPE recovery. Reproducibility requires the same Optuna version,
search declaration, batch size, and numerical results. TPE may propose a
configuration more than once; each proposal counts against the fixed allowance.

Nonfinite intermediate validation metrics do not stop a tuning worker. It executes
all training tasks and every scheduled checkpoint through the full simulated-time
budget, then the seed is marked divergent and the configuration is ineligible.
The first divergence time and separately verified completed budget are recorded.
Worker crashes, incomplete schedules, and held-out evaluation failures remain errors.

The generic tuning pipeline's optional `tuning.select_variant: true` selects one
variant/LR pair across variants of a single algorithm. It preserves each variant's
candidate table, but sends only the winning variant to held-out evaluation. The
campaign runner then carries exactly that pair to the remaining scenarios.
A different algorithm may not be included in the same joint-selection group.

### Shatter: 16 virtual nodes, r = log2(real nodes)

The declaration is
`experiments/methodology/shatter-k16-rlog2-tpe-dirichlet01-h20.json`; its details
page is `/algorithm/shatter-k16-rlog2`. Run its independent campaign with:

```sh
python scripts/run_methodology.py --algorithm shatter-k16-rlog2 --plan
python scripts/run_methodology.py --algorithm shatter-k16-rlog2
```

Each of 32 real nodes hosts 16 virtual endpoints, giving 512 graph vertices.
The virtual graph is 5-regular and redrawn each round with seed `seed + round - 1`.
The launch plan records and checks the initial graph for every tuning/evaluation
seed; subsequent rounds follow that declared seed rule. Virtual endpoints route
contiguous model chunks and share their host's physical bandwidth. Each chunk
index is averaged over received copies plus the local copy. Chunk counts can
differ, so these are not fixed 1/6 mixing weights.

Shatter uses the same non-IID Dirichlet alpha 0.1 scenario, physical capacities,
H choices, LR range, 40 configurations, 12 startup configurations, paired tuning
seeds and full 900-second budgets as the D-PSGD campaign. Two configurations per
TPE batch give six seed runs, matching three workers per GPU. The winning
parameters and one shared checkpoint are frozen for evaluation seeds 110–112
in all four scenarios. Results go to the separate
`output/leaderboard-v2/shatter-k16-rlog2-tpe-dirichlet01-h20/` directory.

The dashboard combines registered algorithms' held-out runs at read time.
Completed D-PSGD measurements remain visible while Shatter is pending or tuning;
unmeasured Shatter rows stay blank and official ranks wait for every paired seed.
Scenario compatibility is checked, and JSON/CSV views include both algorithms.
No underlying D-PSGD experiment files are rewritten. Each algorithm's details
page continues to show its independent TPE history and declaration.

This is the existing real-node chunked-exchange specialization; virtual nodes
are not separate training clients. Its graph and chunk execution have protocol
and integration tests. Identity-hiding transport and privacy-attack evaluation
are outside this simulator, and the leaderboard makes no privacy-guarantee or
complete paper-reproduction claim. The publisher's official artifact link
identifies the authors' repository separately from the local implementation.

### Epidemic Learning: a new regular graph each round

`epidemic-kreg-log2` uses the existing **EL-Oracle** implementation: a fresh
random 5-regular graph over the 32 real nodes each round, full-model exchanges,
and equal averaging of self plus five neighbors. Graph seeds follow
`seed + round - 1`, matching Shatter's topology schedule with real nodes as
vertices. This is the coordinated regular-graph variant from the
[Epidemic Learning paper](https://papers.neurips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html).
Its declaration is `experiments/methodology/epidemic-kreg-log2-tpe-dirichlet01-h20.json`.
It uses the same H/LR ranges, 40 configurations, 12 startup configurations,
three paired seeds, full 900-second budgets, validation objective, physical
hardware, and four frozen-parameter evaluation settings as Shatter.

To queue it after the entire detached Shatter campaign:

```sh
python scripts/queue_methodology.py --after shatter-k16-rlog2 --algorithm epidemic-kreg-log2
```

Run the queue itself as a detached background process to survive terminal exit.
The queue checks the predecessor's PID and creation time, waits for its exit,
and requires successful tuning and all four evaluations before launching the
successor with three workers per GPU. It fails visibly if the predecessor exits
incompletely. A per-campaign lock prevents duplicate queue launches, and existing
experiment output is never overwritten. Queue state and monitor logs live in
the successor's sibling `-launch` directory; its campaign output is created only
when execution begins. `/algorithm/epidemic-kreg-log2` shows the waiting state,
then the ordinary live TPE progress. The leaderboard includes pending rows
without adding invented scores.

### DeMo: jointly tuned compression and gradient accumulation

DeMo is queued after **all Epidemic Learning tuning and four evaluations**
succeed. Its declaration is `experiments/methodology/demo-tpe-dirichlet01-h20.json`
and its detail page is `/algorithm/demo`. The five-dimensional joint TPE search is:

| Parameter | Declared search |
| --- | --- |
| Learning rate | Log-uniform [0.00001, 0.1] |
| Gradient accumulation H | {1, 5, 10, 15, 20} minibatches |
| Compression residual decay | {0, 0.9, 0.99, 0.999, 0.9999} |
| Maximum DCT block dimension | {16, 32, 64, 128} |
| Transmitted coefficients per block | {1, 2, 4, 8, 16, 32} |

H averages gradients at fixed model parameters before one DeMo update; it does
not perform H local SGD updates. Top-k is an absolute coefficient count clipped
to the actual block size. Block dimensions divide each tensor dimension. Residual
decay controls untransmitted information and is separate from SGD momentum.
The LR range is specific to DeMo's sign updates and includes the earlier 0.001
baseline. No earlier scores or winners are imported.

The existing reference-style implementation uses a fixed directed ring all-gather
with 31 stages for 32 clients. Every packet is forwarded to every client, with
indices, values, metadata, endpoint bandwidth and per-transfer latency charged.
The launch plan records the actual ring; it does not describe a random 5-regular
graph. DCT/compression/reconstruction have zero modeled compute cost; wall time
is recorded separately. Stable membership is required.

DeMo receives the same 40 configurations, 12 startup configurations, validation
objective, tuning scenario, seeds 100–102 and full 900-second budgets as the other
campaigns: 120 tuning seed runs and 108,000 allocated simulated seconds. Two
configurations per batch use six workers, three per GPU. All five selected
parameters and the shared checkpoint are frozen for 12 test runs: seeds 110–112
across IID/non-IID and homogeneous/heterogeneous hardware. These add 10,800
allocated simulated seconds. The finite 40-configuration search is not exhaustive
over the five-dimensional space. Output uses the fresh
`output/leaderboard-v2/demo-tpe-dirichlet01-h20/` directory.

```sh
python scripts/run_methodology.py --algorithm demo --plan
python scripts/queue_methodology.py --after epidemic-kreg-log2 --algorithm demo
```

The queue runs detached to survive terminal exit. Its state, monitor log and
eventual campaign log live in the sibling `demo-tpe-dirichlet01-h20-launch/`
directory. Pending dashboard rows have no invented measurements.

### Conflux V2: jointly tuned sample and reducer layout

The original Conflux V2 averaging protocol is registered only for CIFAR-10 /
GNLeNet, at `/algorithm/conflux-v2`. It uses the same 32 clients, 45,000 / 5,000 /
10,000 train/validation/test split, batch size 32, zero momentum and weight decay,
20 ms modeled SGD steps, 1,000,000 bytes/s endpoints and 10 ms transfer latency as
the other CIFAR-10 campaigns. The separate FedOpt optimizer experiments are not
included.

The declaration `experiments/methodology/conflux-v2-tpe-dirichlet01-h20.json`
jointly searches five parameters:

| Parameter | Search space |
| --- | --- |
| Learning rate | Log-uniform [0.001, 1] |
| Local SGD steps H | {1, 5, 10, 15, 20} |
| Active training clients | {2, 4, 8, 16, 32} |
| Model chunks | {1, 2, 4, 8, 16, 32} |
| Reducer cap | {1, 2, 4, 8, 16, 32} |

Every sample contributes fully (`success_fraction=1`) and uses immediate uploads.
Placement assigns chunks greedily relative to public bandwidth and existing
assigned chunk load, among the highest-bandwidth peers up to the reducer cap.
Fewer owners can be used when the chunk count is below that cap. Trainers and
reducers share endpoint bandwidth. The planner verifies declared homogeneous
placements and seeded training samples rather than inventing a fixed graph.

Tuning uses Dirichlet alpha 0.1 / homogeneous hardware, 40 TPE configurations,
12 startup configurations, sampler seed 20261104, two configurations per batch,
three workers per GPU, seeds 100/101/102 and full 900-second budgets. It selects
the best mean validation accuracy at one shared 60-second checkpoint. All five
parameters and that checkpoint are frozen for seeds 110/111/112 in the four
IID/non-IID and homogeneous/heterogeneous evaluation settings. This is 120 tuning
seed runs and 12 test seed runs. Results are isolated under
`output/leaderboard-v2/conflux-v2-tpe-dirichlet01-h20/`.

The offline readout averages every physical client's latest published checkpoint,
including inactive clients' retained models; collection and evaluation costs are
excluded. It differs from a readout of the latest sample aggregate. Stable
membership, public static capacities and instantaneous chunk operations are
limitations. Forty configurations are a finite five-parameter search, and
transfer performance is not independently tuned per scenario. Earlier CIFAR-10
test curves have been inspected, so this remains an exploratory comparison.

```sh
python scripts/run_methodology.py --algorithm conflux-v2 --plan
python scripts/run_methodology.py --algorithm conflux-v2
```

The completed 9 November 2026 campaign selected **LR=0.051598546094309986,
H=15, two active clients, 32 chunks and a reducer cap of eight**, with the shared
validation checkpoint at 900 seconds. All 120 validation runs and 12 held-out
runs completed successfully, using clean source commit `69437c0` with identical
source hashes across all batches and evaluation settings.

Test accuracy at 900 seconds, mean ± sample standard deviation across seeds
110/111/112, is:

| Data / hardware | Conflux V2 accuracy | Rank among six algorithms |
| --- | ---: | ---: |
| IID / homogeneous | 76.80 ± 0.21% | 2 |
| IID / heterogeneous | 77.55 ± 0.42% | 1 |
| Non-IID / homogeneous | 70.85 ± 1.10% | 1 |
| Non-IID / heterogeneous | 66.66 ± 1.09% | 1 |

The [compact audited result](../experiments/methodology/results/conflux-v2-cifar10-gnlenet-20261109.json)
retains the declaration, all candidate mean validation curves, selected parameters,
per-seed test curves, four comparisons and source/package provenance. Raw run
artifacts and validation logs remain in the campaign output and launch directories.
The CPU regression rerun passed 1,085 tests with 25 CUDA checks skipped; the earlier
GPU run passed those checks, and its sole generic-registry expectation failure was
fixed and verified by the 62-test focused rerun.

### Teleportation: jointly tuned active clients and token topology

The independent CIFAR-10/GNLeNet campaign is declared in
`experiments/methodology/teleportation-tpe-dirichlet01-h20.json` and appears at
`/algorithm/teleportation`. It jointly tunes four parameters:

| Parameter | Declared search |
| --- | --- |
| Learning rate | Log-uniform [0.001, 1] |
| Local SGD steps H | {1, 5, 10, 15, 20} |
| Active clients / model tokens b | {2, 4, 8, 16, 32} |
| Token mixing topology | {ring, hypercube, complete} |

Each round uniformly samples b distinct clients and shuffles token assignment.
Each token performs H SGD updates on its current client's partition, mixes with
same-round trained neighbors, and moves to the next client assigned that token.
Remote gossip and full-model handoffs consume the physical clients' shared
bandwidth and per-transfer latency. Self-handoffs are local. Equal self/neighbor
weights are fixed; there is no separately tunable mixing coefficient.
Power-of-two active counts make all hypercube combinations valid. The search
excludes the single-token random-walk case and includes full participation at
b=32, which still shuffles token ownership. Some small graphs coincide: ring
and hypercube at b=2 and b=4; complete also coincides at b=2.

Use the same 40 full-budget TPE configurations, 12 startup configurations,
sampler seed 20261104, seeds 100/101/102, 900 simulated seconds per seed, and
60-second validation checkpoints as the other campaigns. Tuning uses Dirichlet
alpha 0.1 and homogeneous hardware. Two configurations per batch occupy six
workers, three per GPU. Batch size 32, momentum 0, weight decay 0, compute and
network capacities remain the same. No prior observations are imported.

Freeze all four winning values and the shared validation-selected checkpoint
for seeds 110/111/112 across IID/non-IID × homogeneous/heterogeneous hardware:
120 tuning seed runs followed automatically by 12 test runs. Output goes to
`output/leaderboard-v2/teleportation-tpe-dirichlet01-h20/`. The launch plan checks
all 15 declared token graphs, records their actual edges and weights, and records
the first two token assignments for each tuning/evaluation seed. The details
page shows active count and topology in the candidate table and saved winner.

```sh
python scripts/run_methodology.py --algorithm teleportation --plan
python scripts/run_methodology.py --algorithm teleportation
```

This uses the existing Algorithm 1 specialization with a local-SGD extension;
hypercube is not the paper's Base-2 graph. Joint full-budget TPE does not
reproduce the paper's active-node search procedure. Forty configurations are a
finite four-parameter search. Stable membership and oracle token assignment are
required. For compatibility with the common leaderboard, the offline readout
averages all 32 physical clients' latest published models, including retained
inactive models, rather than the paper's active-token average. This convention
is used for both validation selection and test evaluation.

These results measure transfer of parameters selected on one scenario, rather
than each scenario's individually tuned optimum. The simulator's update order
is a declared train-then-average variant and differs from Algorithm 1 in the
original D-PSGD paper; see [the numerical contract](baseline-validation.md#d-psgd-preserve-train-then-average).
The network abstraction uses greedy queued endpoint allocation and an additive
one-way delivery delay per message. It is not a calibrated TCP simulation;
contention, ordering sensitivity and calibration require a separate validation
study before deployment-performance claims.
The shared trainer resets optimizer state per local block; momentum stays zero
in this campaign. Earlier campaigns already inspected the CIFAR-10 test set, so
new random seeds do not make that dataset an untouched final test set.

The sections below describe earlier campaigns and the underlying benchmark tools.


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
The catalog includes all four combinations of IID/non-IID data and
homogeneous/heterogeneous hardware. Non-IID uses Dirichlet alpha 0.1.
The non-IID/heterogeneous setting is the default.
The budget picker defaults to **900 seconds** when available (otherwise the
largest available budget), and preserves a user's choice during live refreshes.
Setting links can be bookmarked, e.g. `?setting=dirichlet01-heterogeneous`.
Accuracy and evaluation-loss curves are shown against simulated time and sent
traffic, averaging available seeds at each checkpoint. Loss has a separate
vertical scale, with lower values better; the table also shows loss mean and
standard deviation at the selected budget. For the CIFAR-10/GNLeNet global-model
runs this is test-set negative log-likelihood per example, evaluated on the
averaged client parameters. It is already recorded in existing checkpoint
exports, so displaying these curves does not require rerunning experiments.
To add other settings, add their result directories to this catalog and restart
the server; no data is pooled across settings. A catalog may include a setting
before its output exists.

Use the Python environment containing the simulator requirements. No additional
dashboard framework, database, external service, or JavaScript build is needed.

## Step-budgeted centralized reference

`experiments/leaderboard-cifar10-centralized.yaml` defines a separate pooled-data
reference for the CIFAR-10/GNLeNet leaderboard. One node has the entire
**45,000-image training split**; the 5,000 validation and 10,000 test images stay
held out. It runs direct PyTorch SGD without the simulator: there are no modeled
compute times, heterogeneous capacities, communication or aggregation. It uses
the same preprocessing, model, batch size 32, zero momentum/decay and one
minibatch update per step. A persistent shuffled loader covers the complete
training partition and reshuffles at epoch boundaries.

```sh
python scripts/run_centralized_baseline.py experiments/leaderboard-cifar10-centralized.yaml --plan
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_centralized_baseline.py \
  experiments/leaderboard-cifar10-centralized.yaml --devices cuda:0 cuda:1 cuda:0 cuda:1
```

LR selection comes first. Five candidates `[.003, .01, .03, .1, .3]` each get
**5,000 SGD steps**, tuning seed 0 and the same balanced 1,000-image validation
subset. The highest endpoint validation accuracy wins; lower LR breaks an exact
tie. All candidates must finish their allocated steps; non-finite endpoints are
ineligible. These are equal **step** budgets within this reference search, not
the earlier collaborative algorithms' equal simulated-time tuning budgets.

The selected LR is frozen for three fresh seeds (10/11/12). Each trains for
**50,000 minibatch steps**, approximately 35.6 epochs, with full validation-set
evaluation at initialization, every 1,000 steps and the exact final step. These
counts are configurable. Each seed saves its **minimum-validation-loss model**;
the earlier step breaks an exact loss tie. Its selected checkpoint
is evaluated **once on the full test set after training finishes**. No test
curve or best-test checkpoint is used for LR or checkpoint selection.

The catalog's optional `reference: {output: ../output/leaderboard-cifar10-centralized-minloss}`
attaches this result to compatible held-out global-model settings at serve time.
It does not rewrite existing experiments, CSVs or rankings. All three final seeds
must complete before lines appear. Tuning views have no test reference. The
dashboard displays the completed reference on the plots. A labelled **dashed
horizontal line** appears on each of the four accuracy/loss plots, with a dashed
legend key.

The same pooled-data reference is shown for IID/non-IID and homogeneous/
heterogeneous settings: partitioning and client hardware profiles do not affect
this single-node training. The reference stays fixed when the simulated budget
picker changes and is labelled with its own SGD-step budget. It is an empirical
reference, not a mathematical upper bound or an equal-compute/time comparison.
Its best checkpoint means lowest loss on the declared validation schedule within
the step limit, not a guarantee of convergence or maximum attainable test accuracy.

The first reference selected checkpoints by validation accuracy. It is archived
unchanged under `output/leaderboard-cifar10-centralized-steps/`. To reuse that
completed, unchanged LR search while regenerating the minimum-loss checkpoints:

```sh
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_centralized_baseline.py \
  experiments/leaderboard-cifar10-centralized.yaml \
  --reuse-tuning output/leaderboard-cifar10-centralized-steps
```

Reuse verifies the search specification, data/model/optimizer settings, seeds,
raw trial settings, validation curves and original LR selection before starting
final workers. LR selection still maximizes validation accuracy at 5,000 steps;
the separate final-checkpoint criterion minimizes validation loss. The rerun
uses the same LR and seeds, scans the full 50,000-step trajectories, and reads
the test set once per newly selected checkpoint. Prior test readouts have already
been inspected; this revised reference is exploratory rather than a new untouched
test-set claim.

Artifacts under the fresh output directory include source/resolved settings,
`selection.json`, the live `reference.json`, per-phase manifests, per-worker
`metrics.json`, validation curves, logs, and `best-model.pt` for each final seed.
The checkpoint contains CPU model state and the selected validation metrics.
Source/package provenance and actual execution wall time are recorded; there
are no simulated-time budgets applied. Existing output directories are refused.

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
| Best accuracy | Highest checkpoint accuracy at or before the budget, selected independently per seed, then averaged across seeds; primary sort key. Each checkpoint uses mean client accuracy (`local`) or single averaged-model accuracy (`global`). Earliest checkpoint breaks exact accuracy ties, including initialization |
| Accuracy SD | Sample standard deviation across seeds; blank for one seed |
| Corresponding loss | Loss at each seed's highest-accuracy checkpoint, then mean and sample standard deviation across seeds; not the minimum loss or loss at a shared checkpoint |
| Average accuracy | Area under the checkpoint curve from zero to the budget, divided by the budget; left-constant integration, in percentage points |
| Worst 10% accuracy | Local readout only: mean of the lowest `ceil(clients / 10)` client accuracies at each seed's best-accuracy checkpoint, then mean across seeds; omitted for global readout |
| Budget traffic | Total sent bytes at the budget endpoint, counted once, including bytes sent by unfinished transfers |
| Best observed accuracy | JSON/CSV compatibility field, equal to the primary best-accuracy score |
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

A seed contributes only once its complete curve reaches the exact selected
budget, even if an earlier checkpoint already has a high score. Existing runs
served by the dashboard are rescored from their recorded curves without
rerunning training or rewriting result files. JSON declares
`checkpoint_policy: best_accuracy_within_budget`; CSV exports include
`loss_mean` and `loss_sd` paired with `accuracy_mean` and `accuracy_sd`.
Registry evaluation metadata also reports the best-accuracy policy and primary
budget; legacy `fixed_endpoint` declarations remain readable. Validation tuning
keeps its declared objective, including shared-checkpoint selection where configured.

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

## Remaining data/hardware combinations

These configurations complete the four-setting matrix:

| Data | Hardware | Configuration |
| --- | --- | --- |
| IID | Heterogeneous | `experiments/leaderboard-cifar10-32-iid-heterogeneous.yaml` |
| Dirichlet alpha 0.1 | Homogeneous | `experiments/leaderboard-cifar10-32-dirichlet01-uniform.yaml` |

Each uses the same five algorithms, frozen LRs, seeds 10/11/12, CIFAR-10/GNLeNet,
32 clients, 900-second horizon, and global-model evaluation every 60 seconds.
Each configuration has two worker slots, one per GPU, so both experiments can
run concurrently with four workers total. Worker scheduling affects execution
wall time; the simulated protocol capacities and budgets match the other settings.

```sh
# Run in separate terminals, or launch both as detached background processes.
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_benchmark.py experiments/leaderboard-cifar10-32-iid-heterogeneous.yaml
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_benchmark.py experiments/leaderboard-cifar10-32-dirichlet01-uniform.yaml
```

Both appear in the same setting picker on port 8766. No LR search is repeated.

## Five-algorithm extension across all four settings

`experiments/leaderboard-cifar10-32-additions-sgp.yaml` adds SGP, GT-DSGD, MoTEF,
RelaySGD and Decentralized AMSGrad to the completed five-algorithm comparisons:

```sh
python scripts/extend_leaderboard.py experiments/leaderboard-cifar10-32-additions-sgp.yaml \
  --baseline-catalog experiments/leaderboard-base-settings.json --plan
CUDA_VISIBLE_DEVICES=0,1 python scripts/extend_leaderboard.py experiments/leaderboard-cifar10-32-additions-sgp.yaml \
  --baseline-catalog experiments/leaderboard-base-settings.json \
  --reuse-tuning output/leaderboard-cifar10-32-extended/tuning
```

The search uses IID data and homogeneous capacities, five declared LR candidates
per algorithm, seed 0, 300 simulated seconds per candidate, and 1,000 validation
images. Thus every addition receives five trials and 1,500 simulated tuning
seconds. The same global-parameter-average readout used in final evaluation
selects the highest endpoint validation accuracy; the lower LR breaks exact
ties. All candidates must finish before selection. There is no early stopping,
extra refinement, or test-based LR selection.

Selected LRs are frozen across all four data/hardware combinations. Each setting
gets three new test seeds (10/11/12) per algorithm, giving 60 final runs with
60/300/900-second views and full test-set evaluation every 60 seconds. Settings
execute sequentially, each with four worker slots shared across both GPUs.
The original SWIFT search was stopped because execution was slow: its 300-second
trials schedule 480,000 client updates, and the first was still running after
18 minutes. SWIFT is deferred and replaced with SGP. See the
[SWIFT execution-cost note](swift.md#cost-of-executing-fixed-time-benchmarks).

RelaySGD uses a balanced tree with count normalization; SGP uses the one-peer
rotating directed exponential graph. The other additions use rings.
MoTEF uses magnitude Top-k at 10%, consensus step .1 and fresh-gradient
momentum weight .1. AMSGrad fixes beta1=.9, beta2=.999 and epsilon=1e-6; its LR
grid is smaller than the SGD grids. These non-LR settings are held fixed during
tuning and all final runs.

The extension preserves original result directories. It writes combined
ten-algorithm reports under `output/leaderboard-cifar10-32-extended-sgp/settings/`,
with the original 15 runs plus 15 additions per setting. New LRs display as
pending until selection; rankings remain provisional until all ten algorithms
have paired checkpoints. A separate **LR tuning · five additions** catalog entry
shows validation trials. `selection.json` records candidate scores, the selected
LRs and selection rule; `pipeline.json` records progress and failures. Failure or
interruption during tuning prevents final evaluation from starting.

The rescheduled search reuses 12 completed non-SWIFT trials without running them
again. `--reuse-tuning` accepts only completed trials whose full options (apart
from execution device), network, seeds, checkpoint schedule and model readout
match. Raw resolved settings and observation CSVs must also agree with the
reported curves. Reused trials retain their original wall time and provenance;
only missing candidates execute, with no extra tuning budget. A fresh search
can omit this flag. A resumed search needs a fresh output directory.

The old `leaderboard-cifar10-32-additions.yaml` configuration and all old outputs
remain intact on disk under `output/leaderboard-cifar10-32-extended/`. The old
search is excluded from the leaderboard menu, active search and final rankings.

The original five retain their existing results and LRs, selected by the earlier
local-client validation search. The new search uses the current global-model
readout; equal tuning budget here applies to the five additions. Final model
readout, data, capacities, budgets and paired seeds match within each setting.

## DeTAG, Teleportation and Moshpit extension

This historical configuration added three algorithms to the completed
ten-algorithm baselines. Its tuning outcomes are now carried into the combined
seven-algorithm batch below; its original artifacts remain intact:

```sh
CUDA_VISIBLE_DEVICES=0,1 python scripts/extend_leaderboard.py \
  experiments/leaderboard-cifar10-32-detag-teleportation-moshpit.yaml \
  --baseline-catalog experiments/leaderboard-ten-settings.json
```

Each receives the same five LRs `[.003, .01, .03, .1, .3]`, seed 0,
300 simulated seconds per trial, and 1,000 validation images. Selection uses
endpoint accuracy of the equal-weight global parameter average. Thus there
are 15 tuning trials and 1,500 simulated tuning seconds per algorithm. The
selected LRs are frozen across seeds 10/11/12 and all four settings: 36 new
900-second test runs, evaluated on the full test set every 60 seconds.

`local_steps` remains **1** throughout. Other method parameters are fixed:
DeTAG uses a ring, phase length 2 and spectral acceleration; the phase averages
gradients at fixed parameters, rather than performing two successive local SGD
steps. Teleportation uses eight active clients and ring mixing. Its global
evaluation follows the same rule as all other algorithms: average the latest
published models of all 32 physical clients, including currently idle clients.
Moshpit uses groups of up to eight, one averaging step, and bandwidth-based
load balancing, with stable membership.

The configuration has eight concurrent workers, four per GPU. The original
batch stopped during tuning: DeTAG's .03, .1 and .3 trials produced non-finite
validation loss at the full 300-second budget. Twelve other trials completed;
no final test runs started. Both successful and divergent outcomes are reused
by the combined batch, without giving any algorithm additional search trials.

## Seven-algorithm combined batch

This historical batch was interrupted after tuning finished for all methods
except AD-PSGD. AD-PSGD and DeTAG were then deferred at the user's request.
The active five-peer-algorithm batch below reuses its completed trials.

`experiments/leaderboard-cifar10-32-seven-additions.yaml` batches DeTAG,
Teleportation, Moshpit, AD-PSGD, Epidemic Learning, Gossip Learning and Shatter:

```sh
CUDA_VISIBLE_DEVICES=0,1 python scripts/extend_leaderboard.py \
  experiments/leaderboard-cifar10-32-seven-additions.yaml \
  --baseline-catalog experiments/leaderboard-ten-settings.json \
  --reuse-tuning output/leaderboard-cifar10-32-extended13/tuning
```

Each algorithm gets five LR candidates `[.003, .01, .03, .1, .3]`, each with
300 simulated seconds, seed 0 and the same 1,000-image validation subset.
All use global-model evaluation. Of the 35 allocated tuning trials, 15 outcomes
are reused (12 successful and three divergent); only 20 new trials execute.
No extra candidates or automatic refinement are added. LRs are selected by
endpoint validation accuracy, then frozen across all four data/hardware settings.

An LR candidate with non-finite global validation metrics at its exact full
tuning budget is marked `diverged` and cannot win. All earlier checkpoints must
be complete and finite. Its raw observations, allocated budget and wall time
remain recorded. Early divergence, malformed observations, worker errors and
non-finite test metrics still fail the pipeline. If an algorithm has no eligible
candidate, final evaluation does not start.

Final runs use three fresh seeds (10/11/12), 900 simulated seconds, the full
10,000-image test set, and global-model evaluation every 60 seconds. There are
84 new test runs (seven algorithms × three seeds × four settings). Combined
reports under `output/leaderboard-cifar10-32-extended17/settings/` include all
earlier results: **17 algorithms and 51 runs per setting**. The default budget
view is 900 seconds. Select **LR tuning · seven additions** to follow the search;
the completed five-addition tuning view remains available separately.

`local_steps` stays **1** for every algorithm. DeTAG, Teleportation and Moshpit
retain the settings described above. The four further additions fix:

| Algorithm | Fixed protocol settings |
| --- | --- |
| AD-PSGD | 16 active and 16 passive clients, arithmetic parameter averaging |
| Epidemic Learning | Local mode, two randomly selected outgoing neighbors per round, directed receiver averaging |
| Gossip Learning | One-second gossip period, age-weighted aggregation |
| Shatter | Two model chunks and virtual-node degree two |

These are the repository's implementations and specializations. Gossip Learning
retains its legacy validation status; Epidemic Learning is not an exact-paper
reproduction claim. Shatter uses the real-node chunked specialization, without
claiming the complete virtual-node privacy protocol. See the README's algorithm
table and linked implementation contracts for details.

Eight workers share both GPUs, four per GPU. Settings run sequentially after
tuning. AD-PSGD's continuously training passive clients can require more actual
execution time than communication-limited methods; this does not change its
simulated budget. Executor task progress between evaluation checkpoints is
recorded in the JSON exports and the expandable provenance details. The driver
and dashboard can run detached from SSH, and all prior result directories are
preserved.

## Five-peer-algorithm batch with AD-PSGD and DeTAG deferred

This completed batch is the baseline for the D-PSGD k-regular extension below.
Its configuration is
`experiments/leaderboard-cifar10-32-peer-additions.yaml`. It includes
Teleportation, Moshpit, Epidemic Learning, Gossip Learning and Shatter:

```sh
CUDA_VISIBLE_DEVICES=0,1 python scripts/extend_leaderboard.py \
  experiments/leaderboard-cifar10-32-peer-additions.yaml \
  --baseline-catalog experiments/leaderboard-ten-settings.json \
  --reuse-tuning output/leaderboard-cifar10-32-extended17/tuning
```

All 25 matching completed validation trials are reused, with the same five LRs,
300-second budget and fixed options described above. No tuning is repeated or
expanded. The five frozen LRs are applied to seeds 10/11/12 across all four
data/hardware settings, giving **60 new 900-second runs**. Eight workers share
both GPUs, four per GPU. `local_steps` remains 1. Full test-set global-model
evaluation runs every 60 seconds, with the 900-second view selected by default.

These combined fifteen-algorithm reports are under
`output/leaderboard-cifar10-32-extended15/settings/`, containing the earlier 30
runs plus 15 additions per setting: **15 algorithms and 45 runs per setting**.
The **LR tuning · five peer additions** view contains only the five retained
algorithms. The earlier SGP/GT-DSGD/MoTEF/RelaySGD/AMSGrad tuning view remains
available separately. AD-PSGD, DeTAG and SWIFT are absent from the active
leaderboard's settings, curves, search views and rankings.

The sixteen-algorithm extension below
preserves all of these completed results.

AD-PSGD was stopped because its five 300-second tuning trials each scheduled
roughly 497,000 executor tasks. Initial task-throughput extrapolation suggested
about 40 minutes per trial under the shared GPU workload; these are estimates,
not completed-run measurements. Continuous passive-client gradients drive much
of that work. Its implementation and executor cost need investigation before
reintroducing it. See the [AD-PSGD execution-cost note](baseline-validation.md#ad-psgd).

DeTAG was deferred for numerical instability in this configuration: LRs .03,
.1 and .3 gave non-finite validation loss; .003 and .01 gave very large finite
loss and only 10% validation accuracy. None of the tested candidates was useful.
A smaller LR range and an implementation check are future work; these outcomes
do not establish that DeTAG is inherently unstable.

The interrupted seven-algorithm batch, its partial AD-PSGD trials, all DeTAG
outcomes and earlier outputs remain on disk. The replacement batch records the
original trial paths, curves, wall times and provenance when reusing results.

## D-PSGD with log2(n)-regular topology

`experiments/leaderboard-cifar10-32-dpsgd-kreg.yaml` adds
**`dpsgd-kreg-log2`** alongside the existing D-PSGD ring variant. The fixed
network has 32 nodes, so `k = log2(32) = 5`, recorded explicitly as `k: 5`.
The existing D-PSGD implementation generates an undirected random 5-regular
graph from the run seed, then uses both directed edges for communication.
Each node has five neighbors and averages their same-round trained models with
its own model. The seeded graph stays fixed across rounds. The tuning seed 0
and final seeds 10/11/12 have been checked for degree five and connectivity.

```sh
python scripts/extend_leaderboard.py \
  experiments/leaderboard-cifar10-32-dpsgd-kreg.yaml \
  --baseline-catalog experiments/leaderboard-fifteen-settings.json --plan
CUDA_VISIBLE_DEVICES=0,1 python scripts/extend_leaderboard.py \
  experiments/leaderboard-cifar10-32-dpsgd-kreg.yaml \
  --baseline-catalog experiments/leaderboard-fifteen-settings.json
```

The independent LR search uses the same five candidates
`[.003, .01, .03, .1, .3]`, 300 simulated seconds each, seed 0, and 1,000
validation images on IID data and homogeneous hardware. Selection maximizes
the global-model validation accuracy at the exact tuning endpoint. The selected
LR is frozen for all four data/hardware settings and three final seeds, giving
**12 new 900-second test runs**. Local steps remain 1, with full test-set
evaluation of the uniformly averaged global model every 60 seconds. The
configured GPU slots and modeled client capacities match the preceding batch.
Communication with five neighbors is fully charged by the existing scheduler.

`experiments/leaderboard-fifteen-settings.json` captures the four completed
fifteen-algorithm sources. Their 180 runs are preserved, and new combined reports
are written under `output/leaderboard-cifar10-32-dpsgd-kreg/settings/`: **16
algorithms and 48 runs per setting**. The main catalog also includes a separate
**LR tuning · D-PSGD k-regular (k = 5)** view. The centralized minimum-validation-
loss reference and earlier tuning views remain available, and the simulated
budget picker still defaults to 900 seconds. Use a fresh output for each launch;
the extension is suitable for detached background execution.

## DSpodFL extension

`experiments/leaderboard-cifar10-32-dspodfl.yaml` adds **DSpodFL** (`dspodfl`)
to the completed sixteen-variant comparison. The existing implementation and
its source audit are described in [DSpodFL](dspodfl.md). This is an LR-only
screening on CIFAR-10/GNLeNet, not a reproduction of the paper's tuned
experiments.

The non-LR policy is fixed beforehand: a connected degree-4 regular graph,
nominal SGD/link probabilities .5, capacity-based participation probabilities,
and a 20 ms minimum iteration opportunity. The capacity policy uses no data or
accuracy information. This topology and policy differ from the paper's
random-geometric experiments.

The search uses the same five LR candidates `[.003, .01, .03, .1, .3]`, 300
simulated seconds each, seed 0, IID data and homogeneous hardware. Selection
uses endpoint accuracy on the fixed 1,000-image validation subset. The chosen
LR is frozen across all four data/hardware settings and seeds 10/11/12:
**12 new 900-second test runs**. Local steps stay at 1. Evaluation uniformly
averages all 32 clients' latest completed models and tests that global model
on the full test set every 60 seconds. Worker slots alternate between both
GPUs; each final setting has three runs. Simulated capacities are unchanged.

```sh
python scripts/extend_leaderboard.py \
  experiments/leaderboard-cifar10-32-dspodfl.yaml \
  --baseline-catalog experiments/leaderboard-sixteen-settings.json --plan
CUDA_VISIBLE_DEVICES=0,1 python scripts/extend_leaderboard.py \
  experiments/leaderboard-cifar10-32-dspodfl.yaml \
  --baseline-catalog experiments/leaderboard-sixteen-settings.json \
  --reuse-tuning output/leaderboard-cifar10-32-dspodfl-fspda/tuning
```

The detached launch reuses all five completed, raw-verified DSpodFL trials from
an interrupted two-method attempt. Omit `--reuse-tuning` for a fresh search.
FSPDA-SA was removed at the user's request after measured task throughput
identified it as a runtime outlier; its implementation, partial trials and
original configuration remain archived under
`output/leaderboard-cifar10-32-dspodfl-fspda/`. See the backlog below. Neither
its partial tuning results nor pending final runs appear in the active catalog.

`experiments/leaderboard-sixteen-settings.json` captures the four completed
source reports. All 192 previous runs are preserved. New combined reports are
under `output/leaderboard-cifar10-32-dspodfl/settings/`: **17 variants and 51
runs per setting** after completion. The active catalog points to these reports
and adds **LR tuning · DSpodFL** alongside the earlier tuning views. Existing
accuracy/loss curves, the centralized dashed reference lines and the default
900-second budget remain available. Launch into a fresh output directory and
detach the driver to survive SSH disconnection.

## Deferred algorithms: investigation backlog

Updated 4 November 2026. These four algorithms are excluded from the active
leaderboard at the user's request. The observations concern this implementation
and CIFAR-10/GNLeNet configuration, not the algorithms' general merit. This is
an investigation plan; the proposed experiments and optimizations have not run.

| Priority | Algorithm | Observed blocker | First investigation |
| --- | --- | --- | --- |
| 1 | DeTAG | All five searched LRs were unusable: exploding finite loss or non-finite loss | Find the first unstable phase, check the numerical update against a reference, then test a smaller LR range |
| 2 | AD-PSGD | About 497,000 executor tasks per 300-second trial; projected roughly 40 minutes of wall time | Profile passive gradient computation, model copies and separate gradient updates on a small saved workflow |
| 3 | SWIFT | 480,000 client updates per 300-second trial; first trial still running after about 19 minutes | Reuse gradient-execution profiling, then measure packet/queue costs and cache staleness |
| 4 | FSPDA-SA | 162,803 executor tasks per 300-second trial; measured early throughput projects 13–15 minutes per trial | Profile gradient/vector operations and model construction; investigate validated resident execution and measure a short CIFAR pilot before re-entry |

AD-PSGD and SWIFT were interrupted before completing their trials. Their runtime
observations do not establish completed-trial costs or accuracy. The AD-PSGD
497,000 figure counts executor tasks; SWIFT's 480,000 figure counts client
updates. These counts are different units and must not be compared directly.

### FSPDA-SA: executor runtime

The attempted 4 November batch is archived under
`output/leaderboard-cifar10-32-dspodfl-fspda/`. It used the existing `fspda`
implementation with a ring, Bernoulli edge probability .5, random coordinate
retention .1 and fixed gains gamma=.1, eta=.01, beta=.1. Only the primal LR was
searched. The implementation's paper/author-code differences are documented
in [FSPDA-SA](fspda.md).

Each 300-second tuning DAG has 162,803 tasks, versus 11,593 for DSpodFL.
An early 20-second throughput sample measured about 199–212 tasks/second for
four concurrent FSPDA-SA workers sharing one GPU. That projects roughly
13–15 minutes per tuning trial and, by scaling the horizon, roughly 40–50
minutes for a 900-second run. These are estimates under that shared workload,
not completed-trial timings or guaranteed final-run costs. DSpodFL's first four
trials completed in 60–63 seconds under the same batch. FSPDA-SA's trials were
interrupted; the observations establish an execution-cost concern, not an
accuracy or numerical-stability conclusion.

Investigate the gradient backend, repeated model construction and tensor-vector
packing/update operations using the saved DAG and executor timing counters.
Separate actual executor overhead from legitimate additional local updates:
sparse, overlapping communication lets FSPDA advance many more iterations per
simulated second. Validate any resident or batched implementation against the
scalar recurrence, preserve charged gradient durations and exact wire bytes,
and measure both numerical equivalence and end-to-end wall time. Reconsider
inclusion only after a short CIFAR pilot supports a manageable full-batch cost.

### DeTAG: numerical stability

Evidence: `output/leaderboard-cifar10-32-extended13/tuning/` contains the raw
trials. On 32 IID clients with a ring, phase length 2 and spectrum-derived eta,
LR .003 ended at 10% validation accuracy with loss about 820; LR .01 also ended
at 10%, with loss about 5.73e13. LRs .03, .1 and .3 produced non-finite loss at
300 seconds. Reclassified divergent outcomes are recorded in
`output/leaderboard-cifar10-32-extended17/tuning/`.

1. **Locate the first unstable update.** Reproduce a short seeded run with
   phase-level validation loss, gradient/tracker/parameter norms, client model
   disagreement, and the first non-finite tensor. Record the actual mixing
   matrix, spectral radius, eta and phase length. Compare individual local
   models with the global average diagnostically to locate where growth starts;
   retain the global-model readout for leaderboard scores.
2. **Check numerical correctness before interpreting a new LR search.** Extend
   the existing independent tests with longer deterministic trajectories and a
   small GNLeNet case. Check gradient division by phase length, the model-then-
   tracker update order, recurrence reset at each phase, parameter/vector
   ordering and dtype, and the tracker-mean conservation identity. Compare CPU
   and CUDA with the reference execution path. Existing small logistic/digits
   checks are useful evidence but do not explain the current CIFAR instability.
3. **Separate mixing effects from LR sensitivity.** On the same validation
   workload, use a complete graph and an eta=0 control, then restore the
   32-client ring and automatic eta. Compare phase lengths 1 and 2 as diagnostic
   controls. These controls change method settings and must be reported as such;
   phase averaging still uses fixed parameters and `local_steps` stays 1.
4. **Test a declared smaller LR range after the checks.** A candidate replacement
   grid is `[1e-5, 3e-5, 1e-4, 3e-4, 1e-3]`, again five candidates with equal
   budgets. This is a proposal, not a known solution. Freeze phase length, eta
   and topology before the replacement search. Confirm the selected LR remains
   finite through a 900-second validation pilot, then perform fresh held-out
   evaluation. Preserve the original failed grid and any implementation changes.

Implementation and maintained contracts: [DeTAG](detag.md),
`tests/test_detag.py`, `dasklearn/optimizer/detag.py`, and
`dasklearn/simulation/detag/`.

### AD-PSGD: execution cost

Evidence: `output/leaderboard-cifar10-32-extended17/tuning-new/runs/adpsgd-*/`
contains the partial workers, saved workflows and executor progress. The stop
snapshot is `output/leaderboard-cifar10-32-seven-launch/adpsgd-deferred.json`.
The observed task throughput suggested roughly 40 minutes per 300-second trial
with concurrent workers; that was an extrapolation. Passive clients continuously
compute gradients, while active clients wait for model exchanges. A large
numerical workload can therefore be legitimate protocol behavior.

1. **Build a measured cost breakdown.** Use a small fixed saved workflow and
   repeat scalar CPU/CUDA execution. Separate DAG generation and loading,
   gradient forward/backward, model copying/device transfers, gradient updates,
   aggregation, evaluation and artifact I/O. Record completed gradient counts,
   updates/second, peak CPU/GPU memory and ready-task parallelism. Compare an
   isolated worker with the current concurrent workload to measure contention.
2. **Audit task counts and dependencies.** Verify that each passive gradient is
   computed once and applied once to the correct current model, including an
   exchange arriving during computation. Check that model reconstruction or
   repeated copies are not doing avoidable work. Do not cache gradients across
   different parameters/minibatches or remove stale-gradient dependencies.
3. **Optimize execution while preserving the update rule.** Investigate resident
   gradient workspaces and batching only independent, ready gradient tasks.
   Profile the separate `gradient_update` path and immutable state copies before
   attempting fusion. Current progress records also show the generic CUDA stream
   batching guard falling back after a state-equivalence assertion; reproduce
   that mismatch in isolation rather than disabling the guard.
4. **Measure before scaling up.** Compare complete model states, model readouts,
   task RNG/minibatches, simulated event order and traffic with the reference
   executor. Run short workload-scaling checks, then complete one 300-second
   validation trial and a 900-second validation pilot to obtain measured costs.
   Publish the expected campaign cost before restoring the full leaderboard
   matrix. Slowing passive clients or capping their updates would change the
   benchmark/protocol and is not an execution optimization.

Implementation and maintained contracts:
[AD-PSGD](baseline-validation.md#ad-psgd), `tests/test_classic_baselines.py`,
`dasklearn/simulation/adpsgd/`, and the shared gradient execution backend.

### SWIFT: execution cost and stale communication

Evidence: `output/leaderboard-cifar10-32-extended/` retains the interrupted
search. Its first trial ran about 1,135 seconds without reaching the final
validation checkpoint. With 32 clients and 20 ms steps, a 300-second horizon
schedules 480,000 local updates. SWIFT keeps training while dense broadcasts
queue. Existing short digits experiments also show poor accuracy with congested
links; that is a separate observation from slow executor performance.

1. **Reuse the AD-PSGD gradient-execution investigation.** Profile gradient-only
   tasks, `swift_pack`, `swift_update`, immutable model copies and ready-task
   parallelism. Establish completed gradients/second and memory costs on a small
   saved DAG before trying another long LR search. Measure DAG/event generation
   separately from tensor execution and global-model evaluation.
2. **Measure queued communication.** Record queued bytes/messages, completed
   deliveries, mailbox version lag and simulated age of cached models over
   time. Check whether packet snapshots and retained DAG references cause
   avoidable execution/memory overhead. Optimize representation and release
   values only after their last consumer; do not drop or coalesce broadcasts
   that the implemented protocol charges.
3. **Diagnose learning behavior separately.** Use validation-only ample-bandwidth
   and congested controls with a fixed LR to distinguish numerical errors from
   stale-cache effects. For heterogeneous compute, compare empirical activation
   rates with the configured influence weights; do not assume uniform weights
   compensate for unequal update rates. Any changed communication interval or
   influence policy needs an explicit variant and its own declared settings.
4. **Verify and stage re-entry.** Check multistep model/cache trajectories,
   broadcast-before-gradient ordering, immutable packets, RNG, and exact
   delivered/partial traffic against the reference. Complete a 300-second
   validation pilot, then a 900-second one, and report measured cost and learning
   behavior before adding it back. Confirm compatibility with the leaderboard's
   offline global readout without feeding the average into training.

Implementation and maintained contracts: [SWIFT](swift.md),
`tests/test_swift.py`, `dasklearn/simulation/swift/`, and
`dasklearn/optimizer/swift.py`.

### Conditions for returning to the leaderboard

Keep old artifacts and identify whether each change fixes an implementation bug,
improves executor cost, or introduces a protocol variant. An executor-only change
must preserve causal model states, RNG, simulated capacities and traffic; use the
existing numerical equivalence guards and add a regression for the observed
failure. Do not weaken a guard to obtain higher throughput.

Use validation data for diagnosis and LR selection. Report diagnostic effort and
all replacement grids separately: an expanded search must not be presented as
the original equal-budget search. After fixing non-LR settings, use the same
declared candidate count, seeds, simulated tuning budget and validation subset
as the comparison protocol. Freeze the LR before fresh held-out evaluation.
Replacing an algorithm's update rule invalidates reuse of its earlier training
results; executor equivalence must be established before reusing results after
performance work.

Before a full campaign, show measured execution/memory costs and a realistic
completion estimate. Check the selected configuration at the intended horizon
under both data partitions and hardware profiles. Keep `local_steps=1`, the
global-average readout and explicit failure reporting. Stable but weak results
should be reported honestly; high accuracy is not a correctness criterion.
These investigations do not change or interrupt the active five-addition batch.

When extending baselines that are still running, `--follow-baselines` permits
live merging without stopping their workers. Combined reports continue to read
their latest results and remain provisional until all paired checkpoints exist.
The extension waits for those sources before declaring completion. Use a worker
count that accounts for the independent jobs; the eight-worker setting assumes
the prior batch has finished. All original output directories remain intact.

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
Every candidate must reach its full budget on every tuning seed. Non-finite
global validation metrics at that exact endpoint make a candidate ineligible;
all other failures or incomplete searches preserve their artifacts and
**block final evaluation**, so a partially evaluated grid cannot silently
select a winner. No automatic retries are added.

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
