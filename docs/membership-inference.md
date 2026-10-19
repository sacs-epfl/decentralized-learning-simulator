# Loss-based membership inference

The optional privacy audit measures whether low prediction loss distinguishes
examples assigned to another client's training partition from held-out examples.
It is independent of the learning protocol and executes as a numerical DAG sink.
It does not change training, communication, or simulated protocol time.

## Attacker and interpretation

The default `--view local` comparison uses an honest-but-curious participant's **own complete
local model after aggregation**. Each other participant is audited separately.
Attackers know the candidate example and its label, can evaluate their own model,
and have auxiliary labeled examples known not to be in the training data. They do
not obtain another participant's private checkpoint, missing model chunks,
training records, or the simulator's global state. The evaluator uses training
partition ownership only to construct and score the audit cohorts.

This measures membership leakage accessible through a local model. It does not
attribute an example to a particular source client: other clients' training
examples can influence the attacker model through mixing. It also does not test
received-model attacks, chunk reconstruction, identity hiding, collusion, active
attacks, or full message histories. D-PSGD peers also receive full neighboring
models; that additional attack surface is outside this first common-view audit.
In particular, a similar local-view score for D-PSGD and Shatter does not
establish equal privacy under stronger attackers. `--view incoming` instead
audits delivered models/chunks as described below.

Membership means **assigned to the target training partition**, rather than a
trace proving that an example appeared in an SGD batch. A short run can leave
some assigned examples unsampled. No differential privacy guarantee is inferred
from empirical attack failure.

## Cohorts and attack rule

For each dataset seed, every held-out class is shuffled and split into two
disjoint pools. These pools are fixed across targets, attackers, checkpoints and
protocols. One pool supplies auxiliary nonmembers for calibration; the other
supplies evaluation nonmembers. Target members are sampled without replacement.
With the default `--cohort-policy class_matched`, the three sets have equal sizes
and exactly matched class histograms. If a class
lacks sufficient held-out examples, its member cohort is trimmed. Reported
counts and class histograms describe the actual evaluated subset, not the entire
client partition. Matching classes reduces label-shift confounding, although it
cannot eliminate all differences in example difficulty or source distribution.

`--cohort-policy global_nonmembers` keeps the exact same selected member IDs and
sample counts as the class-matched control, but samples calibration and evaluation
nonmembers from their respective **global** held-out pools without matching the
target's labels. This preserves label-distribution information that can make
membership inferable under non-IID partitions. The two policies answer different
questions: global nonmembers measure the broader membership task; class matching
controls for this class-distribution signal and focuses on within-class leakage.
Neither should be silently substituted for the other. Every result records the
policy and all three label histograms. Members still inherit the matched control's
trimming, so the global variant is not an exact paper reproduction.

Let sorted calibration losses be `c[0], ..., c[n-1]`. For a requested empirical
false-positive budget `q`, select `threshold = c[floor(n*q)]` and predict a member
when `loss < threshold`. Strict comparison handles ties conservatively. Neither
evaluation member losses nor evaluation nonmember losses select the threshold.
The audit assumes access to auxiliary nonmembers representative of the candidate
label distribution. The target label histogram used for matching is an audit
construction choice, not a claim that a deployed attacker knows membership.

The default `q=0.1` targets at most 10% false positives **on calibration examples**.
The observed evaluation FPR can differ; it is always reported. A cohort of 100
nonmembers resolves FPR only in 1% increments, so these small experiments cannot
support claims at a 0.1% false-positive rate.

Outputs include:

- Balanced attack accuracy: `(TPR + 1 - FPR) / 2`, with a 50% chance baseline.
- ROC-AUC of negative loss, with ties receiving half credit and a 0.5 baseline.
- Evaluation TPR, FPR, attack advantage `TPR - FPR`, and calibration FPR.
- Threshold, sample counts, FPR resolution, mean losses, cohort hash, attacker,
  target, checkpoint round and simulated observation time.

This is a nonmember-calibrated loss-threshold baseline, not an exact reproduction
of Yeom's training-mean threshold attack or LiRA. See
[Yeom et al.](https://arxiv.org/abs/1709.01604) for loss-based membership inference
and [Carlini et al.](https://arxiv.org/abs/2112.03570) for evaluation at low FPR.

## Reuse with other protocols

After simulation, attach an audit to a complete model reference known to be
available to the attacker at the supplied time, before numerical DAG execution:

```python
from dasklearn.privacy import add_local_model_mia_task
from dasklearn.tasks.local import execute_local

round_nr, model = sim.latest_models[0]
task = add_local_model_mia_task(
    sim, attacker=0, model=model, round_nr=round_nr,
    time_us=sim.current_time, targets=[1, 2],
    max_examples=128, target_fpr=0.1,
)
sim.workflow_dag.build_task_indices()
sim.workflow_dag.check_validity()
sim.workflow_dag.save_to_file(sim.data_dir + "/workflow_dag")
execute_local(sim, retain_results=False)
```

Each task writes `mia-<task-name>.json` in the run directory, so separate workers
do not append to a shared file. The task has explicit model dependencies and an
importable operation contract, allowing streaming execution and saved-DAG replay.
The audit takes a private model copy and uses deterministic evaluation loading;
training state and RNG are preserved. Audit computation is offline and does not
consume simulated participant resources. The comparison records its wall time
separately from protocol time.

The built-in adapter supports synthetic data, digits, native CIFAR-10 tensors
(`dataset_base_path`, `lenet`/`gnlenet`), and native Tiny ImageNet tensors. It uses
the same loss objective as the existing training/evaluation path, with no random
augmentation. For other classification datasets/models, the reusable core is:

```python
from dasklearn.privacy import per_example_losses, evaluate_loss_attack

member_losses = per_example_losses(model, members)
nonmember_losses = per_example_losses(model, nonmembers)
calibration_losses = per_example_losses(model, calibration_nonmembers)
metrics = evaluate_loss_attack(
    member_losses, nonmember_losses, calibration_losses, target_fpr=0.1,
)
```

The caller supplies disjoint datasets with deterministic evaluation transforms.
`per_example_losses` supports mapping-style Torch datasets, configurable feature
and label keys, device/batch size, and cross entropy or NLL. The pure
`evaluate_loss_attack` function also accepts loss arrays from other frameworks.

## Attacking incoming models and chunks

Use `--view incoming` to audit the model information actually delivered to peers:

```sh
python scripts/compare_membership.py --view incoming \
  --cohort-policy global_nonmembers --model-bytes 10994 \
  --output output/membership-incoming-global-digits \
  --variants dpsgd shatter-k1 shatter-k4 shatter-k16
```

Omit the cohort option to run the class-matched control in another fresh output
directory. `--model-bytes 10994` fixes the simulated wire size of the digits MLP
for these paired experiments; it does not change its weights or architecture.
The remaining defaults match the local-view experiment below. At the selected
sender rounds (25, 50, 75 and 100 by default), the passive observer records each
completed delivery immediately before its protocol callback. In-flight,
cancelled and not-yet-delivered payloads are not observations, even when bytes
have already been transmitted. Self-deliveries are excluded.

D-PSGD sends full post-local-training models. The attacker evaluates the exact
received model against the **sender's** training partition and independently
held-out examples. No aggregation with the receiver's model is performed.

For Shatter, the attacker collects the unique chunks received by one physical
peer from one sender in one round. Multiple virtual neighbors can deliver the
same chunk; duplicates do not increase parameter coverage. The attacker forms a
complete candidate model by overwriting those slices in its own latest completed
model, frozen immediately before the first delivery in that group. Later own
training or aggregation cannot change this background. If the receiver has not
yet completed any local model, shared public initialization supplies the missing
values. Loss is evaluated after the group's last actual delivery. No chunks are
combined across rounds, senders or receivers, and no unreceived victim state is
supplied. Shatter with one chunk exposes the full received state.

This is one concrete loss attack on partial models, using receiver-state
completion. It may be weaker than optimization-based completion, attacks using
full histories, or collusion. A low score does not prove the chunks contain no
membership information. Sender identities, rounds and chunk indices are visible
in the current implementation and are used by the attacker; this experiment
does not model the Shatter paper's identity-hiding layer.

The audit uses two kinds of participant-visible observations for the background:
completed model-valued local training tasks and calls to `publish_model`.
`Simulation.observers` is opt-in; observers receive payload delivery, completed
compute and model publication callbacks. The built-in observer records references
only and creates numerical audit tasks after simulation. Protocol events, RNG,
traffic, model updates and timings remain unchanged.

To reuse this view with full-model or compatible chunk-exchange protocols:

```python
from dasklearn.privacy import IncomingModelObserver

observer = IncomingModelObserver(
    attackers=[0], rounds=[25, 50],
    chunk_count=4,  # None for full-model payloads
)
sim.observers.append(observer)  # Attach BEFORE simulation.
sim.simulate()
audits = observer.attach_audits(
    sim, max_examples=128, target_fpr=0.1, cohort_policy="global_nonmembers",
)
sim.workflow_dag.build_task_indices()
sim.workflow_dag.check_validity()
sim.workflow_dag.save_to_file(sim.data_dir + "/workflow_dag")
execute_local(sim, retain_results=False)
```

The adapter requires `TaskRef` payloads and visible round metadata. Chunked
payloads additionally require visible chunk indices and the public equal-sized
`ChunkManager` layout; unsupported layouts require another adapter. Complete
backgrounds from other custom training operations can be exposed through the
existing `publish_model` API.

Each attack row adds `first_delivery_time`, `last_delivery_time`, `delivery_count`,
the unique chunk count, visible state-element count/fraction and completion
strategy. `runs/<run>/incoming-observations.json` records the delivered references
and frozen background used to build the actual executed audit DAG.

**Incoming scores are conditional on a delivery.** A non-neighbor in D-PSGD has
no incoming model to attack; that pair is omitted, not assigned a fictitious 50%
success rate. `observed_pairs` and `observed_pair_fraction` report how many of the
possible directed attacker/sender pairs are observed at each checkpoint.
`visible_state_fraction` is averaged over observed pairs, not all pairs. Protocols
therefore have different observation populations; the numbers are not a causal
comparison on identical network edges. `model_accuracy` still reports the
receiver's normal local-model utility at the corresponding round checkpoint,
not accuracy of the reconstructed partial candidate or a state at arrival time.
Its summary is also weighted by the number of observed pairs per receiver.

## Reproduce the local-model comparison

```sh
python scripts/compare_membership.py --output output/membership-digits \
  --variants dpsgd shatter-k1 shatter-k4 shatter-k16 \
  --dataset digits --peers 8 --degree 2 \
  --rounds 100 --checkpoint-interval 25 --local-steps 5 \
  --partitioners iid dirichlet --alpha 0.3 --seeds 0 1 2 3 4
```

Use a fresh output directory; the runner refuses to overwrite prior experiments.
This offline digits experiment uses no downloads or broker processes. Both
protocols run 100 local-training rounds with the same optimizer settings,
initialization and data partitions within a seed. Every peer is separately
treated as an attacker; these are independent views, not a colluding group.
There is no selection of the most successful attacker or checkpoint using test
results. The held-out data supplies both audit pools and model utility evaluation;
it must not subsequently be treated as an untouched dataset for tuning.

The degree is 2 on the real-node D-PSGD graph and on Shatter's virtual-node graph.
Shatter's topology changes between rounds, while D-PSGD uses a fixed topology.
The `shatter-k1` control helps show the effect of changing topology, although this
comparison does not isolate chunking from every other protocol difference.
Training rounds and SGD-step counts are matched; simulated time, bytes and
accuracy need not be equal. Runs fail if the simulation safety limit prevents
completion of the matched training budget.

Artifacts:

- `manifest.json`: arguments, versions, source hash and attacker assumptions.
- `attacks.csv`: each attacker/target/checkpoint measurement.
- `per-seed.csv`: equal-weight pair means within each training seed.
- `summary.csv`: means and sample standard deviations across independent seeds.
- `runs.csv`: training budgets, bytes, simulated time and execution wall time.
- `comparison.png`: attack accuracy and model utility over completed rounds.
- `runs/`: settings, actual executed DAGs, utility results and task audit JSON.

Pairs reuse models and held-out examples and therefore are correlated. The
reported variation is across seed means, not a confidence interval obtained by
treating all attacker/target pairs as independent. This small workload is a first
measurement, not a privacy ranking for other datasets or attack models.

## Validation

`tests/test_membership.py` checks score orientation, tied-score chance behavior,
strict calibration budgets, evaluation/calibration separation, class matching,
cohort reproducibility and seed-level aggregation. A deliberately memorizing
classifier provides a positive control with perfect attack success. Tests also check complete model
states against unaudited execution, plus streaming and saved-DAG replay for
D-PSGD, Shatter and Moshpit. No broker replay has yet been validated for this
operation.

`tests/test_incoming_membership.py` checks completion against a known model,
missing-slice preservation, chunk remainders, sender/round/recipient separation,
duplicate handling, frozen backgrounds and public-initialization fallback. Tests
cover immediate and delayed delivery, exclusion of payloads whose latency has not
expired, unchanged protocol traces/traffic/trained states, and streaming plus
saved-DAG replay for full models and 1/4/16-chunk Shatter views.

## Diagnosing weak MIA results

```sh
python scripts/diagnose_membership.py \
  --source output/membership-incoming-digits \
  --output output/membership-diagnostics --standalone-control
```

This replays saved D-PSGD DAGs into a separate output directory and verifies that
received references are the sender's completed training outputs. It compares
global and class-matched nonmembers on the same selected members, measures
sender-specific training accuracy/loss and global test accuracy/loss, and reports
both 10% and 50% independently nonmember-calibrated FPR operating points.

The optional no-mixing control fits each client's classifier independently using
the same initialization, dataset, per-round seeds, optimizer and SGD-step budget.
It is a diagnostic of local fitting, not a D-PSGD run or a protocol comparison.

Two additional **oracle diagnostics** must not be treated as deployable attack
results: the best threshold selected using evaluation membership labels, and
the AUC achievable from full target label frequencies known to the evaluator.
They help distinguish weak threshold choice and class-distribution signal from
other membership information. `targets.csv`, `per-seed.csv` and `summary.csv`
retain all controls without selecting whichever looks most favorable.
