# Why the first incoming-model MIA looked weak

The main cause identified by these controls is the **nonmember distribution**.
The original comparison class-matched nonmembers to each target's training
examples. That deliberately removes label-distribution information, which is a
substantial source of membership signal under non-IID partitioning. It measures
a narrower task than comparing local members with samples from the global test
distribution.

## Controls on the exact same received D-PSGD models

All 640 recorded D-PSGD deliveries (10 runs, four checkpoints, 16 directed edges)
were checked: each references the sender's completed `train` output for the
correct round, before aggregation. This diagnostic scores the final checkpoint,
once per distinct sender model rather than counting its two recipients twice.
For each target, member identities and counts remain fixed between cohort
policies; only the held-out calibration and evaluation samples change.

Final round 100, eight peers, five seeds, 500 SGD steps per peer. Mean ± sample
SD across seed means:

| Partition | Nonmembers | Loss MIA ROC-AUC | Balanced accuracy, 10% calibration FPR | Balanced accuracy, 50% calibration FPR |
|---|---|---:|---:|---:|
| IID | Class matched | 53.13 ± 3.65% | 50.10 ± 1.41% | 52.11 ± 2.03% |
| IID | Global distribution | 53.32 ± 3.32% | 50.07 ± 1.33% | 52.25 ± 1.92% |
| Dirichlet | Class matched | 53.01 ± 3.41% | 50.73 ± 1.20% | 51.30 ± 3.22% |
| Dirichlet | Global distribution | **66.99 ± 5.16%** | **55.84 ± 2.62%** | **62.19 ± 3.74%** |

These are independently calibrated operating points, not thresholds optimized
using evaluation membership labels. The nominal calibration FPR is not a promise
about actual evaluation FPR. AUC sweeps thresholds and is not classification
accuracy; 67% AUC must not be described as 67% attack accuracy.

The 10% calibration-FPR operating point also compresses the balanced accuracy
differences relative to other thresholds. Even for global nonmembers, a single
near-chance operating point is an incomplete description of attack performance.

## Overfitting and no-mixing control

The final received D-PSGD models have the following unweighted per-sender mean
training and global test accuracies:

| Partition | Sender training accuracy | Global test accuracy |
|---|---:|---:|
| IID | 97.31% | 94.00% |
| Dirichlet | 96.63% | 82.63% |

The non-IID gap is real. Class-matched nonmembers make the loss comparison
substantially harder than global nonmembers, because they have the same labels
as the examples on which a sender specializes.

As a positive control, each client's classifier was also fitted independently,
with the same initialization, partition, per-round task seeds and 500-step
optimizer budget, but **without model mixing**. This is not D-PSGD:

| Partition | No-mixing training accuracy | No-mixing global test accuracy | Global-nonmember loss MIA AUC |
|---|---:|---:|---:|
| IID | 99.54% | 90.92% | 59.74 ± 2.61% |
| Dirichlet | 98.97% | 56.34% | 78.13 ± 2.85% |

This confirms that the same evaluator detects stronger leakage when the model
fits locally and generalizes less well. It does not mean that disabling mixing
is an attack on a faithfully executed D-PSGD protocol.

## Interpreting the global signal

An evaluator-only diagnostic that scores candidates using their label and the
target's complete label histogram reaches 76.87% AUC for the Dirichlet global
cohorts and exactly 50% for the matched cohorts. This oracle assumes knowledge
of the target distribution and is **not** an attack with the main experiment's
assumptions. It demonstrates how much class-distribution information can be
present in the broader membership task. Global-cohort results should not all be
interpreted as sample-specific memorization.

The [Shatter paper](https://petsymposium.org/popets/2025/popets-2025-0043.pdf)
evaluates loss-based MIA against node training samples and the test set. Its
CIFAR experiment uses 100 nodes, Dirichlet alpha 0.1, and ResNet-18; the unprotected
EL baseline reaches about 89.7% AUC there. That is a different workload and
baseline from the small eight-peer digits experiment. The paper also reports
much lower MIA AUC on Twitter, illustrating workload dependence.

The practical correction is to report **both cohort policies**, lead with ROC-AUC
when comparing to that literature, and keep calibrated TPR/FPR and balanced
accuracy as additional measurements. The initial class-matched near-chance result
does not establish that D-PSGD is safe against MIA.

## Reproduction and artifacts

```sh
python scripts/diagnose_membership.py \
  --source output/membership-incoming-digits \
  --output output/membership-diagnostics --standalone-control
```

Use a fresh output directory when repeating. The original experiment directories
are not modified. [manifest.json](manifest.json) records the source experiment,
control definitions and execution-source hash. [targets.csv](targets.csv),
[per-seed.csv](per-seed.csv) and [summary.csv](summary.csv) include all controls,
including oracle diagnostics that are explicitly excluded from attack claims.
