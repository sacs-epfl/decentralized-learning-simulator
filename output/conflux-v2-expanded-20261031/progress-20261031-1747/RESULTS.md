# Progress at 17:47 CET, 31 October 2026

92/96 tuning runs completed; two MobileNet/Teleportation IID runs active; two ResNet/Teleportation IID runs not yet started. Zero failed runs. No confirmation runs yet.
The detached launcher was alive after 1 h 45 min, with one worker per GPU. Approximately 116 GB remained on /mnt/data.

These are exploratory tuning results, using seed 10001 with network and compute heterogeneity together. Learning rates are selected separately per workload/algorithm/partitioner by validation accuracy area (AUC). All scores are means across the 16 nodes' stored checkpoints. No test-set confirmation or seed uncertainty is available yet.

| Workload | Data | V2 AUC | Best completed baseline AUC | V2 final accuracy | Baseline final accuracy |
|---|---|---:|---|---:|---:|
| cifar10-gnlenet | dirichlet | 18.45 | moshpit: 16.09 | 25.97% | 22.00% |
| cifar10-gnlenet | iid | 37.33 | moshpit: 33.67 | 46.31% | 41.20% |
| digits-mlp | dirichlet | 10.20 | dpsgd: 11.40 | 13.88% | 11.83% |
| digits-mlp | iid | 14.27 | dpsgd: 11.94 | 25.99% | 14.17% |
| tiny-mobilenet | dirichlet | 0.94 | moshpit: 0.63 | 1.71% | 1.16% |
| tiny-mobilenet | iid | 0.94 | moshpit: 0.63 | 2.03% | 0.99% |
| tiny-resnet18 | dirichlet | 2.94 | moshpit: 1.95 | 5.40% | 3.46% |
| tiny-resnet18 | iid | 3.82 | moshpit: 2.50 | 7.00% | 4.08% |

AUC is average validation accuracy over the observed interval, expressed in percentage points. Final accuracy is measured at 60 s (digits), 600 s (CIFAR), or 1800 s (Tiny). MobileNet IID and ResNet IID baseline rankings are incomplete because Teleportation has not completed.

## Selected observations

- CIFAR-10 IID: V2 reaches 40% at the 350 s checkpoint; Moshpit at 500 s. The observed checkpoint ratio is 1.43x, with 50 s checkpoint spacing. V2 final validation accuracy is 46.31%, versus 41.20% for Moshpit.
- CIFAR-10 Dirichlet: V2 reaches 20% at 500 s; Moshpit at 550 s. These adjacent checkpoint intervals overlap at their boundary; this is not a strict interval-separated speed win. Final validation accuracy is 25.97% versus 22.00%.
- Tiny/ResNet18 Dirichlet: V2 reaches 2% at 750 s; Moshpit at 1050 s, an observed checkpoint ratio of 1.40x. V2 reaches 5% at 1800 s; Moshpit does not reach 5% within this budget.
- Tiny/ResNet18 IID: V2 reaches 5% at 1350 s; the completed baselines do not reach 5% by 1800 s. Teleportation is still pending. Final V2 accuracy is 7.00%.
- Digits: no algorithm reaches the lowest prespecified 50% target within 60 s. V2 leads IID validation AUC, but D-PSGD leads non-IID validation AUC; V2 has higher final accuracy in that non-IID run, so the curves cross. There is no established time-to-target result.
- Tiny/MobileNet: V2 finishes at 2.03% IID and 1.71% non-IID. The model remains in very early learning, and almost all time-to-accuracy targets are censored. Its IID Teleportation comparison is pending.

These observations support investigating V2's speed advantage on the image workloads, but cannot establish a general fastest-algorithm claim. The 576 confirmation runs will test the frozen rates with seeds 0/1/2 across uniform, network-only, compute-only and joint heterogeneity, and full final test sets. The low attained accuracy indicates that longer-budget follow-up may be needed to study mature convergence. The current preregistered study has not been changed.

All times above are simulated protocol elapsed seconds. GPU wall time measures simulator execution cost and is not used for convergence ranking.

## Artifacts

snapshot.json contains the exact per-run status and metrics. selected-tuning.csv records the selected learning rates, AUC, final accuracy and threshold crossings. runs/ contains compact per-run record.json and curve.csv snapshots. Raw DAGs and per-node metrics remain on sacs006.
