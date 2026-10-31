# QG momentum / Conflux V2, CIFAR-10, 64 nodes

See the [experiment report](../../docs/experiments/qgm.md).

- `summary.csv`, `comparison.csv`: per-seed and grouped final **10000-image test**
  metrics, plus separately named final validation metrics. Reported accuracy is
  the mean of all 64 nodes' stored models, then averaged over seeds.
- `curves.csv`: all-node means on the fixed balanced **1000-image validation**
  subset at 150/300/450/600/750/900 simulated seconds, with aggregate traffic.
- `latest-aggregate.csv`: distinct V2 validation metric for the maximum recorded
  round; includes node age in rounds. Final full-test aggregate metrics are
  separate columns in `summary.csv`.
- `time-to-accuracy*.csv`: first observed validation checkpoint per target.
  Blank observed times/traffic mean unreached and right-censored after 900 s.
  Aggregate means summarize reached seeds only; always inspect `reached`.
- `plan.json`, `selection.json`, `validation-scores.csv`, `tune-audit.json`:
  frozen design, validation-only tuning and LR choices.
- `audit.json`, `provenance.json`: all-node coverage, finite metrics, capacity,
  traffic/work consistency, source and environment evidence.
- `collect.py`, `plot.py`: analysis code operating on the retained server runs
  and compact results respectively. PNG/PDF figures separate validation curves
  from final test metrics and actual GPU execution time from protocol time.

Raw workflow DAGs and per-node files remain on sacs006 at
`/mnt/data/mdevos/conflux/results/qgm64`. No raw trained weights are claimed to
be embedded in the saved DAGs. Three seeds and a small model/LR grid provide
screening evidence, not paper reproduction or a universal algorithm ranking.
