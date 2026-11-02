# Conflux V2 expanded convergence study

Times are simulated elapsed seconds with charged network/compute costs. GPU execution wall seconds are separate.
All-node validation snapshots include idle nodes. Final test scores use all final node checkpoints and the full test split.
Learning rates are selected per workload/algorithm/partitioner by validation accuracy area on tuning seed 10001 under heterogeneous capacities.
Confirmation seeds: [0, 1, 2]. Network and compute assignments use independent RNG streams and preserve arithmetic mean capacities.
Unreached thresholds are right-censored. A strict win requires nonoverlapping first-crossing checkpoint intervals.

Completed runs: 92. Confirmation runs: 0.

Artifacts: study.json, provenance.json, selection.json, per-run config.json/record.json/run.log, raw DAGs and node metrics.
Raw checkpoints/DAGs remain on sacs006 under /mnt/data/mdevos. Historical experiments were not reused.
