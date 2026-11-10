# Plexus

`--algorithm plexus` implements Algorithms 1 and 2 of
[Practical Federated Learning without a Server, EuroMLSys 2025](https://euromlsys.eu/pdf/euromlsys25-10.pdf)
for stable membership. This is a dedicated protocol, separate from the original
Conflux prototype and Conflux V2. It is registered for the coordinator, broker
settings restoration, local executor, YAML/JSON experiment runner, benchmark
runner and the `plexus` alias in the CIFAR and synthetic comparison runners.

## Protocol contract

Each participant starts with the same complete, static membership view and the
advertised endpoint bandwidths. Peer identities are reproducible byte strings
`plexus:{seed}:{index}`, replacing IPv8 public keys. Each participant independently
sorts identities by `MD5(b"%s-%d" % (identity, round))` and selects the first
`sample_size` entries. The sampler first sorts the identities to preserve the
authors' tie ordering. No simulator-wide per-round membership oracle is used.

The aggregator is the highest-bandwidth participant **inside the current training
sample**. Equal bandwidths retain hash order. The initial sample trains from one
common seeded initialization. Each trainer immediately pushes its entire updated
model to that round's aggregator when training completes. The aggregator waits
for `floor(sample_size * success_fraction)` distinct current-sample uploads,
freezes those contributors and computes their uniform parameter average. The
check runs upon upload receipt, so even a threshold rounded down to zero consumes
at least one received model. It then pushes that same full model to every member of the next hash-derived sample.
Receipt of that aggregate activates training for the next round. There is no
readiness coordinator, reducer outside the sample, chunking or free broadcast.

Duplicates cannot contribute twice or activate training twice. Wrong senders and
uploads arriving after aggregation are ignored. A participant's overlapping local
training requests use the shared compute queue; full-participation rounds cannot
advance before every trainer's completed model arrives. For a finite round count,
the final aggregate is delivered and published, without starting another round.

All remote uploads and aggregate deliveries charge the serialized full-model
payload, one configured header and serialized metadata through the shared endpoint
bandwidth scheduler. Aggregators retain their advertised finite capacity. Self
delivery is local, with zero wire bytes and latency. Aggregation takes zero
simulated compute time, as in the other averaging baselines. Executor wall time
and simulated protocol time are different measurements.

Published client checkpoints are received common aggregates. Initially idle
clients retain the common initialization. Snapshot/global-average evaluation uses
the existing offline diagnostic; it is not a charged protocol-wide collection or
an accuracy measurement of only the latest active sample. Per-client network
statistics use the shared measurement files. `plexus-rounds.csv` additionally
records the aggregator, aggregation time, sample, frozen contributors, next sample
and aggregate task. `plexus-deliveries.csv` records individual delivery times.

## Paper versus released code

The authors' source was inspected at
[`82ae6cf48ba829df9c90473b7f9a80728c19d43f`](https://github.com/sacs-epfl/plexus/tree/82ae6cf48ba829df9c90473b7f9a80728c19d43f).
One protocol choice differs between the paper and that release:

| Decision | Paper Algorithms 1–2 | Released `dlsim/plexus/community.py` | This implementation |
| --- | --- | --- | --- |
| Aggregator membership for training round k | Highest bandwidth in sample k | `forward_trained_model` asks for aggregators in sample k+1 | Paper: sample k |

This difference can change traffic and training as sample overlap varies. This is
a paper-based implementation, not a bit-for-bit reproduction of the released
end-to-end system. Both the paper and the release use a floor threshold; this was
verified visually against Algorithm 2 because PDF text extraction corrupts its
mathematical glyphs.

The exact released sampler and parameter-averaging sources, with their MIT
license, are pinned under `tests/reference/plexus/`. Tests execute those original
class bodies with only unused framework imports removed, checking hash ordering
and numerical averages. They do not claim an executed reference-system transport,
timeout, membership or failure-recovery comparison.

## Supported scope and limitations

- Stable membership and a consistent initial membership/capacity view are required;
  churn is rejected. IPv8 discovery, ping/pong availability probes, membership
  reconciliation, aggregation timeouts, acknowledgements and failure recovery are
  not simulated. There is no claim to reproduce the paper's deployment overhead.
- The reference cancels obsolete training on newer aggregate arrival. The shared
  simulator queues local training requests instead; no interrupted compute is
  reclaimed. Partial-participation runs therefore constitute a declared
  nonpreemptive adaptation. Use `success_fraction=1` for the baseline campaign.
- Plain SGD with positive local steps and zero trainer momentum is supported.
  Persistent momentum migration is not implemented. Synchronous global barriers
  and the legacy `test_method=global` path are rejected; use snapshot or individual
  evaluation instead.
- GNLeNet and ResNet-18-GN have no BatchNorm running-statistic buffers. The shared
  averaging operation also averages floating model buffers; the released FedAvg
  averages parameters only. Models with such buffers require a separate audit.
- Static capacity-aware selection assumes capacities are known before training.
  Neither discovering nor periodically measuring bandwidth is modeled.

## Verification and use

`tests/test_plexus.py` checks the original sampler and averaging code, real
training DAGs, exact common averages and next-round inputs, highest-bandwidth
selection within the current sample, upload thresholds and late/duplicate
messages, recipient activation, local compute serialization, streaming execution,
saved diagnostics and CLI/settings restoration. An independent one-round analytic
test checks the exact byte counts and elapsed time for a contended single reducer,
including headers, metadata and latency, and verifies the response to bandwidth.

```sh
python -m pytest -q tests/test_plexus.py tests/test_algorithm_registry.py

python main.py --algorithm plexus --dataset synthetic --model mlp \
  --peers 6 --sample-size 3 --success-fraction 1 --rounds 3 \
  --local-steps 2 --momentum 0 --test-method snapshot --test-period 1 \
  --duration 10 --compute-graph-plot-size 0 --local --work-dir output/plexus-smoke

python main.py --algorithm plexus --dataset cifar10 --model gnlenet \
  --dataset-base-path data/cifar10 --peers 32 --sample-size 4 \
  --local-steps 5 --learning-rate .03 --momentum 0 --seed 100 \
  --partitioner dirichlet --alpha .1 --validation-set-fraction .1 \
  --evaluation-split validation --evaluation-examples 100 \
  --stop duration --duration 30 --test-method snapshot --test-period 15 \
  --compute-graph-plot-size 0 --local --device cuda:0 \
  --work-dir output/plexus-cifar10-preflight
```

The short CIFAR command is an execution check; it is not tuned or a paper result.
No completed Conflux or Teleportation scores or winning parameters are imported
into a Plexus tuning campaign.

## Recorded validation (10 November 2026)

The complete CPU suite passed: **1,120 passed, 25 skipped**, with five existing
dataset/dependency warnings. The dedicated protocol file passed all 23 cases.
The normal CLI executed a three-round synthetic run. Two CUDA CIFAR-10/GNLeNet
validation preflights (32 peers, four active trainers, H=5, 30 simulated seconds,
checkpoints at 0/15/30 seconds) completed on uniform and heterogeneous hardware.
Both used `success_fraction=1`; every recorded aggregate included all four sampled
trainers. They completed 16 and 12 aggregations, respectively. These are execution
checks rather than accuracy comparisons.

The [validation record](../experiments/methodology/results/plexus-implementation-20261110.json)
contains pinned reference hashes, protocol file hashes, test commands and paths to
the local test logs and preflight outputs under `output/plexus-validation-20261110/`.
