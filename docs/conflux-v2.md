# Conflux V2

Conflux V2 is available as `--algorithm conflux-v2`. This integrates the original
averaging protocol from `origin/conflux-v2` at `99ee912`, plus the deterministic
sample/layout cache from follow-up commit `3723378`. Historical research outputs
and the separate FedOpt experiments remain on their branches.

A seeded uniform sample trains from the current common model. Each trainer splits
its updated model into chunks and uploads each chunk once to its assigned
reducer. Every reducer uses the same contributor set and a fixed summation order.
Reducers send their averaged chunks to all members of the next training sample;
each recipient reconstructs and publishes the same common result.

Reducer assignment uses public static bandwidth capacities. It considers the
highest-bandwidth peers up to `relay_count`, then assigns each successive chunk
to minimize assigned chunk load relative to bandwidth. Equal-capacity candidate
order rotates each round. Reducers may also train, or be outside the sample;
all roles share the peer's contended network endpoint. More chunks do not always
improve throughput, because each transfer incurs metadata and delivery latency.

With `success_fraction=1` and `--immediate-uploads`, trainers upload immediately
after finishing, overlapping communication with other trainers' computation.
Every reducer still waits for the full common sample. Otherwise, a temporary
coordinator freezes the first `ceil(sample_size * success_fraction)` ready
contributors before uploading. Partial participation can bias learning when
client speed correlates with data; the leaderboard fixes full participation.

Stable membership is required. There is no reducer/coordinator failure recovery
or congestion-adaptive placement. Chunking, averaging and reconstruction cost zero
simulated compute time. Training and transfers use the shared simulator resource
APIs; the implementation does not add momentum or Adam at reducers.

## CIFAR-10 / GNLeNet leaderboard

The campaign uses 32 clients, the existing CIFAR-10 split and capacities, and the
same validation-only search budget as Teleportation. It jointly tunes LR, local
steps, active clients, chunks and reducer cap, then freezes every selected option
and one shared validation checkpoint for all four held-out settings. The readout
is the uniform parameter average of all physical clients' latest published
models, including inactive retained checkpoints. This differs from measuring
only the latest common sample aggregate.

```sh
python scripts/run_methodology.py --algorithm conflux-v2 --plan
python scripts/run_methodology.py --algorithm conflux-v2
```

See [the campaign declaration](../experiments/methodology/conflux-v2-tpe-dirichlet01-h20.json)
and [leaderboard methodology](leaderboard.md#conflux-v2-jointly-tuned-sample-and-reducer-layout).
Numerical checks cover common averages, partial-contributor consistency, upload
overlap, byte conservation, serialized training and cached execution equivalence.

The completed campaign selected LR `0.051598546094309986`, H=15, two trainers,
32 chunks and a reducer cap of eight. All four held-out settings reuse these
values and the validation-selected 900-second checkpoint. See the
[audited results](../experiments/methodology/results/conflux-v2-cifar10-gnlenet-20261109.json)
for the complete search, per-seed test curves and source provenance.
