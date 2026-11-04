# MoTEF-VR

`motef-vr` implements Algorithm 2 of [Islamov, Gao and Stich, ICLR 2025,
Towards Faster Decentralized Stochastic Optimization with Communication
Compression](https://proceedings.iclr.cc/paper_files/paper/2025/file/39b77b5e422b4e070e2811b73ea9bcf7-Paper-Conference.pdf).
It is registered separately from [`motef`](motef.md), whose Algorithm 1
estimator, defaults and protocol remain unchanged. Both use `--motef-*` options.

## Recurrence and sampling

With one row per client and symmetric Metropolis matrix W:

```text
x_next = x + gamma * (W h - h) - learning_rate * v
q_h    = Compress(x_next - h)
h_next = h + q_h
m_next = Gradient(x_next; fresh_batch)
       + (1-lambda) * (m - Gradient(x; fresh_batch))
v_next = v + gamma * (W g - g) + m_next - m
q_g    = Compress(v_next - g)
g_next = g + q_g
```

Both consensus terms read old memories. The model step reads the old tracker;
the estimator reads the exact old m and **two fresh gradients**, never a cached
gradient from a previous batch. Even lambda=1 executes and charges both passes.
For a common completed iteration, mean(v)=mean(m), and the average model moves
by minus learning_rate times the previous momentum mean.

Each paired-gradient DAG task depends explicitly on both the previous completed
model and the proposal. It samples and materializes one minibatch, including
transforms, once. Independent private copies of that batch feed the two passes;
Python, NumPy and PyTorch forward randomness starts from the same task seed for
each pass, replaying the RNG state after batch materialization. Task seeds depend on session seed, client and iteration. Sampling is
fresh per iteration; it can legitimately select examples seen in earlier batches.
All supported RNG states are restored on return, including the selected CUDA
device. Custom transforms/forwards with private RNG generators or external mutable
state are outside this coupling contract.

Model buffers start from the previous local state on both passes. Only the
x_next forward's updated buffers survive; the old-point forward's buffer updates
are discarded. Parameters, old state, packets and input models are never mutated.
Weight decay is coupled separately at each gradient's own parameter vector, so
its contribution follows the same variance-reduced recurrence. No shared SGD
momentum or hidden optimizer step is applied.

## Initialization and source discrepancies

One charged local minibatch gradient initializes m0=v0. Set h0=g0=0 and both
weighted memories to zero. Clients share the seeded initial model; no initial
client-specific gradients, estimates or neighbor state are exchanged for free.
Round zero in the measurements denotes bootstrap, without a model step.

Algorithm 2 lists starting estimates as inputs and omits M0 from its input line,
although the recurrence uses it. We choose the tracking-consistent initialization
above, as for base MoTEF. This cold-memory, single-minibatch start differs from
the paper's warm initialization for its bounds. Theorem 3 also specifies an
initial batch-size lower bound involving unknown problem constants; this
implementation does **not** claim that bound or reproduce a large initial batch.

The authors' source was inspected at
[`10919bd38e093897329d1b096ba037701084d635`](https://github.com/mlolab/MoTEF/tree/10919bd38e093897329d1b096ba037701084d635).
`MoTEF.py:BEER_Momentum` implements the base EMA update, rather than Algorithm 2,
and initializes V to a gradient but M to zero. Its main Top-k helper orders signed
values. `synthetic/solvers.py:beerM` uses consistent full-gradient M/V initialization
and magnitude Top-k. These are base-method sources, not an executable VR oracle.
The VR implementation follows the published Algorithm 2 and is checked against
independently differentiated changing-minibatch objectives. We retain consistent
M/V initialization and magnitude Top-k; we do not claim a direct VR source-code
reproduction.

## Compute, communication and supported scope

Bootstrap costs one gradient duration at each client's own simulated speed.
Each iteration costs **two** such durations, serially, including heterogeneous
speeds. One paired DAG task therefore represents two gradient evaluations;
training-task counts are not gradient-evaluation counts. Work scheduled beyond
the horizon remains charged in scheduled compute totals, as in base MoTEF.

A single fused neighbor packet carries both compressed residuals after the
paired gradients finish. For D parameters, sparse channels retain
k=max(1,ceil(D*keep_ratio)) global coordinates each, with int64 indices and
parameter-dtype values. Payload bytes are 2*k*(8+element_size); identity sends
2*parameter_bytes. Zero selected values still consume bytes. Each neighbor
packet also pays one generic header and serialized round metadata. Explicit
payload accounting overrides `model_size_bytes`. Transfers use the standard
bandwidth contention and latency scheduler, including partial horizon traffic.

Clients wait for their own pair and all same-iteration neighbor packets before
updating weighted memories and publishing a completed model. Optional synchronous
mode adds an iteration-start global barrier. Early packets are buffered and
duplicates cannot repeat work. Snapshots reference only completed models (or the
seeded initialization before completion), never an unfinished proposal. Vector
operations, compression and graph construction have zero simulated cost; actual
executor wall time is measured separately. Model-delta transfers do not overlap
the gradient pair.

Supported scope matches base MoTEF: fixed ring/complete/seeded connected regular
graphs, one or more stable clients, global magnitude Top-k, unscaled random-k,
identity, `local_steps=1`, shared `momentum=0`, fully trainable nonempty floating
parameters of one dtype, individual or snapshot evaluation. Multi-step local SGD,
churn, directed/time-varying mixing, LoRA and quantization are excluded. CPU and
CUDA scalar execution are checked; other device types are rejected. Distributed
deployment is untested.
The importable operation contracts are serialized into saved DAGs and usable by
the ordinary worker/replay machinery. CUDA batching/residency optimizations for
ordinary SGD do not accelerate this paired custom operation. Convergence or
benchmark competitiveness is not implied by accepted numerical settings.

Measurements are `motef-vr-rounds.csv` and `motef-vr-mixing.json`. The standard
settings JSON preserves the separate algorithm and MoTEF-specific controls.
The CLI, local execution, experiment-file runner, comparison runners and saved
DAG replay accept `motef-vr`. `main.py --local --from-dir PATH --device cpu`
loads the saved DAG without simulating again; use a copy of the saved directory
when preserving the original evaluation files. No active leaderboard configuration is changed.

## Usage and validation

```sh
python main.py --local --algorithm motef-vr --dataset synthetic --model mlp \
  --peers 4 --rounds 5 --duration 3 --local-steps 1 --momentum 0 \
  --learning-rate .05 --motef-compressor topk --motef-keep-ratio .01 \
  --motef-consensus-stepsize .1 --motef-momentum-weight .1 --device cpu \
  --test-method snapshot --test-period 1 --work-dir /tmp/motef-vr-smoke
python -m pytest -q tests/test_motef.py tests/test_motef_vr.py
```

The VR tests independently differentiate logistic objectives on changing batches
across five iterations, three compressors, two graphs, optional global barriers,
lambda=.2/1 and heterogeneous speeds. They check every model, estimator, tracker,
compressed residual, estimate and weighted memory, both mean invariants, exact
wire bytes, doubled compute durations, charged bootstrap and causal dependencies.
Other checks cover transformed-batch reuse, Python/NumPy/dropout coupling, local
BatchNorm buffers, immutable inputs, RNG restoration, one/two-client graphs,
early/duplicate packets, initialization/completed snapshots, settings round trips,
experiment resolution, reproducibility, streaming, saved DAG replay and CUDA
agreement. The maintained suite additionally checks the shared trainer and base
MoTEF regressions.

Validation results and runtime measurements are recorded in
[validation/motef-vr.json](validation/motef-vr.json). Runtime pilots are deliberately
short and untuned. Full LR tuning, 900-second runs and leaderboard inclusion are
left for the main experiment chat.

On 4 November 2026 the full CPU suite passed **968 tests**, with 25 CUDA or
optional tests skipped, in 158 seconds. The final VR/registry/experiment subset
(including two subsequently added integration cases) passed **115 tests**, with
two CUDA skips. The MoTEF/VR numerical and coupling suite passed **100 tests**
on the second RTX 5000 Ada, including CPU/CUDA agreement. Public local execution,
experiment execution and public saved-DAG replay passed; replay metrics matched
exactly. See the record for the precise test scopes.

The final 32-client CIFAR-10/GroupNorm LeNet pilots used batch 32, ring mixing,
Top-k .01, gamma=lambda=.1, LR=.01, one thread, 1 MB/s nominal bandwidth,
20 ms mean gradient time and 10 ms latency. Global evaluation used the full
10,000-example test set at zero and the horizon. Three simulated seconds took
25.59 seconds of numerical execution with uniform capacities (1,024 completed
client updates), and 16.45 seconds with heterogeneous capacities (489 updates).
These include data startup and evaluation. D=85,354; each neighbor payload is
20,496 bytes before header/metadata. A separate 30-second **event-only** run
scheduled 10,477 and 4,735 paired-gradient tasks respectively; it did no training.

Extrapolating those rates and measured steady executor cost gives approximately
**65–85 minutes per 900-second uniform run** and **30–45 minutes per heterogeneous
run** on one RTX 5000 Ada. A four-scenario, three-seed matrix would take roughly
9–13 serial GPU hours before tuning. These ranges are workload-specific and do
not measure full-horizon DAG memory or sustained runtime. Different compression,
thread counts, evaluation protocols or GPU sharing can change them materially.
The user explicitly deferred full leaderboard evaluation; no inclusion is proposed.

Reproduce only a short runtime pilot with:

```sh
CUDA_VISIBLE_DEVICES=1 python scripts/pilot_motef_vr.py \
  --data /mnt/data/mdevos/conflux/data/cifar10 --device cuda:0 \
  --durations 1 3 --profiles uniform heterogeneous \
  --output /tmp/motef-vr-new-pilot
```

Check GPU availability/load first. The output directory must be fresh; the pilot
never edits an active leaderboard catalog. Numeric execution includes scheduled
in-flight tasks past the protocol horizon, matching the ordinary DAG executor.
