# Decentralized AMSGrad

`damsgrad` implements **Algorithm 3** of [Chen, Karimi, Zhao and Li,
*On the Convergence of Decentralized Adaptive Gradient Methods*](https://proceedings.mlr.press/v189/chen23b.html)
(ACML 2022, PMLR publication 2023). It adds coordinate-wise adaptation and
dynamic average consensus on the adaptive state. This implementation follows
the published algorithm rather than running independent PyTorch AMSGrad
optimizers followed by model averaging.

## Update and initialization

Each client stores four parameter-sized vectors: first moment m, second moment
v, running maximum h (the paper's v-hat), and raw consensus tracker s (the
paper's half-step u-tilde). With a fixed symmetric doubly stochastic W:

```
g       = minibatch_gradient(x) + weight_decay * x
m_next  = beta1 * m + (1-beta1) * g
v_next  = beta2 * v + (1-beta2) * g**2
h_next  = maximum(h, v_next)
x_mix   = W @ x
s_mix   = W @ s
denom   = sqrt(maximum(s_mix, epsilon))
x_next  = x_mix - learning_rate * m_next / denom
s_next  = s_mix + (h_next - h)
```

All gradients and exchanged values use the **old** model/state. Only x and s
are exchanged; the first/second moments and running maximum remain local.
Mixing occurs before subtracting the local adaptive direction. The new h
increment enters s **after** the model update, so the denominator deliberately
lags the local running maximum by one iteration. This is the overlapping
Algorithm 3, not the alternative update order discussed below it in the paper.

There is **no Adam bias correction**, no additive epsilon outside the square
root, no AdamW decay, and no extra momentum from the shared SGD settings.
Epsilon is a floor in **squared-gradient units**. Defaults are beta1=.9,
beta2=.999 and epsilon=1e-6; the floor follows the paper's experiment setting,
while these moment decays are explicit implementation defaults. Learning
rate is the existing shared constant `--learning-rate` setting.

Initialize the shared seeded model with m0=v0=0 and h0=s0=epsilon at every
client. The paper specifies m0, h0 and the initial half-step tracker; it omits
v0 from the input line, for which we explicitly choose the usual zero second
moment. No gradient bootstrap or neighbor-state oracle is used. Initial
packets, including the known constant tracker, still consume wire bytes.

At a common completed logical iteration, `mean(s)=mean(h)`. W preserves the
model mean before the local directions are subtracted. The mean direction is
the mean of the locally preconditioned moments, not a centralized Adam step.
Local h values can remain different while s approaches their common network
mean. Snapshots under unequal client speeds can contain different logical
iterations and need not satisfy a common-iteration invariant across clients.

Even with one client this algorithm uses **the previous maximum** in the
denominator. It therefore differs from ordinary current-denominator AMSGrad;
the paper explicitly identifies this distinction. The first update uses
sqrt(epsilon), making its learning-rate/epsilon combination consequential.
Positive finite inputs do not guarantee stable tuning.

## Communication, timing and supported scope

The underlying graph is fixed and connected: ring, complete, or seeded
connected regular. Symmetric Metropolis weights give both model and adaptive
state the same W, with a self-loop. The ring has weights 1/3 including self
for at least three clients; one- and two-client graphs retain valid weights
without duplicate edges. Graph setup has zero simulated cost.

At iteration start each client sends one **fused dense packet** per neighbor:
its old model vector and old raw consensus tracker. Payload is exactly
`2*parameter_bytes`, plus the generic control header and serialized round
metadata. There are no sparse indices, gradient packets, moment packets, or
acknowledgements. The generic `model_size_bytes` override does not replace
this explicit tensor layout. Initial/zero vectors are charged in full.

Those transfers overlap one charged gradient computation at the old local
model. Completion requires the local gradient and every same-iteration
neighbor packet. Early packets are buffered and duplicate messages or
completion callbacks cannot apply an update twice. Default execution has
neighbor dependencies; `--synchronous` additionally adds a global start
barrier per iteration. Unequal client speeds affect when operations occur,
without changing their same-iteration inputs. No stale-message, directed or
time-varying extension is claimed.

All traffic uses the shared upload/download contention and latency scheduler.
Vector arithmetic, moment updates, mixing and packet packing have zero
simulated compute cost; each gradient has a positive simulated duration.
Actual numerical CPU/CUDA execution time does not feed back into protocol
time. Only completed model updates are published for snapshots. Saved
`damsgrad-mixing.json` records W, hyperparameters, initialization and payload
size; `damsgrad-rounds.csv` records start, gradient-ready and completion times.

Supported settings are stable membership, `local_steps=1`, shared momentum
zero, individual or snapshot evaluation, and fully trainable nonempty floating
parameters with a common dtype. Epsilon must remain positive and finite in
that dtype. Model buffers stay local and follow the local gradient forward
pass. LoRA, frozen/mixed-dtype layouts, churn, local-SGD blocks, shared
momentum, compression, learning-rate schedules and the alternative
current-denominator update are excluded and unsupported settings fail early
where exposed by the baseline CLI.

## Usage

```sh
python main.py --algorithm damsgrad --local --dataset digits --model mlp \
  --peers 8 --local-steps 1 --momentum 0 --learning-rate .001 \
  --damsgrad-topology ring --damsgrad-beta1 .9 --damsgrad-beta2 .999 \
  --damsgrad-epsilon 1e-6
```

The same options are registered in the synthetic/digits, CIFAR and Tiny
ImageNet comparison runners. CIFAR/Tiny execution and performance are not
validated in this addition.

## Validation

[Independent tests](../tests/test_damsgrad.py) compare complete logistic
model/moment/maximum/tracker/packet trajectories against a NumPy matrix
recurrence and separately computed autograd gradients. They cover both ring
and complete graphs, zero and nonzero moment decays, skewed capacities and
optional global barriers. Additional checks cover raw tracker conservation
and convergence with unequal local second moments, the single-client lag,
floor placement and lack of bias correction, exact two-vector payload bytes,
overlap and readiness, early/duplicate messages, local buffers and immutable
inputs, small/regular graphs, duration stopping, streaming snapshots, seeded
serialized replay, CPU/CUDA agreement and CLI/broker settings registration.

The README label is **Numerical · Protocol · Integration**. The primary paper
and supplement are pinned by PDF SHA-256 in the [validation record](validation/damsgrad.json).
No authors' implementation comparison was executed, so **Reference** is not
claimed. A local CLI CUDA smoke run exercises the main entry point; a live
multi-process Decentralized AMSGrad deployment remains unvalidated. This is
not a reproduction of the paper's MNIST CNN experiments or tuning procedure.

The `conflux-gpu` full suite passed **838 tests** (five existing warnings,
121.20 s), including **54 Decentralized AMSGrad cases**. The main CLI also
completed four synthetic CUDA rounds on four clients with finite final
snapshot metrics.

## Digits screening

The frozen [per-seed summary](validation/damsgrad-digits-summary.csv) and
[curves](validation/damsgrad-digits-curves.csv) contain **126 CUDA runs**, of
which 42 use Decentralized AMSGrad. All runs use eight clients, a 2410-parameter
float32 MLP, three simulated seconds, one-second snapshots, batch size 32,
one local step, no shared momentum/decay, and seeds 0/1/2. AMSGrad uses the
declared defaults beta1=.9, beta2=.999 and epsilon=1e-6 throughout.

At 100 MB/s, all three methods use the same fixed learning-rate grid
**.001/.01/.05**, both IID and Dirichlet α=.3 partitions, and uniform or
heterogeneous capacities. The .05 grid point includes the established SGD
scale from earlier screening. Every grid point is reported; there was no
held-out rate selection or best-rate-only comparison. These methods have
different useful learning-rate scales, so a single shared rate is insufficient
for an overall performance claim.

Final mean local test accuracy (%), mean ± sample standard deviation at
100 MB/s:

| Profile | Data | Learning rate | Decentralized AMSGrad | DSGD | GT-DSGD |
| --- | --- | ---: | ---: | ---: | ---: |
| Uniform | IID | .001 | 95.76 ± .78 | 9.27 ± 1.22 | 9.21 ± 1.37 |
| Uniform | IID | .01 | 97.24 ± 1.30 | 39.10 ± 5.88 | 31.79 ± 3.86 |
| Uniform | IID | .05 | 30.43 ± 35.39 | 87.00 ± 3.57 | 83.60 ± 4.78 |
| Uniform | Dirichlet | .001 | 92.75 ± 1.32 | 8.73 ± 2.30 | 8.61 ± 2.40 |
| Uniform | Dirichlet | .01 | 95.29 ± 1.65 | 25.82 ± 11.66 | 23.55 ± 10.68 |
| Uniform | Dirichlet | .05 | 16.56 ± 11.44 | 79.29 ± 2.12 | 77.56 ± 5.37 |
| Heterogeneous | IID | .001 | 82.43 ± 11.37 | 8.49 ± 1.94 | 8.46 ± 1.96 |
| Heterogeneous | IID | .01 | 94.61 ± 2.80 | 10.70 ± .90 | 10.56 ± .78 |
| Heterogeneous | IID | .05 | 15.62 ± 9.68 | 33.06 ± 18.00 | 30.51 ± 16.57 |
| Heterogeneous | Dirichlet | .001 | 48.93 ± 25.93 | 8.43 ± 2.27 | 8.46 ± 2.30 |
| Heterogeneous | Dirichlet | .01 | 87.48 ± 6.19 | 11.49 ± 2.15 | 10.89 ± 1.35 |
| Heterogeneous | Dirichlet | .05 | 13.94 ± 6.85 | 22.98 ± 12.29 | 19.85 ± 13.17 |

AMSGrad learns quickly at .001/.01 on this small task, while many .05 runs
remain near chance. SGD controls improve markedly at .05. The screening
therefore shows both the value of adaptation here and its sensitivity to
tuning, without establishing a larger-model or general algorithm ranking.
Test curves were inspected to report the screening; no model or rate was
selected using those curves.

The remaining 18 runs use **200 kB/s**, Dirichlet data and a fixed shared rate
.01. Final accuracy is 46.61 ± 2.71% / 30.52 ± 3.74% for AMSGrad under uniform /
heterogeneous capacity, 8.81 ± 2.13% / 8.43 ± 2.36% for DSGD, and
8.43 ± 2.36% / 8.35 ± 2.49% for GT-DSGD. Other congested rates were not swept,
so these are fixed-rate mechanism checks rather than tuned comparisons.

Uniform ample-capacity traffic is **103.51 MB** for AMSGrad, **41.83 MB** for
DSGD and **69.87 MB** for GT-DSGD, with 2664, 2136 and 1790 completed local
updates respectively. Under uniform congestion those counts become 96, 190
and 96 updates, with 3.93, 3.75 and 3.85 MB sent. Counts exclude unfinished
model updates; traffic includes partial transfers and packets for work in
progress. Scheduled gradients are recorded separately in the per-seed CSV.

AMSGrad sends a fused **19280-byte** payload (two 9640-byte vectors) per
neighbor, plus header/round metadata. DSGD sends one dense vector after its
gradient; GT-DSGD sends two sequential dense exchanges; AMSGrad sends old
states before the gradient and overlaps the exchange. Different ordering,
preconditioning, exchange counts and overlap mean this is not an isolated
ablation of adaptive learning rates.

Latency is 2 ms. Uniform gradients take 9 ms. Heterogeneous gradient times are
drawn from 6/15/36/90 ms independently of bandwidth ranking; a quarter of
clients get 4× nominal bandwidth and the remainder .5×. Capacities are paired
across methods and rates within profile/partition/seed. Profile means differ.
Actual scalar CUDA execution times overlap other validation work and do not
determine simulated protocol time.

Reproduce one ample-bandwidth grid point with:

```sh
python scripts/validate_algorithms.py --variants damsgrad dsgd gt-dsgd --dataset digits \
  --peers 8 --local-steps 1 --duration 3 --test-period 1 --bandwidth 100000000 \
  --latency-ms 2 --learning-rate .001 --profiles uniform heterogeneous \
  --partitioners iid dirichlet --seeds 0 1 2 --device cuda:0 \
  --output output/damsgrad-validation/digits-lr001
```

Repeat with rates .01 and .05 in separate output directories. For congestion
use rate .01, `--bandwidth 200000 --partitioners dirichlet`, and a separate
directory. The validation record preserves all four exact commands,
manifests, paired capacities and source hashes. Audits verify finite metrics,
all client snapshots against completed-update histories, balanced sent/received
bytes, symmetric doubly stochastic W, explicit payloads and charged causal
timing for every run.
