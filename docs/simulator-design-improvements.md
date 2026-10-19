# Simulator programming model

Implemented on 2026-10-19. Protocols use explicit task references, shared compute
and transport primitives, and optional numerical feedback. The maintained
algorithms and experiment runners use the new API. Saved workflows use task
format version 2; regenerate older DAGs. The old task-registration and send APIs
have been removed.

## Events and computation

Participants remain state machines reacting to completion events, messages and
timers. Protocols whose schedules do not depend on numerical results still build
a complete DAG before executing it. Adaptive protocols pause simulation at a
feedback completion, execute its dependency closure, then extend the DAG using
the observation. Simulated protocol time and executor wall time remain separate.

[`TaskRef`](../dasklearn/tasks/references.py) identifies one output of one task.
Ordinary tuples remain application data. References can appear inside nested
dictionaries, lists and tuples; the graph derives a unique dependency edge for
each referenced output. Multiple uses of an output resolve together. DAG
validation checks input/output consistency, output indices and cycles.

[`BaseClient.compute()`](../dasklearn/simulation/client.py) declares an operation,
reserves a simulated resource and schedules completion:

```python
trained = self.compute(
    'train', {'model': model}, params={'round': round_nr},
    duration=self.get_train_time(), on_complete='trained',
)
```

Durations are nonnegative integer microseconds. The default resource, `compute`,
is shared with training. Named resources allow explicitly independent work.
Operations wait for their input references and previous reservations on the same
resource. Zero duration is an explicit zero-cost assumption. `compute_time` is
charged centrally when operations complete.

A single-output operation returns a `TaskRef`; other operations return a tuple
of references, including an empty tuple for an operation with no outputs. A
completion handler receives an `Event[ComputeCompletion]` with `task`, `outputs`,
`duration`, `params`, `callback` and `feedback` fields. Numerical outputs are
exposed only when feedback is requested.

`create_compute_task(operation, data, name=None)` is the lower-level declaration
API. It derives dependencies and output contracts but schedules no completion
and reserves no resource. Existing algorithms use it for operations whose
modeled costs remain zero, such as their instantaneous aggregation and chunking
steps. Protocol code must schedule the appropriate continuation itself.

Register custom numerical operations using an importable implementation:

```python
from dasklearn.tasks.operations import register_operation
register_operation('my_update', 'my_package.operations:update', outputs=2)
```

Implementations accept `(settings, data)` and return a list or tuple with the
registered number of outputs. The import path and output contract travel with
the task so workers need no separate registry setup. The module must be installed
or importable on every worker. Executors protect predecessor values; custom
operations receive defensive copies. An implementation can opt into shared
inputs with `readonly_inputs=True` when it guarantees that it never mutates them.
Optimizer memory should be explicit task inputs and outputs, as in QGM and DeMo.

## Payloads and control messages

All maintained protocols use `send` for models, chunks, compressed packets and
other payloads:

```python
self.send(peer, payload=trained, size_bytes=model_bytes,
          metadata={'round': round_nr}, on_receive='on_incoming_model')
```

`size_bytes` means payload bytes. By default, the wire size adds
`control_message_bytes` plus the serialized metadata length once. Explicit
`overhead_bytes` replaces that default overhead calculation. The original
full-model protocols specify zero overhead to preserve their existing accounting
assumptions; compressed and chunked protocols retain their charged headers.
Mutable payloads and metadata are copied when the send is declared. Sends wait
until referenced outputs are available. The receiver gets `payload`, `metadata`,
`from`, `to` and `callback`; model handlers also receive `model` as the payload.

Self-delivery is a queued local event with zero wire bytes and latency. Network
delivery uses the shared bandwidth scheduler and configured latency. The
`immediate=True` option starts a ready network transfer within the current
handler; Conflux uses it to preserve its existing pull-response ordering.

`send_control(peer, message_type, message, delay=0)` uses the same transport.
Its `delay` is relative, while `send(..., send_time=...)` accepts an
absolute simulated timestamp. Control sends retain the original serialized
message size plus header and invoke the receiver's `on_message` handler.

## Participant context and protocol helpers

[`ParticipantContext`](../dasklearn/simulation/protocol.py) provides local time,
online status, a recursively frozen configuration snapshot, named random streams
and timers. `self.context.rng('selection')` gives a reproducible stream scoped to
the run seed, participant and stream name. Shared protocol randomness and memory
monitoring use separate simulator streams. Transfer identifiers do not consume
protocol randomness. Asynchronous peer choices and gossip phase offsets are now
isolated from instrumentation; historical seeded schedules may consequently
differ even though their sampling rules and timing costs are preserved.

`self.context.oracle(name)` requests an explicitly installed global service.
D-PSGD and Shatter install an availability oracle; D-PSGD also installs its
instantaneous offline-sentinel service. Conflux installs its bootstrap peer
service. These preserve the existing failure-awareness and discovery assumptions
without disguising them as local observations. Simulator topology and collective
coordination services remain available to protocol implementations; the context
is an API boundary, not a Python security sandbox.

`RoundInbox(expected_senders)` accepts the first message per sender, rejects
unexpected senders and guards completion with `finish(local_ready=...)`. Early
arrivals are valid. It imposes no global barrier or timeout policy. QGM and CHOCO
use it for their neighbor exchanges. Other protocols retain state machines whose
readiness rules are specific to their collective or round structure.

`self.set_timer(delay, 'handler', data)` returns a cancellable timer. Timers use
simulated microseconds and invoke participant methods with copied data. Equal-time
events follow their insertion order; accuracy selection in the example breaks
ties by participant index.

## Numerical feedback and replay

Request actual values with `feedback=True`:

```python
self.compute('evaluate', {'model': trained},
             params={'round': round_nr, 'evaluation_split': 'validation'},
             duration=evaluation_duration, feedback=True,
             on_complete='evaluated')

# In evaluated(event), event.data['values'][0] is the validation report.
```

`evaluate` returns accuracy, loss, peer, round, evaluation split and evaluation
start time. It requires validation data separate from the final test split. The
synthetic dataset supplies validation examples from previously unassigned data,
leaving existing client partitions and test examples unchanged. Model versions
are identified by the task's input references. Protocols decide when to evaluate,
which reports to send, and whether to wait for new reports or use stale ones.
The example adds evaluation completion and report receipt timestamps.

[`LocalExecutionSession`](../dasklearn/tasks/incremental.py) executes dependencies
on demand and retains runtime state across pauses. `simulate()` creates one
lazily when the first feedback result is required. `execute_local(simulation)`
then finishes the remaining tasks using that session instead of repeating work.
Dry runs through the coordinator require recorded feedback for adaptive decisions.

`simulate_async(RemoteExecutionSession(simulation))` obtains results from
persistent broker workers. `--feedback-execution distributed` selects this path
in the coordinator. The coordinator owns the retained values and dispatches one
ready operation per acknowledged RPC. Workers receive already resolved inputs.
Graph extension therefore does not trigger shutdown at temporary sinks; brokers
shut down only after the coordinator completes the run.

Both feedback executors conservatively pause global simulated-time advancement
while resolving an observation. Numerical completion does not advance simulated
time. Evaluation cost and report delivery determine when a participant can use a
metric, regardless of how quickly the executor computed it. Feedback sessions pin
all results until `close()` because future consumers are not known yet. This path
prioritizes correctness over memory bounds and parallel throughput; the existing
bounded and batched executor remains the default for complete static DAGs.

Adaptive runs save `feedback.pkl`, including observations, decision-point
signatures, settings, source hashes and an event-trace digest. Pass it to
`simulate(replay=...)`, `--feedback-replay`, or the example's `--replay` option.
Replay checks provenance and reproduces protocol decisions without numerical
execution during simulation. It is an observation replay, not a model checkpoint;
executing the resulting graph still performs its numerical work. Like saved DAGs,
journals use pickle and must come from trusted runs.

## Example and verification

The [accuracy-selection example](../dasklearn/simulation/accuracy_selection.py)
trains all participants, waits for their validation reports at node zero,
requests the best checkpoint, charges its transfer and trains it once more.
It is an API example rather than a paper baseline.

```sh
python scripts/run_feedback.py --output output/feedback-example
```

For distributed execution, start the example coordinator and two brokers in
separate terminals:

```sh
python scripts/run_feedback.py --distributed --brokers 2 --output output/feedback-brokers
python broker.py --coordinator tcp://127.0.0.1:5555 --port 5556 --workers 2
python broker.py --coordinator tcp://127.0.0.1:5555 --port 5557 --workers 2
```

`tests/test_programming_model.py` checks nested references, literal tuples,
multiple outputs, resource timing, transport accounting, timers, input ownership,
graph extension and rejection of the removed APIs. `tests/test_feedback_execution.py`
checks result-dependent choices, report visibility, wall-time independence,
executor state, replay and the broker RPC lifecycle. Existing mathematical,
causal and execution tests cover the migrated baselines. Run `python -m pytest -q`.

Validation on 2026-10-19 passed 257 tests on CPU, with 13 accelerator tests
skipped, and smoke execution covered all 15 registered protocol variants. Real
two-broker runs with four workers completed both adaptive accuracy selection and
static QGM execution. The integration guide's ring example executed four peers
for three rounds.

A synthetic comparison against pre-migration commit
`4f2d611cc60e72c109d0427c86c6b8214e52942c` used four peers, two rounds, one local
step, seed 3 and a two-second simulation budget. FL, D-PSGD, Epidemic, Conflux,
Shatter, Moshpit, Teleportation, DSpodFL, QGM, CHOCO and DeMo matched canonical
task graphs, final published model states, compute totals and wire bytes. Conflux
graph ordering was compared with a fixed Python hash seed. The asynchronous
protocols using newly isolated random streams are covered by current execution
and reproducibility tests rather than historical schedule equivalence.

Potential later work includes bounded retention for adaptive runs, parallel ready
operations, safe advancement of independent participants past feedback points,
and a coroutine interface over the same event semantics.
