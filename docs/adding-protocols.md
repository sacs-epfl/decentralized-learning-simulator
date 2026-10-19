# Adding a protocol

A protocol describes participant state and reactions to events. Numerical
operations become DAG tasks; the simulator models their completion and network
delivery. Start with [`BaseClient`](../dasklearn/simulation/client.py) and
[`Simulation`](../dasklearn/simulation/simulation.py). The
[programming model](simulator-design-improvements.md) documents the full API and
[accuracy-selection example](../dasklearn/simulation/accuracy_selection.py) shows
a protocol whose decisions depend on real validation results.

## Implement participant behavior

Create `dasklearn/simulation/my_protocol/` with `__init__.py`, `client.py`,
`simulation.py` and `settings.py`. Use `compute` for timed operations and `send`
for payloads. Callback names are participant method names; these callbacks do
not need separate event registration.

This minimal client trains, sends to its successor in a ring, waits for its
predecessor and averages before the next round:

```python
from collections import defaultdict
from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.protocol import RoundInbox
from dasklearn.tasks.references import TaskRef


class MyProtocolClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.predecessor = (index - 1) % self.context.config['participants']
        self.successor = (index + 1) % self.context.config['participants']
        self.rounds = defaultdict(lambda: {
            'local': None, 'inbox': RoundInbox([self.predecessor]),
        })

    def init_client(self, event):
        model = self.compute('initialize', {}, duration=0)
        self.train_round(1, model)

    def train_round(self, round_nr, model):
        self.compute('train', {'model': model}, params={'round': round_nr},
                     duration=self.get_train_time(), on_complete='trained')

    def trained(self, event):
        round_nr = event.data['params']['round']
        model = event.data['outputs'][0]
        self.rounds[round_nr]['local'] = model
        self.send(self.successor, model, size_bytes=self.simulator.model_size,
                  metadata={'round': round_nr}, on_receive='received')
        self.try_finish(round_nr)

    def received(self, event):
        round_nr = event.data['metadata']['round']
        self.rounds[round_nr]['inbox'].add(event.data['from'], event.data['payload'])
        self.try_finish(round_nr)

    def try_finish(self, round_nr):
        state = self.rounds[round_nr]
        if not state['inbox'].finish(local_ready=state['local'] is not None):
            return
        # Explicit assumption: aggregation takes zero simulated time.
        task = self.create_compute_task('aggregate', {
            'models': [state['local'], state['inbox'].messages[self.predecessor]],
            'peer': self.index, 'round': round_nr, 'time': self.context.now,
        })
        model = TaskRef(task.name, 0)
        self.publish_model(round_nr, model)
        if round_nr < self.context.config['rounds']:
            self.train_round(round_nr + 1, model)
```

Keep the first version simple: require at least two participants, positive
rounds and local steps, and stable membership. Add churn, retries or timeouts
only with explicit protocol rules. An inbox accepts early messages and guards
against duplicates; it does not decide what to do when a peer never responds.

## Register settings and the simulation

In `settings.py`, subclass the shared settings and add only protocol-specific
options:

```python
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dasklearn.session_settings import SessionSettings

@dataclass_json
@dataclass
class MyProtocolSettings(SessionSettings):
    pass
```

In `simulation.py`, select the client and reject unsupported configurations:

```python
from dasklearn.simulation.simulation import Simulation
from .client import MyProtocolClient

class MyProtocolSimulation(Simulation):
    CLIENT_CLASS = MyProtocolClient

    def __init__(self, settings):
        if settings.participants < 2 or settings.rounds < 1 or settings.learning.local_steps < 1:
            raise ValueError('Require at least two peers, positive rounds and local_steps')
        if settings.churn != 'none':
            raise ValueError('This protocol requires stable membership')
        super().__init__(settings)
```

Add this entry to [`ALGORITHMS`](../dasklearn/algorithms.py):

```python
'my-protocol': ('my_protocol', 'MyProtocolSettings', 'MyProtocolSimulation'),
```

That registry supports coordinator discovery and broker settings restoration.
New CLI options go in [`args.py`](../args.py) and their settings mapping in
[`main.py`](../main.py). To include the protocol in comparison runners, also add
its alias and configuration to
[`peer_baselines.py`](../dasklearn/simulation/peer_baselines.py) and the relevant
runner. The shared `SessionSettings` fields need no new CLI mapping.

## Add numerical operations or runtime feedback when needed

Built-ins cover initialization, training, aggregation, chunking and evaluation.
For a new update rule, put its numerical implementation in an importable module
and register it with `register_operation(name, 'module:function', outputs=N)`.
It receives `(settings, data)` and returns a list or tuple of exactly `N` outputs.
Ensure the module is available on broker workers. References in the operation's
inputs supply its dependency edges automatically. Ordinary tuples are data;
use `TaskRef` for dependencies. Carry persistent optimizer state as explicit
inputs and outputs rather than hiding it in protocol callbacks.

Protocol handlers receive references, not tensors. To choose a peer using actual
accuracy, schedule `compute('evaluate', ..., feedback=True, on_complete='evaluated')`
with a validation split and a modeled evaluation duration. Read
`event.data['values'][0]` in the callback, then send the resulting report through
the network. Select using received reports; do not inspect another participant's
state or read final test accuracy. `simulate()` resolves feedback locally;
`simulate_async(RemoteExecutionSession(simulation))` uses brokers.

Use `self.context.rng('selection')` for reproducible participant randomness,
`set_timer(delay, 'handler', data)` for simulated timers, and explicitly installed
oracle services for any intended global information. All durations are integer
microseconds. `send` charges payload bytes, one header and serialized metadata by
default; self-delivery is free. Choose and document any different overhead or
zero-cost compute assumptions.

## Verify the implementation

Use synthetic data first. After registering the example above:

```sh
python main.py --algorithm my-protocol --dataset synthetic --model mlp \
  --peers 4 --rounds 3 --local-steps 1 --momentum 0 \
  --test-method snapshot --test-period 1 --duration 10 --dry-run
python -m pytest -q
```

For an execution check in an asynchronous test, construct the settings and
simulation, call `setup_directories()` and `simulate()`, validate the DAG with
`workflow_dag.check_validity()`, then call `execute_local(simulation)`. See
[`tests/helpers.py`](../tests/helpers.py) for reusable setup. A dry run builds a
static graph; an adaptive coordinator dry run needs recorded feedback.

Check the update equations against a small independent numerical reference,
message and compute completion ordering, wire bytes, duplicates and early
arrivals, input immutability, state ownership and seeded reproducibility. For
adaptive protocols, also check report delivery times, wall-time independence and
replay. Add a short protocol contract in `docs/` explaining assumptions,
unsupported settings and deviations from any source algorithm.
