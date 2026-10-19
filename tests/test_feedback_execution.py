import asyncio
import copy
import pickle
import time
from types import SimpleNamespace

import pytest
import torch

from dasklearn import functions
from dasklearn.broker import Broker
from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.accuracy_selection import AccuracySelectionSimulation
from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.simulation import Simulation
from dasklearn.tasks.incremental import LocalExecutionSession, RemoteExecutionSession
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.operations import execute_operation, register_operation
from helpers import settings


def score(delay=0, winner=1):
    def evaluate(settings, data):
        time.sleep(delay)
        return [{'accuracy': 90 if data['peer'] == winner else 10,
                 'loss': .1, 'peer': data['peer'], 'round': data['round'],
                 'evaluation_split': 'validation', 'evaluated_at': data['time']}]
    return evaluate


def configuration(tmp_path):
    return settings(SessionSettings, tmp_path, algorithm='feedback-example', participants=3,
                    test_method='individual', test_period=0, duration=10_000_000,
                    model_size_bytes=4096, network_latency_ms=10)


def simulate(tmp_path):
    simulation = AccuracySelectionSimulation(configuration(tmp_path))
    simulation.setup_directories()
    simulation.simulate()
    return simulation


@pytest.mark.asyncio
async def test_feedback_changes_selection_and_never_uses_undelivered_reports(tmp_path, monkeypatch):
    traces = []
    for repeat, (delay, winner) in enumerate([(0, 1), (.01, 1), (0, 2)]):
        monkeypatch.setattr(functions, 'evaluate', score(delay, winner))
        simulation = simulate(tmp_path / str(repeat))
        selector = simulation.clients[0]
        assert selector.selected_peer == winner
        assert selector.selected_at == max(r['received_at'] for r in selector.reports.messages.values())
        assert all(r['completed_at'] <= r['received_at'] for r in selector.reports.messages.values())
        assert selector.received_at > selector.selected_at + 20_000
        # Only the chosen participant transmits a full checkpoint.
        assert simulation.clients[winner].bw_scheduler.total_bytes_sent > simulation.model_size
        assert simulation.clients[3-winner].bw_scheduler.total_bytes_sent < simulation.model_size
        traces.append(simulation.event_trace_hash.hexdigest())
    assert traces[0] == traces[1]
    assert traces[2] != traces[0]


@pytest.mark.asyncio
async def test_feedback_and_full_dag_execution_produce_the_same_model(tmp_path, monkeypatch):
    monkeypatch.setattr(functions, 'evaluate', score())
    simulation = simulate(tmp_path / 'incremental')
    actual = execute_local(simulation)
    static = Simulation(configuration(tmp_path / 'static'))
    static.setup_directories()
    static.workflow_dag = copy.deepcopy(simulation.workflow_dag)
    expected = execute_local(static)
    assert len(actual) == len(expected) == len(simulation.workflow_dag.tasks)
    for name in actual:
        for left, right in zip(actual[name], expected[name]):
            if isinstance(left, torch.nn.Module):
                for key, value in left.state_dict().items():
                    torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)


@pytest.mark.asyncio
async def test_feedback_replay_needs_no_numerical_execution_and_checks_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(functions, 'evaluate', score())
    original = simulate(tmp_path / 'original')
    journal = original.data_dir + '/feedback.pkl'
    def unexpected(*args):
        raise AssertionError('Replay should not execute numerical work during simulation')
    monkeypatch.setattr(functions, 'evaluate', unexpected)
    replay = AccuracySelectionSimulation(configuration(tmp_path / 'replay'))
    replay.setup_directories()
    replay.simulate(replay=journal, allow_feedback=False)
    assert replay.execution_session is None
    assert replay.clients[0].selected_peer == original.clients[0].selected_peer
    assert replay.event_trace_hash.hexdigest() == original.event_trace_hash.hexdigest()
    changed = configuration(tmp_path / 'changed')
    changed.seed += 1
    with pytest.raises(ValueError, match='source or settings'):
        AccuracySelectionSimulation(changed).simulate(replay=journal)
    dry = AccuracySelectionSimulation(configuration(tmp_path / 'dry'))
    dry.setup_directories()
    with pytest.raises(ValueError, match='dry run'):
        dry.simulate(allow_feedback=False)


def stateful(settings, data):
    if functions.model_trainers is None:
        functions.model_trainers = {'count': 0}
    functions.model_trainers['count'] += 1
    return [functions.model_trainers['count']]


register_operation('api_stateful', f'{__name__}:stateful', 1)


class StatefulClient(BaseClient):
    def init_client(self, event):
        self.observations = []
        self.compute('api_stateful', {}, duration=5, feedback=True, on_complete='observed')

    def observed(self, event):
        self.observations.append((event.time, event.data['values'][0]))
        if len(self.observations) < 2:
            self.compute('api_stateful', {}, duration=5, feedback=True, on_complete='observed')


class StatefulSimulation(Simulation):
    CLIENT_CLASS = StatefulClient


@pytest.mark.asyncio
async def test_feedback_pauses_preserve_private_executor_state(tmp_path, monkeypatch):
    outer_state = object()
    monkeypatch.setattr(functions, 'model_trainers', outer_state)
    simulation = StatefulSimulation(settings(SessionSettings, tmp_path, participants=1, test_period=0))
    simulation.setup_directories()
    simulation.simulate()
    assert simulation.clients[0].observations == [(5, 1), (10, 2)]
    assert functions.model_trainers is outer_state


class BrokerWire:
    """Exercise real broker configuration, queues and result handling in process."""
    def __init__(self, simulation):
        self.simulation = simulation
        self.messages = []
        self.readers = []
        self.brokers = {}
        for index in range(2):
            broker = Broker(SimpleNamespace(workers=1, profile=False))
            broker.identity = f'broker_{index}'
            broker.communication = SimpleNamespace(
                connect_to=lambda *args: None,
                send_message_to_coordinator=self.coordinator_result)
            self.brokers[broker.identity] = broker
            async def start_workers(broker=broker):
                broker.workers_ready = True
                broker.worker_queue = SimpleNamespace(put=lambda item: self.execute(broker, item))
                for task in broker.pending_tasks:
                    broker.put_in_worker_queue(task)
                broker.pending_tasks = []
            broker.start_workers = start_workers
            self.readers.append(asyncio.create_task(broker.worker_result_queue_task()))
        simulation.broker_addresses = {name: name for name in self.brokers}
        simulation.brokers_available_future.set_result(True)

    def execute(self, broker, item):
        name, function, data, contract = item
        result = execute_operation(function, broker.settings, data, contract)
        now = time.time()
        broker.worker_result_queue.put_nowait((name, result, {'received': now, 'finished': now, 'worker': 0}))

    def coordinator_result(self, message):
        message = pickle.loads(message)
        self.messages.append(message)
        self.simulation.on_message('broker', message)

    def send_message_to_all_brokers(self, message):
        for broker in self.brokers.values():
            broker.on_message('coordinator', pickle.loads(message))

    def send_message_to_broker(self, identity, message):
        self.brokers[identity].on_message('coordinator', pickle.loads(message))

    async def close(self):
        for reader in self.readers:
            reader.cancel()
        await asyncio.gather(*self.readers, return_exceptions=True)


@pytest.mark.asyncio
async def test_distributed_feedback_extends_past_temporary_sinks_and_matches_local(tmp_path, monkeypatch):
    monkeypatch.setattr(functions, 'evaluate', score())
    # Reference session isolates function globals as actual separate worker
    # processes do. The in-process wire needs its own isolated runtime too.
    local = simulate(tmp_path / 'local')
    expected = execute_local(local)
    remote = AccuracySelectionSimulation(configuration(tmp_path / 'remote'))
    remote.setup_directories()
    session = RemoteExecutionSession(remote)
    wire = BrokerWire(remote)
    remote.communication = wire
    worker_runtime = LocalExecutionSession(remote)
    original_execute = wire.execute
    def isolated_execute(broker, item):
        with worker_runtime.runtime():
            original_execute(broker, item)
    wire.execute = isolated_execute
    try:
        await asyncio.wait_for(remote.simulate_async(session), timeout=15)
        await asyncio.wait_for(session.execute_all(), timeout=15)
        assert remote.clients[0].selected_peer == local.clients[0].selected_peer
        assert remote.event_trace_hash.hexdigest() == local.event_trace_hash.hexdigest()
        assert len(session.results) == len(expected)
        assert all(message['type'] == 'execution_result' for message in wire.messages)
        assert sum(b.completed_tasks for b in wire.brokers.values()) == len(expected)
        final = remote.clients[0].model
        for key, value in session.results[final.task][0].state_dict().items():
            torch.testing.assert_close(value, expected[local.clients[0].model.task][0].state_dict()[key], rtol=0, atol=0)
    finally:
        await wire.close()


@pytest.mark.asyncio
async def test_remote_failure_reaches_waiting_coordinator(tmp_path):
    simulation = Simulation(configuration(tmp_path))
    session = RemoteExecutionSession(simulation)
    future = asyncio.get_running_loop().create_future()
    session.pending['broken'] = future
    session.on_result({'task': 'broken', 'error': 'worker failed'})
    with pytest.raises(RuntimeError, match='worker failed'):
        await future


def no_outputs(settings, data):
    functions.model_trainers = {'effect_executed': True}
    return []


register_operation('api_no_outputs', f'{__name__}:no_outputs', 0)


class EffectClient(BaseClient):
    def init_client(self, event):
        self.compute('api_no_outputs', {}, duration=1, feedback=True, on_complete='observed')

    def observed(self, event):
        self.observation = (event.time, event.data['values'])


class EffectSimulation(Simulation):
    CLIENT_CLASS = EffectClient


@pytest.mark.asyncio
async def test_feedback_executes_operations_with_no_outputs(tmp_path):
    simulation = EffectSimulation(settings(SessionSettings, tmp_path, participants=1, test_period=0))
    simulation.setup_directories()
    simulation.simulate()
    assert simulation.clients[0].observation == (1, ())
    assert simulation.execution_session.state['model_trainers'] == {'effect_executed': True}
