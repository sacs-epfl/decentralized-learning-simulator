import copy
import pickle
from types import SimpleNamespace
from pathlib import Path

import pytest

from dasklearn import functions
from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, COMPUTE_DONE
from dasklearn.simulation.protocol import RoundInbox
from dasklearn.simulation.simulation import Simulation
from dasklearn.tasks.dag import WorkflowDAG
from dasklearn.tasks.incremental import LocalExecutionSession
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.operations import register_operation
from dasklearn.tasks.references import TaskRef
from dasklearn.tasks.task import Task
from helpers import settings
from dasklearn.algorithms import ALGORITHMS, class_paths, import_class
from dasklearn.session_settings import LearningSettings


def pair(settings, data):
    return [[data['value']], data['value'] + 1]


def consume(settings, data):
    assert data['literal'] == ('producer', 0)
    data['nested'][0][0].append(99)
    return [data['nested'][1][0] + data['nested'][1][1]]


register_operation('api_pair', f'{__name__}:pair', 2)
register_operation('api_consume', f'{__name__}:consume', 1)


class Client(BaseClient):
    def __init__(self, *args):
        super().__init__(*args)
        self.deliveries = []
        self.completions = []

    def completed(self, event):
        self.completions.append((event.time, event.data['outputs']))

    def received(self, event):
        self.deliveries.append((event.time, event.data['from'], event.data['payload']))


def setup(tmp_path, **kwargs):
    sim = Simulation(settings(SessionSettings, tmp_path, participants=2, test_period=0, **kwargs))
    sim.clients = [Client(sim, i) for i in range(2)]
    return sim


def drain(sim):
    while sim.events:
        _, _, event = sim.events.pop(0)
        sim.current_time = event.time
        sim.process_event(event)


@pytest.mark.asyncio
async def test_nested_references_multiple_outputs_literals_and_roundtrip(tmp_path):
    sim = setup(tmp_path)
    a, b = sim.clients[0].compute('api_pair', {}, params={'value': 4}, duration=10)
    task = sim.clients[1].create_compute_task('api_consume', {
        'nested': ((a,), [b, b]), 'literal': ('producer', 0)}, name='consumer')
    assert len(task.inputs) == 2
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    assert task.inputs_resolve == 0
    saved = pickle.loads(pickle.dumps(sim.workflow_dag.serialize()))
    sim.workflow_dag = WorkflowDAG.unserialize(saved)
    result = execute_local(sim)
    assert result['consumer'] == [10]
    assert result[a.task][0] == [4]
    # The distributed resolver follows paths into tuples, resolves duplicates
    # once, and leaves ordinary tuple literals untouched.
    task = sim.workflow_dag.tasks['consumer']
    task.build_index(task.data)
    task.set_data(a, [4])
    task.set_data(a, [4])
    assert task.inputs_resolve == 1
    task.set_data(b, 5)
    assert task.has_all_inputs()
    assert task.data['nested'] == (([4],), [5, 5])
    assert task.data['literal'] == ('producer', 0)


@pytest.mark.asyncio
async def test_compute_resources_dependency_availability_and_send_snapshot(tmp_path):
    sim = setup(tmp_path, network_latency_ms=1)
    first, _ = sim.clients[0].compute('api_pair', {}, params={'value': 3}, duration=10,
                                     on_complete='completed')
    sim.clients[0].compute('api_pair', {}, params={'value': 5}, duration=4, on_complete='completed')
    sim.clients[0].compute('api_pair', {}, params={'value': 6}, duration=2,
                           resource='evaluation', on_complete='completed')
    metadata = {'phase': [1]}
    sim.clients[0].send(1, {'value': first}, size_bytes=100, metadata=metadata, on_receive='received')
    metadata['phase'].append(2)
    drain(sim)
    assert [time for time, _ in sim.clients[0].completions] == [2, 10, 14]
    assert sim.clients[0].compute_time == 16
    expected_bytes = 100 + sim.settings.control_message_bytes + len(pickle.dumps({'phase': [1]}))
    assert sim.clients[0].bw_scheduler.total_bytes_sent == expected_bytes
    assert sim.clients[1].bw_scheduler.total_bytes_received == expected_bytes
    assert sim.clients[1].deliveries == [(10 + expected_bytes + 1000, 0, {'value': first})]
    # An operation on another participant cannot consume a future value early.
    sim.current_time = 0
    out = sim.clients[1].compute('api_consume', {'nested': ((first,), [first, first])},
                                params={'literal': ('producer', 0)}, duration=3)
    assert sim.workflow_dag.tasks[out.task].available_at == 13


@pytest.mark.asyncio
async def test_local_delivery_control_freezing_timers_and_scoped_randomness(tmp_path):
    sim = setup(tmp_path)
    client = sim.clients[0]
    payload = {'values': [1]}
    client.send(0, payload, size_bytes=100, on_receive='received')
    payload['values'].append(2)
    cancelled = client.set_timer(3, 'received', {'from': 1, 'payload': 'cancelled'})
    cancelled.cancel()
    client.set_timer(4, 'received', {'from': 1, 'payload': 'timer'})
    drain(sim)
    assert client.deliveries == [(0, 0, {'values': [1]}), (4, 1, 'timer')]
    assert client.bw_scheduler.total_bytes_sent == 0
    other = setup(tmp_path / 'other').clients[0]
    assert client.context.rng().random() == other.context.rng().random()
    client.context.rng('instrumentation').random()
    assert client.context.rng().random() == other.context.rng().random()
    with pytest.raises(TypeError):
        client.context.config['learning']['local_steps'] = 100
    with pytest.raises(ValueError, match='not configured'):
        client.context.oracle('availability')


def test_round_inbox_early_duplicates_and_completion_guard():
    inbox = RoundInbox([2, 1])
    assert inbox.add(2, 'early')
    assert not inbox.add(2, 'duplicate')
    assert not inbox.finish()
    assert inbox.add(1, 'later')
    assert not inbox.finish(local_ready=False)
    assert inbox.finish()
    assert not inbox.finish()
    assert inbox.messages[2] == 'early'
    with pytest.raises(ValueError):
        inbox.add(9, 'unexpected')


@pytest.mark.asyncio
async def test_incremental_extension_keeps_previous_sinks_and_input_values(tmp_path):
    sim = setup(tmp_path)
    source = sim.clients[0].create_compute_task('api_pair', {'value': 4}, name='producer')
    session = LocalExecutionSession(sim)
    ref = TaskRef(source.name, 0)
    assert session.resolve(ref) == [4]
    data = {'nested': ((ref,), [TaskRef(source.name, 1)] * 2), 'literal': ('producer', 0)}
    consumer = sim.clients[1].create_compute_task('api_consume', data)
    assert session.resolve(TaskRef(consumer.name, 0)) == 10
    assert session.resolve(ref) == [4]
    session.close()
    with pytest.raises(RuntimeError, match='closed'):
        session.resolve(ref)


@pytest.mark.asyncio
async def test_invalid_contracts_references_and_old_apis_are_rejected(tmp_path):
    sim = setup(tmp_path)
    client = sim.clients[0]
    with pytest.raises(ValueError, match='Unknown .*reference'):
        client.create_compute_task('train', {'model': TaskRef('missing')})
    source = client.create_compute_task('initialize', {})
    with pytest.raises(ValueError, match='Invalid output'):
        client.create_compute_task('train', {'model': TaskRef(source.name, 3)})
    for duration in (-1, 1.5, True):
        with pytest.raises(ValueError):
            client.compute('initialize', {}, duration=duration)
    with pytest.raises(ValueError, match='Register operation'):
        client.compute('unregistered', {}, duration=0)
    for name in ('send_model', 'send_message_to_client', 'add_compute_task'):
        assert not hasattr(client, name)
    with pytest.raises(ValueError, match='regenerate'):
        Task.from_json_dict({'name': 'old', 'func': 'train', 'data': {}})


@pytest.mark.asyncio
@pytest.mark.parametrize('algorithm', list(ALGORITHMS))
async def test_every_registered_protocol_executes_with_new_api(tmp_path, algorithm):
    config_type, simulation_type = map(import_class, class_paths(algorithm))
    options = {
        'fl': dict(sample_size=2), 'dpsgd': dict(k=2), 'epidemic': dict(k=2),
        'conflux': dict(sample_size=2, chunks_in_sample=2),
        'shatter': dict(k=2, r=2), 'moshpit': dict(group_size=2),
        'teleportation': dict(sample_size=2), 'dspodfl': dict(degree=2),
        'gossip': dict(gl_period=200_000), 'super-gossip': dict(k=2), 'lubor': dict(k=2),
    }.get(algorithm, {})
    config = settings(config_type, tmp_path, algorithm=algorithm, participants=4,
                      rounds=2, duration=2_000_000, test_period=0, test_method='individual',
                      learning=LearningSettings(.1, 0, 0, 16, 1), **options)
    simulation = simulation_type(config)
    simulation.setup_directories()
    simulation.simulate()
    simulation.workflow_dag.check_validity()
    assert simulation.workflow_dag.tasks
    assert all(task.operation is not None for task in simulation.workflow_dag.tasks.values())
    results = execute_local(simulation)
    assert len(results) == len(simulation.workflow_dag.tasks)


def test_dag_validation_detects_cycles_and_missing_edges():
    dag = WorkflowDAG()
    a = Task('a', 'dummy', {'input': TaskRef('b')})
    b = Task('b', 'dummy', {'input': TaskRef('a')})
    a.inputs = [(b, 0)]
    b.outputs = [(a, 0)]
    b.inputs = [(a, 0)]
    a.outputs = [(b, 0)]
    dag.tasks = {'a': a, 'b': b}
    with pytest.raises(RuntimeError, match='cycle'):
        dag.check_validity()
    b.outputs = []
    with pytest.raises(RuntimeError, match='reverse dependency'):
        dag.check_validity()


@pytest.mark.asyncio
async def test_disabled_evaluation_can_finish_without_plotting_empty_data(tmp_path):
    simulation = setup(tmp_path)
    simulation.setup_directories()
    simulation.merge_accuracies_files()
    simulation.plot_loss()
    assert not Path(simulation.data_dir, 'accuracies.png').exists()
