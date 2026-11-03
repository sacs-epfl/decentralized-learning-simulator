"""Independent gradients, causal mailbox selection, and wait-free timing."""
import copy
import json
import pickle
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn.optimizer.swift import update, pack
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.events import Event, INCOMING_MODEL
from dasklearn.simulation.swift.settings import SwiftSettings
from dasklearn.simulation.swift.simulation import SwiftSimulation, communication_weights
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.references import TaskRef
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings
from test_classic_baselines import logistic, run, assert_model, mean_models


def config(tmp_path, **kwargs):
    values = dict(algorithm='swift', participants=4, rounds=6, duration=3 * MICROSECONDS,
                  test_period=0, learning=LearningSettings(.1, 0, .01, 64, 1),
                  model_size_bytes=1024, network_latency_ms=3)
    values.update(kwargs)
    return settings(SwiftSettings, tmp_path, **values)


@pytest.mark.parametrize('graph', [nx.cycle_graph(4), nx.complete_graph(4),
                                   nx.random_regular_graph(3, 10, seed=12)])
@pytest.mark.parametrize('nonuniform', [False, True])
def test_ccs_expected_matrix_is_symmetric_and_doubly_stochastic(graph, nonuniform):
    n = len(graph)
    influence = np.arange(1, n + 1) ** 3 if nonuniform else np.ones(n)
    p, w = communication_weights(graph, influence)
    np.testing.assert_allclose(w.sum(axis=1), 1)
    assert np.min(w) >= 0 and np.min(w.diagonal()) >= 1 / n - 1e-14
    expected = np.eye(n)
    for i in range(n):
        active = np.eye(n)
        active[:, i] = w[i]
        expected += p[i] * (active - np.eye(n))
    np.testing.assert_allclose(expected, expected.T, atol=1e-14)
    np.testing.assert_allclose(expected.sum(axis=0), 1)
    np.testing.assert_allclose(expected.sum(axis=1), 1)
    if not nonuniform:
        degree = graph.degree(0)
        np.testing.assert_allclose(w, (nx.to_numpy_array(graph, nodelist=range(n)) + np.eye(n)) / (degree + 1))


def test_nonuniform_complete_ccs_has_closed_form():
    p, w = communication_weights(nx.complete_graph(4), [1, 2, 3, 4])
    np.testing.assert_allclose(p, [.1, .2, .3, .4])
    np.testing.assert_allclose(w, .25 * np.eye(4) + .75 * np.tile(p, (4, 1)))


@pytest.mark.asyncio
@pytest.mark.parametrize('interval', [1, 2, 3])
@pytest.mark.parametrize('skew', [False, True])
async def test_multistep_logistic_reference_and_latest_delivered_mailbox(tmp_path, logistic, interval, skew):
    # Match the independent fixture's model when calculating wire bytes.
    from unittest.mock import patch
    initial, partitions = logistic
    with patch('dasklearn.simulation.swift.simulation.create_model', return_value=copy.deepcopy(initial)):
        sim = SwiftSimulation(config(tmp_path, averaging_interval=interval, influence=[1, 2, 3, 4]))
    sim.setup_directories()
    arrivals = {i: {} for i in range(4)}
    reads = {}

    class Trace:
        def on_payload(self, client, event):
            peer = event.data['from']
            version = event.data['metadata']['version']
            old = arrivals[client.index].get(peer, (0, sim.initial_packet))
            if version > old[0]:
                arrivals[client.index][peer] = (version, event.data['payload'])

        def on_compute_complete(self, client, event):
            reads[client.index, event.data['params']['round']] = dict(arrivals[client.index])

    sim.observers.append(Trace())
    def capacities():
        for c in sim.clients:
            c.simulated_speed = [4, 17, 8, 31][c.index] if skew else 10
            c.bw_scheduler.bw_limit = 100000
    sim.apply_traces = capacities
    sim.simulate()
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    result = execute_local(sim)
    for client in sim.clients:
        for r, state in client.rounds.items():
            assert 'completed_time' in state
            task = sim.workflow_dag.tasks[state['model'].task]
            peers = sim.neighbors(client.index) if r % interval == 0 else []
            wanted = [reads[client.index, r].get(p, (0, sim.initial_packet)) for p in peers]
            assert task.data['packets'] == [ref for _, ref in wanted]
            assert state['versions'] == dict(zip(peers, [v for v, _ in wanted]))
            base = result[state['base'].task][0]
            gradient_model = copy.deepcopy(base)
            part = partitions[client.index]
            torch.nn.functional.cross_entropy(gradient_model(part.features), part.labels).backward()
            weights = [sim.weights[client.index, p] for p in peers]
            models = [result[sim.workflow_dag.tasks[ref.task].data['model'].task][0] for _, ref in wanted]
            expected = mean_models([base] + models,
                                   [1 - sum(weights)] + weights)
            with torch.no_grad():
                for target, old, grad in zip(expected.parameters(), base.parameters(), gradient_model.parameters()):
                    target.sub_(grad.grad + .01 * old, alpha=.1)
            assert_model(result[task.name][0], expected)
            assert state['completed_time'] == r * client.get_train_time()
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == sum(
        c.bw_scheduler.total_bytes_received for c in sim.clients)
    expected_bytes = sum(8 * (sim.packet_size + sim.settings.control_message_bytes + len(pickle.dumps({'version': v})))
                         for v in range(6))
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == expected_bytes
    assert any(v > 0 for c in sim.clients for s in c.rounds.values() for v in s['versions'].values())


@pytest.mark.asyncio
async def test_training_never_waits_for_slow_compute_or_unfinished_transfers(tmp_path):
    s = config(tmp_path, rounds=0, duration=MICROSECONDS // 2, stop='duration')
    sim = SwiftSimulation(s)
    sim.packet_size = 1000000  # Simulate an oversized dense parameter payload.
    sim.setup_directories()
    def capacities():
        for c in sim.clients:
            c.simulated_speed = 10 if c.index else 10000
            c.bw_scheduler.bw_limit = 1000
    sim.apply_traces = capacities
    sim.simulate()
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    assert not any('completed_time' in state for state in sim.clients[0].rounds.values())
    for client in sim.clients[1:]:
        completed = [state for state in client.rounds.values() if 'completed_time' in state]
        assert len(completed) >= 10
        assert completed[-1]['completed_time'] == len(completed) * client.get_train_time()
        assert all(set(state['versions'].values()) == {0} for state in completed)
    # Counters include partial wire traffic even when no model has arrived.
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) < sim.packet_size
    execute_local(sim, retain_results=False)


@pytest.mark.asyncio
async def test_versions_ignore_duplicates_and_out_of_order_arrivals(tmp_path):
    sim = run(SwiftSimulation, config(tmp_path))
    client = sim.clients[0]
    for version, ref in [(50, TaskRef('new', 0)), (49, TaskRef('stale', 0)), (50, TaskRef('duplicate', 0))]:
        client.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
            {'from': 1, 'model': ref, 'metadata': {'version': version}}))
    assert client.mailbox[1] == (50, TaskRef('new', 0))
    client.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
        {'from': 2, 'model': TaskRef('non-neighbor', 0), 'metadata': {'version': 100}}))
    assert 2 not in client.mailbox


@pytest.mark.asyncio
@pytest.mark.parametrize('n,topology', [(1, 'ring'), (2, 'ring'), (4, 'complete'), (6, 'kreg')])
async def test_streaming_execution_and_fixed_time_snapshots(tmp_path, n, topology):
    sim = run(SwiftSimulation, config(tmp_path, participants=n, topology=topology,
              degree=2, test_period=MICROSECONDS))
    execute_local(sim, retain_results=False)
    sim.merge_accuracies_files()
    frame = pd.read_csv(sim.data_dir + '/accuracies.csv')
    assert len(frame) == 3 * n
    assert frame.groupby('time').peer.nunique().eq(n).all()
    assert frame['round'].eq(6).all()
    assert np.isfinite(frame[['accuracy', 'loss']]).all().all()


@pytest.mark.asyncio
async def test_seeded_dag_and_training_reproducibility(tmp_path):
    dags, frames = [], []
    for repeat in range(2):
        Task.COUNTER.clear()
        sim = run(SwiftSimulation, config(tmp_path / str(repeat), test_period=MICROSECONDS), True)
        dags.append(sim.workflow_dag.serialize())
        execute_local(sim, retain_results=False)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv'))
    assert dags[0] == dags[1]
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


@pytest.mark.asyncio
async def test_saved_dag_replay_preserves_custom_operation_contracts(tmp_path):
    from dasklearn.tasks.dag import WorkflowDAG
    sim = run(SwiftSimulation, config(tmp_path), True)
    expected = execute_local(sim)
    sim.workflow_dag = WorkflowDAG.load_from_file(sim.data_dir + '/workflow_dag')
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    actual = execute_local(sim)
    assert expected.keys() == actual.keys()
    for name, values in expected.items():
        if isinstance(values[0], torch.nn.Module):
            assert_model(values[0], actual[name][0])
        else:
            torch.testing.assert_close(values[0], actual[name][0], rtol=0, atol=0)


def test_update_does_not_mutate_models_gradients_or_buffers():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    model(torch.randn(8, 3))
    model.gradient = [torch.ones_like(p) for p in model.parameters()]
    other = copy.deepcopy(model)
    before = copy.deepcopy(model.state_dict())
    gradients = [g.clone() for g in model.gradient]
    s = SimpleNamespace(learning=LearningSettings(.1, 0, .01, 8, 1))
    result = update(s, {'gradient_model': model, 'packets': [pack(s, {'model': other})[0]], 'weights': [.5]})[0]
    assert not hasattr(result, 'gradient')
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(other.state_dict()[name], value, rtol=0, atol=0)
    for name, value in model.named_buffers():
        torch.testing.assert_close(dict(result.named_buffers())[name], value, rtol=0, atol=0)
    for a, b in zip(model.gradient, gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
async def test_cpu_cuda_with_training_backend_enabled_matches_reference(tmp_path):
    from dasklearn.tasks.training import TrainingExecutor
    sim = run(SwiftSimulation, config(tmp_path), True)
    sim.settings.gpu_residency = sim.settings.training_batching = False
    cpu = execute_local(sim)
    sim.settings.torch_device_name = 'cuda:0'
    sim.settings.gpu_residency = sim.settings.training_batching = True
    backend = TrainingExecutor(sim.settings, autotune=False)
    cuda = execute_local(sim, training_backend=backend)
    for name in cpu:
        if isinstance(cpu[name][0], torch.nn.Module):
            assert_model(cpu[name][0], cuda[name][0])
        else:
            torch.testing.assert_close(cpu[name][0], cuda[name][0], rtol=2e-5, atol=2e-7)


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [dict(averaging_interval=0), dict(averaging_interval=1.5),
    dict(influence=[1, 2]), dict(influence=[1, 0, 1, 1]), dict(influence=[1, float('nan'), 1, 1]),
    dict(churn='synthetic'), dict(synchronous=True), dict(test_method='global'), dict(lora=True),
    dict(topology='unknown'), dict(topology='kreg', degree=4),
    dict(learning=LearningSettings(.1, 0, 0, 32, 5)), dict(learning=LearningSettings(0, 0, 0, 32, 1)),
    dict(learning=LearningSettings(.1, .9, 0, 32, 1))])
async def test_unsupported_settings_fail_early(tmp_path, extra):
    with pytest.raises(ValueError):
        SwiftSimulation(config(tmp_path, **extra))


def test_cli_and_baseline_alias_pass_swift_options():
    from args import get_args
    from dasklearn.settings_factory import session_options
    from dasklearn.simulation.peer_baselines import baseline_spec
    args = get_args(['--algorithm', 'swift', '--local-steps', '1', '--momentum', '0',
                     '--swift-averaging-interval', '3', '--swift-topology', 'complete',
                     '--swift-influence', '1', '2'])
    s = SwiftSettings(**session_options(args, 'cpu'))
    assert s.averaging_interval == 3 and s.topology == 'complete' and s.influence == [1, 2]
    assert baseline_spec('swift', 2, 2) == (SwiftSimulation, SwiftSettings, {})


def test_experiment_config_preserves_list_influence_and_cli_override(tmp_path):
    from args import get_args
    from dasklearn.settings_factory import session_options
    path = tmp_path / 'swift.json'
    path.write_text(json.dumps({'version': 1, 'settings': {
        'algorithm': 'swift', 'peers': 2, 'rounds': 3, 'local_steps': 1, 'momentum': 0,
        'swift_influence': [1, 2], 'swift_averaging_interval': 2}}))
    args = get_args(['--config', str(path)])
    s = SwiftSettings(**session_options(args, 'cpu'))
    assert s.influence == [1., 2.] and s.averaging_interval == 2
    override = get_args(['--config', str(path), '--swift-influence', '3', '4'])
    assert override.swift_influence == [3., 4.]


@pytest.mark.parametrize('value', [[], 1, '1,2', [True, 2], [1, float('inf')], [1, '2']])
def test_invalid_list_influence_config_is_rejected(tmp_path, value):
    from args import create_parser
    from dasklearn.experiments import validate_options
    with pytest.raises(ValueError, match='swift_influence'):
        validate_options(create_parser(), {'swift_influence': value}, tmp_path)
