"""RelaySum checked against individual delayed origins, not another relay loop."""
import copy
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn.optimizer.relaysgd import forward, finish
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.events import Event, INCOMING_MODEL
from dasklearn.simulation.relaysgd.settings import RelaySGDSettings
from dasklearn.simulation.relaysgd.simulation import RelaySGDSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.references import TaskRef
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings
from test_classic_baselines import logistic, run, assert_model


def config(tmp_path, **kwargs):
    values = dict(algorithm='relaysgd', participants=4, rounds=6, duration=4 * MICROSECONDS,
                  test_period=0, learning=LearningSettings(.1, 0, .01, 64, 1), network_latency_ms=3)
    values.update(kwargs)
    return settings(RelaySGDSettings, tmp_path, **values)


def vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def from_vector(template, values):
    model = copy.deepcopy(template)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(values[offset:offset + p.numel()].view_as(p))
            offset += p.numel()
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize('topology', ['balanced', 'path', 'star'])
@pytest.mark.parametrize('normalization', ['counts', 'initial'])
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.parametrize('skew', [False, True])
async def test_independent_delayed_origin_logistic_reference(
        tmp_path, monkeypatch, logistic, topology, normalization, synchronous, skew):
    import dasklearn.simulation.relaysgd.simulation as implementation
    initial, parts = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    sim = run(RelaySGDSimulation, config(tmp_path, topology=topology,
              normalization=normalization, synchronous=synchronous), skew)
    result = execute_local(sim)
    # The oracle explicitly enumerates individual origin histories by distance.
    # It never computes expected sums through the implementation's relay recurrence.
    distances = dict(nx.all_pairs_shortest_path_length(sim.graph))
    x = torch.stack([vector(initial)] * 4)
    history = []
    for r in range(1, 7):
        proposals = []
        for i, part in enumerate(parts):
            model = from_vector(initial, x[i])
            torch.nn.functional.cross_entropy(model(part.features), part.labels).backward()
            g = torch.cat([p.grad.flatten() for p in model.parameters()]) + .01 * x[i]
            proposals.append(x[i] - .1 * g)
        history.append(torch.stack(proposals))
        next_x = []
        for i, c in enumerate(sim.clients):
            represented = [j for j in range(4) if distances[i][j] <= r]
            values = [history[r - 1 - max(distances[i][j] - 1, 0)][j] for j in represented]
            expected = sum(values)
            if normalization == 'counts':
                expected = expected / len(represented)
            else:
                expected = (expected + (4 - len(represented)) * vector(initial)) / 4
            state = c.rounds[r]
            assert state['done'] and state['population'] == len(represented)
            assert_model(result[state['model'].task][0], from_vector(initial, expected))
            for peer, packet in state['sent'].items():
                branch = sim.graph.copy()
                branch.remove_edge(i, peer)
                origins = [j for j in nx.node_connected_component(branch, i) if distances[i][j] < r]
                assert packet['count'] == len(origins)
                wanted = sum(history[r - 1 - distances[i][j]][j] for j in origins)
                torch.testing.assert_close(result[packet['sum'].task][0], wanted, rtol=2e-5, atol=2e-7)
                task = sim.workflow_dag.tasks[packet['sum'].task]
                for ref in task.data['packets']:
                    source = sim.workflow_dag.tasks[ref.task]
                    assert source.data['round'] == r - 1 and source.data['peer'] != peer
            if r >= nx.eccentricity(sim.graph, i):
                assert state['population'] == 4
            next_x.append(expected)
        x = torch.stack(next_x)
    expected_bytes = sum(2 * 3 * (sim.parameter_bytes + 8 + sim.settings.control_message_bytes +
                         len(pickle.dumps({'round': r}))) for r in range(1, 7))
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == expected_bytes
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == expected_bytes


@pytest.mark.asyncio
async def test_early_current_messages_cannot_replace_previous_round_relay_state(tmp_path):
    sim = run(RelaySGDSimulation, config(tmp_path), True)
    c = sim.clients[0]
    before = len(sim.workflow_dag.tasks)
    # Completed messages and a duplicate train completion cannot send again.
    packet = c.rounds[1]['inbox'].messages[1]
    c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
        {'from': 1, 'model': packet, 'metadata': {'round': 1}}))
    c.finish_train(Event(sim.current_time, 0, 'finish_train', {'params': {'round': 1}}))
    for peer in sim.neighbors(0):
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
            {'from': peer, 'model': {'sum': TaskRef('early'), 'count': 1}, 'metadata': {'round': 9}}))
    assert not c.rounds[9]['inbox'].completed and not c.rounds[9]['done']
    assert len(sim.workflow_dag.tasks) == before
    for count in [0, -1, 4, True]:
        with pytest.raises(ValueError, match='count'):
            c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
                {'from': 1, 'model': {'sum': TaskRef('bad'), 'count': count}, 'metadata': {'round': 9}}))
    with pytest.raises(ValueError, match='Unexpected sender'):
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
            {'from': 3, 'model': {'sum': TaskRef('bad'), 'count': 1}, 'metadata': {'round': 9}}))


@pytest.mark.asyncio
async def test_training_and_neighbor_delivery_are_both_required(tmp_path):
    sim = run(RelaySGDSimulation, config(tmp_path, rounds=2, network_latency_ms=20), True)
    for c in sim.clients:
        for r, state in c.rounds.items():
            assert state['completed_time'] >= state['gradient_ready']
            for peer in sim.neighbors(c.index):
                assert state['completed_time'] >= sim.clients[peer].rounds[r]['gradient_ready'] + 20000
            if r > 1:
                task = sim.workflow_dag.tasks[state['local'].task]
                assert task.data['time'] >= c.rounds[r - 1]['completed_time']


@pytest.mark.asyncio
@pytest.mark.parametrize('n,topology', [(1, 'balanced'), (1, 'path'), (1, 'star'), (2, 'balanced'),
                                       (5, 'balanced'), (5, 'path'), (5, 'star')])
async def test_tree_edge_cases_streaming_and_snapshots(tmp_path, n, topology):
    sim = run(RelaySGDSimulation, config(tmp_path, participants=n, topology=topology,
                                         test_period=MICROSECONDS))
    assert nx.is_tree(sim.graph) and len(sim.graph.edges) == n - 1
    execute_local(sim, retain_results=False)
    sim.merge_accuracies_files()
    frame = pd.read_csv(sim.data_dir + '/accuracies.csv')
    assert len(frame) == 4 * n and frame.groupby('time').peer.nunique().eq(n).all()
    assert frame.loc[frame.time == 4, 'round'].eq(6).all()
    assert np.isfinite(frame[['accuracy', 'loss']]).all().all()
    assert all(c.rounds[6]['population'] == n for c in sim.clients)


@pytest.mark.asyncio
async def test_single_worker_reduces_to_plain_sgd(tmp_path, monkeypatch, logistic):
    import dasklearn.simulation.relaysgd.simulation as implementation
    from test_classic_baselines import local_sgd
    initial, parts = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    sim = run(RelaySGDSimulation, config(tmp_path, participants=1, learning=LearningSettings(.1, 0, 0, 64, 1)))
    result = execute_local(sim)
    assert_model(result[sim.clients[0].rounds[6]['model'].task][0], local_sgd(initial, parts[0], .1, 6))
    assert sim.clients[0].bw_scheduler.total_bytes_sent == 0


@pytest.mark.asyncio
async def test_reproducible_metrics_and_saved_dag_replay(tmp_path):
    from dasklearn.tasks.dag import WorkflowDAG
    dags, frames = [], []
    for repeat in range(2):
        Task.COUNTER.clear()
        sim = run(RelaySGDSimulation, config(tmp_path / str(repeat), test_period=MICROSECONDS), True)
        dags.append(sim.workflow_dag.serialize())
        sim.workflow_dag = WorkflowDAG.load_from_file(sim.data_dir + '/workflow_dag')
        sim.workflow_dag.build_task_indices()
        sim.workflow_dag.check_validity()
        execute_local(sim, retain_results=False)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv'))
    assert dags[0] == dags[1]
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


def test_vector_operations_preserve_inputs_and_local_buffers(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    model(torch.randn(8, 3))
    original = copy.deepcopy(model.state_dict())
    old = torch.ones_like(vector(model))
    old_copy = old.clone()
    parcel = forward(None, {'local_model': model, 'packets': [old]})[0]
    for normalization in ['counts', 'initial']:
        s = config(tmp_path, normalization=normalization)
        result = finish(s, {'local_model': model, 'packets': [parcel], 'counts': [1], 'initial_model': model})[0]
        for name, value in model.named_buffers():
            torch.testing.assert_close(dict(result.named_buffers())[name], value, rtol=0, atol=0)
    for name, value in original.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(old, old_copy, rtol=0, atol=0)
    assert not parcel.requires_grad


@pytest.mark.asyncio
@pytest.mark.parametrize('normalization', ['counts', 'initial'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
async def test_cpu_cuda_models_and_parcels_match(tmp_path, normalization):
    sim = run(RelaySGDSimulation, config(tmp_path, normalization=normalization), True)
    cpu = execute_local(sim)
    sim.settings.torch_device_name = 'cuda:0'
    gpu = execute_local(sim)
    for name, values in cpu.items():
        for a, b in zip(values, gpu[name]):
            if isinstance(a, torch.nn.Module):
                assert_model(a, b)
            else:
                torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-7)


@pytest.mark.asyncio
async def test_duration_stop_publishes_only_completed_rounds(tmp_path):
    sim = run(RelaySGDSimulation, config(tmp_path, rounds=1, stop='duration', duration=MICROSECONDS))
    for c in sim.clients:
        assert sum(state['done'] for state in c.rounds.values()) > 1
        r, model = sim.latest_models[c.index]
        assert c.rounds[r]['done'] and c.rounds[r]['model'] == model
        assert c.rounds[r]['completed_time'] <= MICROSECONDS
    execute_local(sim, retain_results=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [dict(topology='ring'), dict(normalization='local'), dict(churn='synthetic'),
    dict(lora=True), dict(test_method='global'), dict(learning=LearningSettings(.1, 0, 0, 32, 5)),
    dict(learning=LearningSettings(.1, .9, 0, 32, 1)), dict(learning=LearningSettings(0, 0, 0, 32, 1)),
    dict(learning=LearningSettings(float('nan'), 0, 0, 32, 1)), dict(learning=LearningSettings(.1, 0, -.1, 32, 1))])
async def test_unsupported_settings_fail_early(tmp_path, extra):
    with pytest.raises(ValueError):
        RelaySGDSimulation(config(tmp_path, **extra))


def test_cli_and_comparison_alias():
    from args import get_args
    from dasklearn.settings_factory import session_options
    from dasklearn.simulation.peer_baselines import baseline_spec
    args = get_args(['--algorithm', 'relaysgd', '--local-steps', '1', '--momentum', '0',
                     '--relaysgd-topology', 'path', '--relaysgd-normalization', 'initial'])
    s = RelaySGDSettings(**session_options(args, 'cpu'))
    assert s.topology == 'path' and s.normalization == 'initial'
    assert baseline_spec('relaysgd', 4, 4) == (RelaySGDSimulation, RelaySGDSettings, {})
