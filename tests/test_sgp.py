"""Independent SGP trajectories, mass conservation and directed-message timing."""
import copy
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn.optimizer.sgp import prepare, mix
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.events import Event, INCOMING_MODEL
from dasklearn.simulation.sgp.settings import SGPSettings
from dasklearn.simulation.sgp.simulation import SGPSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.references import TaskRef
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings
from test_classic_baselines import logistic, run, assert_model, local_sgd


def config(tmp_path, **kwargs):
    values = dict(algorithm='sgp', participants=4, rounds=6, duration=4 * MICROSECONDS,
                  test_period=0, learning=LearningSettings(.1, 0, .01, 64, 1), network_latency_ms=3)
    values.update(kwargs)
    return settings(SGPSettings, tmp_path, **values)


def vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def model_from(template, values):
    result = copy.deepcopy(template)
    with torch.no_grad():
        offset = 0
        for p in result.parameters():
            p.copy_(values[offset:offset + p.numel()].view_as(p))
            offset += p.numel()
    return result


def matrix(n, topology, r):
    # Independent sender-column specification; no implementation graph/weight API.
    P = torch.zeros(n, n, dtype=torch.float64)
    for j in range(n):
        destinations = {j}
        if n > 1:
            if topology == 'complete':
                destinations.update(range(n))
            else:
                offset = 2 ** ((r - 1) % (n - 1).bit_length()) if topology == 'exponential' else 1
                destinations.add((j + offset) % n)
                if topology == 'irregular' and n > 2 and j == 0:
                    destinations.add(2)
        for i in destinations:
            P[i, j] = 1 / len(destinations)
    return P


@pytest.mark.asyncio
@pytest.mark.parametrize('topology', ['exponential', 'ring', 'irregular', 'complete'])
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.parametrize('skew', [False, True])
@pytest.mark.parametrize('decay', [0., .01])
async def test_independent_logistic_trajectory_and_mass_conservation(
        tmp_path, monkeypatch, logistic, topology, synchronous, skew, decay):
    import dasklearn.simulation.sgp.simulation as implementation
    initial, parts = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    sim = run(SGPSimulation, config(tmp_path, topology=topology, synchronous=synchronous,
                                   learning=LearningSettings(.1, 0, decay, 64, 1)), skew)
    result = execute_local(sim)
    x = torch.stack([vector(initial)] * 4)
    mass = torch.ones(4, dtype=torch.float64)
    expected_bytes = 0
    for r in range(1, 7):
        P = matrix(4, topology, r)
        torch.testing.assert_close(P.sum(0), torch.ones(4, dtype=torch.float64))
        z = x / mass.to(x.dtype).unsqueeze(1)
        gradients = []
        for i, part in enumerate(parts):
            model = model_from(initial, z[i])
            torch.nn.functional.cross_entropy(model(part.features), part.labels).backward()
            gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]) + decay * z[i])
        gradients = torch.stack(gradients)
        half = x - .1 * gradients
        next_x, next_mass = P.to(x.dtype) @ half, P @ mass
        next_z = next_x / next_mass.to(x.dtype).unsqueeze(1)
        torch.testing.assert_close(next_x.sum(0), x.sum(0) - .1 * gradients.sum(0), rtol=2e-5, atol=3e-7)
        assert next_mass.sum().item() == pytest.approx(4., abs=1e-13)
        for i, c in enumerate(sim.clients):
            state = c.rounds[r]
            assert state['done'] and state['inbox'].completed
            assert state['mass'] == pytest.approx(next_mass[i].item(), abs=1e-14)
            assert state['packet']['mass'] == pytest.approx(mass[i].item() * P[i, i].item())
            assert_model(result[state['model'].task][0], model_from(initial, next_z[i]))
            torch.testing.assert_close(result[state['numerator'].task][1], next_x[i], rtol=2e-5, atol=3e-7)
            torch.testing.assert_close(result[state['packet']['numerator'].task][1],
                                       half[i] * P[i, i], rtol=2e-5, atol=3e-7)
            assert set(state['inbox'].messages) == {j for j in range(4) if j != i and P[i, j] > 0}
            task = sim.workflow_dag.tasks[state['packet']['numerator'].task]
            assert task.data['model'] == state['base']
            if r > 1:
                assert task.data['numerator'] == c.rounds[r - 1]['numerator']
                assert task.data['model'] == c.rounds[r - 1]['model']
        edges = int((P > 0).sum().item()) - 4
        expected_bytes += edges * (sim.parameter_bytes + 8 + sim.settings.control_message_bytes + len(pickle.dumps({'round': r})))
        x, mass = next_x, next_mass
    if topology == 'irregular':
        assert not torch.allclose(mass, torch.ones_like(mass))
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == expected_bytes
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == expected_bytes


@pytest.mark.asyncio
async def test_changing_oracle_is_evaluated_at_ratio_and_not_mass_scaled(tmp_path, monkeypatch, logistic):
    from dasklearn import functions
    import dasklearn.simulation.sgp.simulation as implementation
    initial, _ = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    noise = torch.randn(6, 4, 8, generator=torch.Generator().manual_seed(471)) * .1
    scales = torch.tensor([.5, 1., 2., 3.]).view(4, 1)
    def oracle(s, params):
        model = copy.deepcopy(params['model'])
        g = scales[params['peer']] * vector(model) + noise[params['round'] - 1, params['peer']]
        model.gradient = [g[:6].reshape(2, 3), g[6:]]
        return [model]
    monkeypatch.setattr(functions, 'compute_gradient', oracle)
    sim = run(SGPSimulation, config(tmp_path, topology='irregular', learning=LearningSettings(.1, 0, 0, 64, 1)), True)
    result = execute_local(sim)
    P = matrix(4, 'irregular', 1)
    x, mass = torch.stack([vector(initial)] * 4), torch.ones(4, dtype=torch.float64)
    for k in range(6):
        z = x / mass.to(x.dtype).unsqueeze(1)
        g = scales * z + noise[k]
        x, mass = P.to(x.dtype) @ (x - .1 * g), P @ mass
        for i, c in enumerate(sim.clients):
            assert_model(result[c.rounds[k + 1]['model'].task][0], model_from(initial, x[i] / mass[i]))


def test_pushsum_corrects_nonuniform_stationary_mass(tmp_path):
    P = matrix(4, 'irregular', 1)
    initial = torch.nn.Linear(3, 2).double()
    # Distinct origins, no gradients: raw numerators approach a biased limit,
    # while their ratios recover the unweighted initial average.
    x = torch.arange(32, dtype=torch.float64).reshape(4, 8)
    target = x.mean(0)
    mass = torch.ones(4, dtype=torch.float64)
    for _ in range(160):
        next_x, next_mass = [], P @ mass
        for i in range(4):
            local = model_from(initial, x[i] / mass[i])
            local.gradient = [torch.zeros_like(p) for p in local.parameters()]
            _, own = prepare(local, x[i], .1, 0, P[i, i].item())
            packets = [own] + [x[j] * P[i, j] for j in range(4) if j != i and P[i, j]]
            model, numerator = mix(None, {'local_model': local, 'packets': packets, 'mass': next_mass[i].item()})
            next_x.append(numerator)
        x, mass = torch.stack(next_x), next_mass
    torch.testing.assert_close(x / mass.unsqueeze(1), target.expand_as(x), rtol=1e-12, atol=1e-12)
    assert (x - target).abs().max() > 1


@pytest.mark.asyncio
async def test_complete_graph_matches_centralized_minibatch_sgd(tmp_path, monkeypatch, logistic):
    import dasklearn.simulation.sgp.simulation as implementation
    initial, parts = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    sim = run(SGPSimulation, config(tmp_path, topology='complete', learning=LearningSettings(.1, 0, 0, 64, 1)))
    result = execute_local(sim)
    model = copy.deepcopy(initial)
    for r in range(1, 7):
        model.zero_grad()
        # Equal client weights, not example weights (partitions have different sizes).
        sum(torch.nn.functional.cross_entropy(model(part.features), part.labels) / 4 for part in parts).backward()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(p.grad, alpha=-.1)
        for c in sim.clients:
            assert_model(result[c.rounds[r]['model'].task][0], model)
            assert c.rounds[r]['mass'] == 1.


@pytest.mark.asyncio
async def test_directed_waiting_and_network_costs(tmp_path):
    sim = SGPSimulation(config(tmp_path, topology='irregular', rounds=2, network_latency_ms=20))
    sim.setup_directories()
    def capacities():
        for c in sim.clients:
            c.simulated_speed = 100 if c.index == 1 else 2
            c.bw_scheduler.bw_limit = 100000
    sim.apply_traces = capacities
    sim.simulate()
    for c in sim.clients:
        for r, state in c.rounds.items():
            assert state['completed_time'] >= state['gradient_ready']
            for sender in sim.incoming(c.index, r):
                assert state['completed_time'] >= sim.clients[sender].rounds[r]['gradient_ready'] + 20000
            if r > 1:
                task = sim.workflow_dag.tasks[state['packet']['numerator'].task]
                assert task.data['time'] >= c.rounds[r - 1]['completed_time']
    # Client 0 sends to slow client 1 but only waits for incoming client 3.
    assert sim.clients[0].rounds[1]['completed_time'] < sim.clients[1].rounds[1]['gradient_ready']


@pytest.mark.asyncio
async def test_early_duplicate_and_wrong_phase_messages(tmp_path):
    sim = run(SGPSimulation, config(tmp_path))
    c = sim.clients[0]
    before = len(sim.workflow_dag.tasks)
    packet = c.rounds[1]['inbox'].messages[3]
    c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL, {'from': 3, 'model': packet, 'metadata': {'round': 1}}))
    c.finish_gradient(Event(sim.current_time, 0, 'finish_gradient', {'params': {'round': 1}}))
    # Round 9 uses hop1, round10 hop2: validate against the receiving round.
    c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
        {'from': 3, 'model': {'numerator': TaskRef('early'), 'mass': .5}, 'metadata': {'round': 9}}))
    assert not c.rounds[9]['done'] and not c.rounds[9]['inbox'].completed
    with pytest.raises(ValueError, match='Unexpected sender'):
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
            {'from': 3, 'model': packet, 'metadata': {'round': 10}}))
    for mass in [0., -1., float('inf'), float('nan'), True, 1]:
        with pytest.raises(ValueError, match='mass'):
            c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
                {'from': 3, 'model': {'numerator': TaskRef('bad'), 'mass': mass}, 'metadata': {'round': 9}}))
    assert len(sim.workflow_dag.tasks) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('n,topology', [(1, 'exponential'), (2, 'exponential'), (3, 'exponential'),
    (5, 'exponential'), (8, 'exponential'), (1, 'irregular'), (2, 'irregular'), (5, 'irregular'), (5, 'complete')])
async def test_topology_edge_cases_streaming_and_snapshots(tmp_path, n, topology):
    sim = run(SGPSimulation, config(tmp_path, participants=n, topology=topology, test_period=MICROSECONDS))
    assert nx.is_strongly_connected(nx.compose_all(sim.graphs))
    for r in range(1, len(sim.graphs) + 1):
        P = matrix(n, topology, r)
        for i in range(n):
            assert sim.outgoing(i, r) == [j for j in range(n) if j != i and P[j, i]]
        torch.testing.assert_close(P.sum(0), torch.ones(n, dtype=torch.float64))
    execute_local(sim, retain_results=False)
    sim.merge_accuracies_files()
    frame = pd.read_csv(sim.data_dir + '/accuracies.csv')
    assert len(frame) == 4 * n and frame.groupby('time').peer.nunique().eq(n).all()
    assert frame.loc[frame.time == 4, 'round'].eq(6).all()
    assert np.isfinite(frame[['accuracy', 'loss']]).all().all()


def test_exponential_cycle_reaches_exact_average_for_power_of_two():
    product = torch.eye(8, dtype=torch.float64)
    for r in range(1, 4):
        product = matrix(8, 'exponential', r) @ product
    torch.testing.assert_close(product, torch.full((8, 8), 1 / 8, dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.asyncio
async def test_single_worker_is_plain_sgd(tmp_path, monkeypatch, logistic):
    import dasklearn.simulation.sgp.simulation as implementation
    initial, parts = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    sim = run(SGPSimulation, config(tmp_path, participants=1, learning=LearningSettings(.1, 0, 0, 64, 1)))
    result = execute_local(sim)
    assert_model(result[sim.clients[0].rounds[6]['model'].task][0], local_sgd(initial, parts[0], .1, 6))
    assert sim.clients[0].bw_scheduler.total_bytes_sent == 0
    assert sim.clients[0].rounds[6]['mass'] == 1.


@pytest.mark.asyncio
async def test_reproducibility_and_saved_dag(tmp_path):
    from dasklearn.tasks.dag import WorkflowDAG
    dags, frames = [], []
    for repeat in range(2):
        Task.COUNTER.clear()
        sim = run(SGPSimulation, config(tmp_path / str(repeat), topology='irregular', test_period=MICROSECONDS), True)
        dags.append(sim.workflow_dag.serialize())
        sim.workflow_dag = WorkflowDAG.load_from_file(sim.data_dir + '/workflow_dag')
        sim.workflow_dag.build_task_indices()
        sim.workflow_dag.check_validity()
        execute_local(sim, retain_results=False)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv'))
    assert dags[0] == dags[1]
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


def test_inputs_immutable_and_buffers_local(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    model(torch.randn(8, 3))
    model.gradient = [torch.ones_like(p) for p in model.parameters()]
    before = copy.deepcopy(model.state_dict())
    numerator = vector(model) * 2
    original = numerator.clone()
    local, packet = prepare(model, numerator, .1, .01, .5)
    result, x = mix(None, {'local_model': local, 'packets': [packet, packet], 'mass': 2.})
    assert not hasattr(local, 'gradient') and not hasattr(result, 'gradient')
    assert not packet.requires_grad and not x.requires_grad
    torch.testing.assert_close(numerator, original, rtol=0, atol=0)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    for name, value in local.named_buffers():
        torch.testing.assert_close(dict(result.named_buffers())[name], value, rtol=0, atol=0)


@pytest.mark.asyncio
@pytest.mark.parametrize('topology', ['exponential', 'irregular'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
async def test_cpu_cuda_all_states_match(tmp_path, topology):
    sim = run(SGPSimulation, config(tmp_path, topology=topology), True)
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
async def test_duration_stop_publishes_complete_ratios(tmp_path):
    sim = run(SGPSimulation, config(tmp_path, rounds=1, topology='irregular', stop='duration', duration=MICROSECONDS))
    for c in sim.clients:
        assert sum(state['done'] for state in c.rounds.values()) > 1
        r, model = sim.latest_models[c.index]
        assert c.rounds[r]['done'] and c.rounds[r]['model'] == model
        assert c.rounds[r]['completed_time'] <= MICROSECONDS
    execute_local(sim, retain_results=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [dict(topology='unknown'), dict(churn='synthetic'), dict(lora=True),
    dict(test_method='global'), dict(learning=LearningSettings(.1, 0, 0, 32, 5)),
    dict(learning=LearningSettings(.1, .9, 0, 32, 1)), dict(learning=LearningSettings(0, 0, 0, 32, 1)),
    dict(learning=LearningSettings(float('nan'), 0, 0, 32, 1)), dict(learning=LearningSettings(.1, 0, -.1, 32, 1))])
async def test_unsupported_settings_fail_early(tmp_path, extra):
    with pytest.raises(ValueError):
        SGPSimulation(config(tmp_path, **extra))


def test_cli_and_baseline_alias():
    from args import get_args
    from dasklearn.settings_factory import session_options
    from dasklearn.simulation.peer_baselines import baseline_spec
    args = get_args(['--algorithm', 'sgp', '--local-steps', '1', '--momentum', '0', '--sgp-topology', 'irregular'])
    s = SGPSettings(**session_options(args, 'cpu'))
    assert s.topology == 'irregular'
    assert baseline_spec('sgp', 4, 4) == (SGPSimulation, SGPSettings, {})
