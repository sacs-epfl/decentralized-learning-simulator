"""Independent phase trajectories, fixed-point gradients and overlapped timing."""
import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn.simulation.detag.settings import DeTAGSettings
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.detag.simulation import DeTAGSimulation
from dasklearn.simulation.events import Event, INCOMING_MODEL
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.references import TaskRef
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings
from test_classic_baselines import logistic, run, assert_model


def config(tmp_path, **kwargs):
    values = dict(algorithm='detag', participants=4, rounds=4, duration=4 * MICROSECONDS,
                  test_period=0, learning=LearningSettings(.1, 0, .01, 64, 1),
                  network_latency_ms=3)
    values.update(kwargs)
    return settings(DeTAGSettings, tmp_path, **values)


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
@pytest.mark.parametrize('R', [1, 2, 3])
@pytest.mark.parametrize('eta', [0., .2])
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.parametrize('topology', ['ring', 'complete'])
async def test_independent_logistic_phases_and_every_accelerated_step(
        tmp_path, monkeypatch, logistic, R, eta, synchronous, topology):
    import dasklearn.simulation.detag.simulation as implementation
    initial, parts = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    sim = run(DeTAGSimulation, config(tmp_path, phase_length=R, eta=eta, synchronous=synchronous, topology=topology), True)
    result = execute_local(sim)
    W = torch.tensor([[1 / (3 if topology == 'ring' else 4)
                       if topology == 'complete' or (i - j) % 4 in [0, 1, 3] else 0.
                       for j in range(4)] for i in range(4)])
    x = torch.stack([vector(initial)] * 4)
    y = torch.zeros_like(x)
    old_g = torch.zeros_like(x)
    for phase in range(1, 5):
        gradients = []
        for i, part in enumerate(parts):
            model = from_vector(initial, x[i])
            torch.nn.functional.cross_entropy(model(part.features), part.labels).backward()
            gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]) + .01 * x[i])
        g = torch.stack(gradients)
        z = torch.stack([x, y], dim=1)
        previous = z.clone()
        for step in range(1, R + 1):
            next_z = (1 + eta) * torch.einsum('ij,jcd->icd', W, z) - eta * previous
            for i, c in enumerate(sim.clients):
                inner = c.rounds[phase]['steps'][step]
                torch.testing.assert_close(result[inner['mixed'].task][0], next_z[i], rtol=2e-5, atol=2e-7)
                torch.testing.assert_close(result[inner['gradient'].task][1], g[i] * step / R, rtol=2e-5, atol=2e-7)
                # Every gradient in the phase evaluates the original parameters.
                task = sim.workflow_dag.tasks[inner['local_model'].task]
                assert_model(result[task.data['model'].task][0], from_vector(initial, x[i]))
                assert task.data['round'] == (phase - 1) * R + step
            previous, z = z, next_z
        next_x = z[:, 0] - .1 * z[:, 1]
        next_y = z[:, 1] + g - old_g
        torch.testing.assert_close(next_y.mean(0), g.mean(0), rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(next_x.mean(0), x.mean(0) - .1 * old_g.mean(0), rtol=2e-5, atol=2e-7)
        for i, c in enumerate(sim.clients):
            state = c.rounds[phase]
            assert state['done']
            assert_model(result[state['model'].task][0], from_vector(initial, next_x[i]))
            torch.testing.assert_close(result[state['tracker'].task][1], next_y[i], rtol=2e-5, atol=2e-7)
            if phase == 1:
                assert_model(result[state['model'].task][0], initial)
            if phase > 1:
                assert state['previous_gradient'] == c.rounds[phase - 1]['gradient']
        x, y, old_g = next_x, next_y, g
    degree = 2 if topology == 'ring' else 3
    expected = sum(4 * degree * (sim.packet_size + sim.settings.control_message_bytes +
                   len(pickle.dumps({'phase': p, 'step': s})))
                   for p in range(1, 5) for s in range(1, R + 1))
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == expected
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == expected
    assert all(sum(c.incoming_counter.values()) == 4 * R * degree for c in sim.clients)


@pytest.mark.asyncio
@pytest.mark.parametrize('bandwidth', [10000, 1000000])
async def test_compute_and_network_overlap_without_free_gossip(tmp_path, bandwidth):
    sim = DeTAGSimulation(config(tmp_path, rounds=3, eta=.2, network_latency_ms=10))
    sim.packet_size = 1000
    sim.setup_directories()
    def capacities():
        for c in sim.clients:
            c.simulated_speed = 10
            c.bw_scheduler.bw_limit = bandwidth
    sim.apply_traces = capacities
    sim.simulate()
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    for c in sim.clients:
        last_time = 0
        for phase in range(1, 4):
            for step in range(1, 3):
                inner = c.rounds[phase]['steps'][step]
                assert inner['started_time'] == last_time
                assert inner['gradient_ready'] == inner['started_time'] + c.get_train_time()
                assert inner['gossip_ready'] > inner['started_time']
                assert inner['completed_time'] == max(inner['gradient_ready'], inner['gossip_ready'])
                if bandwidth == 1000000:
                    assert inner['gossip_ready'] < inner['gradient_ready']
                else:
                    assert inner['gradient_ready'] < inner['gossip_ready']
                last_time = inner['completed_time']
        if bandwidth == 1000000:
            assert last_time == 6 * c.get_train_time()
    execute_local(sim, retain_results=False)


@pytest.mark.asyncio
async def test_changing_stochastic_oracle_averages_R_fixed_parameter_draws(tmp_path, monkeypatch, logistic):
    from dasklearn import functions
    from dasklearn.session_settings import LearningSettings
    import dasklearn.simulation.detag.simulation as implementation
    initial, _ = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    R = 3
    noise = torch.randn(12, 4, vector(initial).numel(), generator=torch.Generator().manual_seed(92)) * .1
    observed = {}
    def oracle(s, params):
        model = copy.deepcopy(params['model'])
        observed[params['round'], params['peer']] = vector(model)
        values = vector(model) + noise[params['round'] - 1, params['peer']]
        offset = 0
        model.gradient = []
        for p in model.parameters():
            model.gradient.append(values[offset:offset + p.numel()].view_as(p).clone())
            offset += p.numel()
        return [model]
    monkeypatch.setattr(functions, 'compute_gradient', oracle)
    sim = run(DeTAGSimulation, config(tmp_path, phase_length=R, eta=0.,
              learning=LearningSettings(.1, 0, 0, 64, 1)), True)
    result = execute_local(sim)
    W = torch.tensor([[1 / 3 if (i - j) % 4 in [0, 1, 3] else 0. for j in range(4)] for i in range(4)])
    M = torch.linalg.matrix_power(W, R)
    x = torch.stack([vector(initial)] * 4)
    y, old_g = torch.zeros_like(x), torch.zeros_like(x)
    for k in range(4):
        g = x + noise[k * R:(k + 1) * R].mean(0)
        for step in range(R):
            for i in range(4):
                torch.testing.assert_close(observed[k * R + step + 1, i], x[i])
        next_x = M @ x - .1 * (M @ y)
        next_y = M @ y + g - old_g
        for i, c in enumerate(sim.clients):
            state = c.rounds[k + 1]
            assert_model(result[state['model'].task][0], from_vector(initial, next_x[i]))
            torch.testing.assert_close(result[state['gradient'].task][1], g[i], rtol=2e-5, atol=2e-7)
            torch.testing.assert_close(result[state['tracker'].task][1], next_y[i], rtol=2e-5, atol=2e-7)
        x, y, old_g = next_x, next_y, g


@pytest.mark.asyncio
@pytest.mark.parametrize('n,topology', [(1, 'ring'), (2, 'ring'), (4, 'complete'), (6, 'kreg')])
async def test_spectrum_topologies_streaming_and_snapshots(tmp_path, n, topology):
    sim = run(DeTAGSimulation, config(tmp_path, participants=n, topology=topology, degree=2,
                                      test_period=MICROSECONDS))
    assert sim.settings.eta == pytest.approx((1 - np.sqrt(1 - sim.spectral_radius ** 2)) /
                                            (1 + np.sqrt(1 - sim.spectral_radius ** 2)))
    np.testing.assert_allclose(sim.weights.sum(0), 1)
    np.testing.assert_allclose(sim.weights.sum(1), 1)
    execute_local(sim, retain_results=False)
    sim.merge_accuracies_files()
    frame = pd.read_csv(sim.data_dir + '/accuracies.csv')
    assert len(frame) == 4 * n and frame.groupby('time').peer.nunique().eq(n).all()
    assert frame.loc[frame.time == 4, 'round'].eq(4).all()
    assert np.isfinite(frame[['accuracy', 'loss']]).all().all()


@pytest.mark.asyncio
async def test_early_duplicate_and_invalid_messages(tmp_path):
    sim = run(DeTAGSimulation, config(tmp_path))
    c = sim.clients[0]
    before = len(sim.workflow_dag.tasks)
    c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
        {'from': 1, 'model': TaskRef('duplicate'), 'metadata': {'phase': 1, 'step': 1}}))
    for peer in [1, 3]:
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
            {'from': peer, 'model': TaskRef('early'), 'metadata': {'phase': 9, 'step': 2}}))
    assert not c.rounds[9]['steps'][2]['inbox'].completed
    assert not c.rounds[9]['steps'][2]['done'] and len(sim.workflow_dag.tasks) == before
    for metadata in [{'phase': 0, 'step': 1}, {'phase': 9, 'step': 3}]:
        with pytest.raises(ValueError):
            c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
                {'from': 1, 'model': TaskRef('bad'), 'metadata': metadata}))
    with pytest.raises(ValueError, match='Unexpected sender'):
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
            {'from': 2, 'model': TaskRef('bad'), 'metadata': {'phase': 9, 'step': 2}}))


@pytest.mark.asyncio
async def test_reproducibility_and_saved_dag_replay(tmp_path):
    from dasklearn.tasks.dag import WorkflowDAG
    dags, frames = [], []
    for repeat in range(2):
        Task.COUNTER.clear()
        sim = run(DeTAGSimulation, config(tmp_path / str(repeat), test_period=MICROSECONDS), True)
        dags.append(sim.workflow_dag.serialize())
        sim.workflow_dag = WorkflowDAG.load_from_file(sim.data_dir + '/workflow_dag')
        sim.workflow_dag.build_task_indices()
        sim.workflow_dag.check_validity()
        execute_local(sim, retain_results=False)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv'))
    assert dags[0] == dags[1]
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
async def test_cpu_cuda_full_state_equivalence(tmp_path):
    sim = run(DeTAGSimulation, config(tmp_path, eta=.2), True)
    cpu = execute_local(sim)
    sim.settings.torch_device_name = 'cuda:0'
    gpu = execute_local(sim)
    for name, values in cpu.items():
        for a, b in zip(values, gpu[name]):
            if isinstance(a, torch.nn.Module):
                assert_model(a, b)
            else:
                torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-7)


def test_numerical_operations_preserve_inputs_and_local_buffers(tmp_path, monkeypatch):
    from dasklearn import functions
    from dasklearn.optimizer.detag import packet, gradient, gossip, finish
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    model(torch.randn(8, 3))
    model.gradient = [torch.ones_like(p) for p in model.parameters()]
    before = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(functions, 'compute_gradient', lambda *a, **k: [copy.deepcopy(model)])
    s = config(tmp_path, eta=.2)
    old_tracker, old_gradient = torch.ones_like(vector(model)), torch.zeros_like(vector(model))
    tracker_copy, gradient_copy = old_tracker.clone(), old_gradient.clone()
    initial = packet(s, {'model': model, 'tracker': old_tracker})[0]
    initial_copy = initial.clone()
    local, g = gradient(s, {'model': model, 'accumulator': old_gradient})
    mixed = gossip(s, {'packets': [initial, initial + .1], 'weights': [.5, .5], 'previous': initial})[0]
    result, next_tracker = finish(s, {'packet': mixed, 'local_model': local,
                                      'gradient': g, 'previous_gradient': old_gradient})
    assert not hasattr(local, 'gradient') and not hasattr(result, 'gradient')
    assert not next_tracker.requires_grad and not initial.requires_grad
    torch.testing.assert_close(old_tracker, tracker_copy, rtol=0, atol=0)
    torch.testing.assert_close(old_gradient, gradient_copy, rtol=0, atol=0)
    torch.testing.assert_close(initial, initial_copy, rtol=0, atol=0)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(local.state_dict()[name], value, rtol=0, atol=0)
    for name, value in local.named_buffers():
        torch.testing.assert_close(dict(result.named_buffers())[name], value, rtol=0, atol=0)


@pytest.mark.asyncio
async def test_duration_stop_publishes_only_complete_phases(tmp_path):
    sim = run(DeTAGSimulation, config(tmp_path, rounds=1, stop='duration', duration=MICROSECONDS))
    for c in sim.clients:
        assert sum(state['done'] for state in c.rounds.values()) > 1
        published_phase, published_model = sim.latest_models[c.index]
        assert c.rounds[published_phase]['done']
        assert c.rounds[published_phase]['model'] == published_model
        assert c.rounds[published_phase]['completed_time'] <= MICROSECONDS
    execute_local(sim, retain_results=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [dict(phase_length=0), dict(phase_length=1.5), dict(eta=-.1), dict(eta=1),
    dict(eta=float('nan')), dict(churn='synthetic'), dict(lora=True), dict(test_method='global'),
    dict(topology='unknown'), dict(topology='kreg', degree=4),
    dict(learning=LearningSettings(.1, 0, 0, 32, 5)), dict(learning=LearningSettings(.1, .9, 0, 32, 1)),
    dict(learning=LearningSettings(0, 0, 0, 32, 1)), dict(learning=LearningSettings(float('nan'), 0, 0, 32, 1)),
    dict(learning=LearningSettings(.1, 0, -.1, 32, 1))])
async def test_unsupported_settings_fail_early(tmp_path, extra):
    with pytest.raises(ValueError):
        DeTAGSimulation(config(tmp_path, **extra))


def test_cli_and_baseline_alias():
    from args import get_args
    from dasklearn.settings_factory import session_options
    from dasklearn.simulation.peer_baselines import baseline_spec
    args = get_args(['--algorithm', 'detag', '--local-steps', '1', '--momentum', '0',
                     '--detag-topology', 'complete', '--detag-phase-length', '3', '--detag-eta', '.2'])
    s = DeTAGSettings(**session_options(args, 'cpu'))
    assert s.phase_length == 3 and s.eta == .2 and s.topology == 'complete'
    assert baseline_spec('detag', 4, 4) == (DeTAGSimulation, DeTAGSettings, {})
