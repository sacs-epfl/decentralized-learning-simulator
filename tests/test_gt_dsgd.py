"""Independent ATC GT-DSGD trajectories, tracker conservation and wire timing."""
import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn.optimizer.gt_dsgd import flat, prepare, track, mix
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.events import Event, INCOMING_MODEL
from dasklearn.simulation.gt_dsgd.settings import GTDSGDSettings
from dasklearn.simulation.gt_dsgd.simulation import GTDSGDSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.references import TaskRef
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings
from test_classic_baselines import logistic, run, assert_model, local_sgd


def config(tmp_path, **kwargs):
    values = dict(algorithm='gt-dsgd', participants=4, rounds=5, duration=3 * MICROSECONDS,
                  test_period=0, learning=LearningSettings(.1, 0, .01, 64, 1),
                  model_size_bytes=1024, network_latency_ms=3)
    values.update(kwargs)
    return settings(GTDSGDSettings, tmp_path, **values)


def model_from(template, vector):
    model = copy.deepcopy(template)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(vector[offset:offset + p.numel()].view_as(p))
            offset += p.numel()
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.parametrize('skew', [False, True])
@pytest.mark.parametrize('decay', [0., .01])
@pytest.mark.parametrize('topology', ['ring', 'complete'])
async def test_independent_logistic_reference_and_tracker_conservation(
        tmp_path, monkeypatch, logistic, synchronous, skew, decay, topology):
    import dasklearn.simulation.gt_dsgd.simulation as implementation
    initial, partitions = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    s = config(tmp_path, synchronous=synchronous, topology=topology,
               learning=LearningSettings(.1, 0, decay, 64, 1))
    sim = run(GTDSGDSimulation, s, skew)
    result = execute_local(sim)
    # Construct W independently for the supported regular ring/complete graphs.
    W = torch.tensor([[1 / (3 if topology == 'ring' else 4)
                       if topology == 'complete' or (i - j) % 4 in [0, 1, 3] else 0.
                       for j in range(4)] for i in range(4)])
    x = torch.stack([torch.cat([p.detach().flatten() for p in initial.parameters()])] * 4)
    tracker = torch.zeros_like(x)
    previous = torch.zeros_like(x)
    for r in range(1, 6):
        gradients = []
        for i in range(4):
            model = model_from(initial, x[i])
            part = partitions[i]
            torch.nn.functional.cross_entropy(model(part.features), part.labels).backward()
            gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]) + decay * x[i])
        gradients = torch.stack(gradients)
        corrected = tracker + gradients - previous
        next_tracker = W @ corrected
        proposals = x - .1 * next_tracker
        next_x = W @ proposals
        torch.testing.assert_close(next_tracker.mean(0), gradients.mean(0), rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(next_x.mean(0), x.mean(0) - .1 * gradients.mean(0), rtol=2e-5, atol=2e-7)
        for i, client in enumerate(sim.clients):
            state = client.rounds[r]
            assert state['done']
            assert_model(result[state['model'].task][0], model_from(initial, next_x[i]))
            torch.testing.assert_close(result[state['tracker'].task][0], next_tracker[i], rtol=2e-5, atol=2e-7)
            torch.testing.assert_close(result[state['gradient'].task][1], gradients[i], rtol=2e-5, atol=2e-7)
            torch.testing.assert_close(result[state['corrected'].task][2], corrected[i], rtol=2e-5, atol=2e-7)
            assert state['tracker_ready'] >= state['gradient_ready']
            assert state['completed_time'] >= state['tracker_ready']
            assert state['tracker_inbox'].completed and state['model_inbox'].completed
            if r > 1:
                before = client.rounds[r - 1]
                task = sim.workflow_dag.tasks[state['gradient'].task]
                assert task.data['tracker'] == before['tracker']
                assert task.data['previous_gradient'] == before['gradient']
                assert task.data['model'] == before['model']
        x, tracker, previous = next_x, next_tracker, gradients
    degree = 2 if topology == 'ring' else 3
    expected = sum(4 * degree * (sim.packet_size + s.control_message_bytes +
                   len(pickle.dumps({'round': r, 'phase': phase})))
                   for r in range(1, 6) for phase in ['tracker', 'model'])
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == expected
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == expected
    assert all(sum(c.incoming_counter.values()) == 5 * 2 * degree for c in sim.clients)


@pytest.mark.asyncio
async def test_single_client_reduces_to_plain_sgd(tmp_path, monkeypatch, logistic):
    import dasklearn.simulation.gt_dsgd.simulation as implementation
    initial, parts = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    sim = run(GTDSGDSimulation, config(tmp_path, participants=1, learning=LearningSettings(.1, 0, 0, 64, 1)))
    result = execute_local(sim)
    expected = local_sgd(initial, parts[0], .1, 5)
    assert_model(result[sim.clients[0].rounds[5]['model'].task][0], expected)
    assert sim.clients[0].bw_scheduler.total_bytes_sent == 0


@pytest.mark.asyncio
async def test_both_exchanges_wait_for_delivery_and_local_gradient(tmp_path):
    s = config(tmp_path, rounds=1, network_latency_ms=20)
    sim = run(GTDSGDSimulation, s, True)
    for c in sim.clients:
        state = c.rounds[1]
        assert state['tracker_ready'] >= state['gradient_ready']
        assert state['completed_time'] >= state['tracker_ready']
        for phase, ready in [('tracker', state['tracker_ready']), ('model', state['completed_time'])]:
            for p, ref in state[phase + '_inbox'].messages.items():
                sent_at = sim.clients[p].rounds[1]['gradient_ready' if phase == 'tracker' else 'tracker_ready']
                assert ready >= sent_at + 20000
                assert sim.workflow_dag.tasks[ref.task].available_at <= ready
    # A fast node must not use its own gradient before the slow neighbor has sent.
    assert sim.clients[0].rounds[1]['tracker_ready'] > sim.clients[0].rounds[1]['gradient_ready']


@pytest.mark.asyncio
async def test_early_model_packets_and_duplicate_phase_messages(tmp_path):
    sim = run(GTDSGDSimulation, config(tmp_path))
    c = sim.clients[0]
    before = len(sim.workflow_dag.tasks)
    for phase in ['tracker', 'model']:
        # Duplicates of completed phases cannot schedule another mix or step.
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
            {'from': 1, 'model': TaskRef('duplicate', 0), 'metadata': {'round': 1, 'phase': phase}}))
    assert len(sim.workflow_dag.tasks) == before
    # A future second-phase packet may arrive before this node starts that round.
    for phase in ['tracker', 'model']:
        for peer in [1, 3]:
            event = Event(sim.current_time, 0, INCOMING_MODEL,
                {'from': peer, 'model': TaskRef('early', 0), 'metadata': {'round': 9, 'phase': phase}})
            c.on_incoming_model(event)
        assert set(c.rounds[9][phase + '_inbox'].messages) == {1, 3}
        assert not c.rounds[9][phase + '_inbox'].completed
    assert not c.rounds[9]['done'] and len(sim.workflow_dag.tasks) == before
    c.on_incoming_model(event)
    assert len(sim.workflow_dag.tasks) == before
    with pytest.raises(ValueError, match='Unexpected sender'):
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
            {'from': 2, 'model': TaskRef('bad', 0), 'metadata': {'round': 9, 'phase': 'tracker'}}))


@pytest.mark.asyncio
async def test_duration_stop_continues_past_round_setting(tmp_path):
    sim = run(GTDSGDSimulation, config(tmp_path, rounds=1, stop='duration', duration=MICROSECONDS // 2))
    assert all(sum(s['done'] for s in c.rounds.values()) > 1 for c in sim.clients)
    assert all(s['completed_time'] <= MICROSECONDS // 2 for c in sim.clients
               for s in c.rounds.values() if s['done'])
    execute_local(sim, retain_results=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('n,topology', [(1, 'ring'), (2, 'ring'), (4, 'complete'), (6, 'kreg')])
async def test_streaming_topologies_and_fixed_time_snapshots(tmp_path, n, topology):
    sim = run(GTDSGDSimulation, config(tmp_path, participants=n, topology=topology, degree=2,
                                      test_period=MICROSECONDS))
    execute_local(sim, retain_results=False)
    sim.merge_accuracies_files()
    frame = pd.read_csv(sim.data_dir + '/accuracies.csv')
    assert len(frame) == 3 * n and frame.groupby('time').peer.nunique().eq(n).all()
    assert frame['round'].between(1, 5).all()
    assert frame.loc[frame.time == 3, 'round'].eq(5).all()
    assert np.isfinite(frame[['accuracy', 'loss']]).all().all()


@pytest.mark.asyncio
async def test_stochastic_gradient_differences_reuse_exact_previous_draw(tmp_path, monkeypatch, logistic):
    from dasklearn import functions
    import dasklearn.simulation.gt_dsgd.simulation as implementation
    initial, _ = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    # Known changing oracle noise makes a redraw at x_{k-1} observably wrong.
    generator = torch.Generator().manual_seed(472)
    noise = torch.randn(5, 4, flat(initial).numel(), generator=generator) * .1
    scales = torch.tensor([.5, 1., 2., 3.]).view(4, 1)
    def oracle(s, params):
        model = copy.deepcopy(params['model'])
        values = scales[params['peer']] * torch.cat([p.detach().flatten() for p in model.parameters()])
        values += noise[params['round'] - 1, params['peer']]
        offset = 0
        model.gradient = []
        for p in model.parameters():
            model.gradient.append(values[offset:offset + p.numel()].view_as(p).clone())
            offset += p.numel()
        return [model]
    monkeypatch.setattr(functions, 'compute_gradient', oracle)
    sim = run(GTDSGDSimulation, config(tmp_path, learning=LearningSettings(.1, 0, 0, 64, 1)), True)
    result = execute_local(sim)
    W = torch.tensor([[1 / 3 if (i - j) % 4 in [0, 1, 3] else 0. for j in range(4)] for i in range(4)])
    x = torch.stack([torch.cat([p.detach().flatten() for p in initial.parameters()])] * 4)
    y = torch.zeros_like(x)
    previous = torch.zeros_like(x)
    for k in range(5):
        g = scales * x + noise[k]
        y = W @ (y + g - previous)
        x = W @ (x - .1 * y)
        for i, client in enumerate(sim.clients):
            state = client.rounds[k + 1]
            assert_model(result[state['model'].task][0], model_from(initial, x[i]))
            torch.testing.assert_close(result[state['tracker'].task][0], y[i], rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(y.mean(0), g.mean(0), rtol=2e-5, atol=2e-7)
        previous = g


@pytest.mark.asyncio
async def test_seeded_reproducibility_and_saved_dag_state(tmp_path):
    from dasklearn.tasks.dag import WorkflowDAG
    dags, frames = [], []
    for repeat in range(2):
        Task.COUNTER.clear()
        sim = run(GTDSGDSimulation, config(tmp_path / str(repeat), test_period=MICROSECONDS), True)
        dags.append(sim.workflow_dag.serialize())
        sim.workflow_dag = WorkflowDAG.load_from_file(sim.data_dir + '/workflow_dag')
        sim.workflow_dag.build_task_indices()
        sim.workflow_dag.check_validity()
        execute_local(sim, retain_results=False)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv'))
    assert dags[0] == dags[1]
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


def test_no_input_mutation_and_buffers_stay_local(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    model(torch.randn(8, 3))
    model.gradient = [torch.ones_like(p) for p in model.parameters()]
    before = copy.deepcopy(model.state_dict())
    old_tracker, old_gradient = torch.ones_like(flat(model)), torch.zeros_like(flat(model))
    tracker_copy, gradient_copy = old_tracker.clone(), old_gradient.clone()
    local, gradient, corrected = prepare(model, old_tracker, old_gradient, .01)
    s = config(tmp_path)
    tracker, proposal = track(s, {'local_model': local, 'packets': [corrected, corrected + 1], 'weights': [.5, .5]})
    result = mix(s, {'local_model': local, 'packets': [proposal, proposal + 1], 'weights': [.5, .5]})[0]
    assert not hasattr(local, 'gradient') and not hasattr(result, 'gradient')
    assert not gradient.requires_grad and not tracker.requires_grad and not proposal.requires_grad
    torch.testing.assert_close(old_tracker, tracker_copy, rtol=0, atol=0)
    torch.testing.assert_close(old_gradient, gradient_copy, rtol=0, atol=0)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(local.state_dict()[name], value, rtol=0, atol=0)
    for name, value in local.named_buffers():
        torch.testing.assert_close(dict(result.named_buffers())[name], value, rtol=0, atol=0)


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
async def test_cpu_cuda_models_and_tracker_states_match(tmp_path):
    sim = run(GTDSGDSimulation, config(tmp_path), True)
    cpu = execute_local(sim)
    sim.settings.torch_device_name = 'cuda:0'
    gpu = execute_local(sim)
    for name, values in cpu.items():
        for a, b in zip(values, gpu[name]):
            if isinstance(a, torch.nn.Module):
                assert_model(a, b)
            else:
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [dict(churn='synthetic'), dict(lora=True), dict(test_method='global'),
    dict(topology='unknown'), dict(topology='kreg', degree=4), dict(topology='kreg', degree=0),
    dict(learning=LearningSettings(.1, 0, 0, 32, 5)), dict(learning=LearningSettings(.1, .9, 0, 32, 1)),
    dict(learning=LearningSettings(0, 0, 0, 32, 1)), dict(learning=LearningSettings(float('nan'), 0, 0, 32, 1)),
    dict(learning=LearningSettings(.1, 0, -.1, 32, 1))])
async def test_unsupported_settings_fail_early(tmp_path, extra):
    with pytest.raises(ValueError):
        GTDSGDSimulation(config(tmp_path, **extra))


def test_cli_and_comparison_alias():
    from args import get_args
    from dasklearn.settings_factory import session_options
    from dasklearn.simulation.peer_baselines import baseline_spec
    args = get_args(['--algorithm', 'gt-dsgd', '--local-steps', '1', '--momentum', '0',
                     '--gt-dsgd-topology', 'kreg', '--gt-dsgd-degree', '2'])
    s = GTDSGDSettings(**session_options(args, 'cpu'))
    assert s.topology == 'kreg' and s.degree == 2
    assert baseline_spec('gt-dsgd', 4, 4) == (GTDSGDSimulation, GTDSGDSettings, {})
