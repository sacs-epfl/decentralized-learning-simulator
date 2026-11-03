"""Independent base MoTEF recurrence, compressed memories and causal exchange."""
import copy
import math
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn.optimizer.motef import bootstrap, compress, decompress, gradient, propose, receive
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.events import Event, INCOMING_MODEL
from dasklearn.simulation.motef.settings import MoTEFSettings
from dasklearn.simulation.motef.simulation import MoTEFSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.references import TaskRef
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings
from test_classic_baselines import logistic, run, assert_model
from test_sgp import model_from, vector


def config(tmp_path, **kwargs):
    values = dict(algorithm='motef', participants=4, rounds=5, duration=4 * MICROSECONDS,
                  test_period=0, learning=LearningSettings(.1, 0, .01, 64, 1), network_latency_ms=3,
                  keep_ratio=.5, consensus_stepsize=.2, momentum_weight=.2)
    values.update(kwargs)
    return settings(MoTEFSettings, tmp_path, **values)


def independent_compress(value, compressor, ratio, seed):
    if compressor == 'identity':
        return value.clone()
    k = math.ceil(value.numel() * ratio)
    if compressor == 'topk':
        indices = torch.tensor(np.argsort(-np.abs(value.numpy()))[:k].copy())
    else:
        indices = torch.randperm(value.numel(), generator=torch.Generator().manual_seed(seed))[:k]
    result = torch.zeros_like(value)
    result[indices] = value[indices]
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('compressor', ['topk', 'randomk', 'identity'])
@pytest.mark.parametrize('topology', ['ring', 'complete'])
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.parametrize('momentum_weight', [.2, 1.])
async def test_independent_logistic_models_momentum_trackers_and_estimates(
        tmp_path, monkeypatch, logistic, compressor, topology, synchronous, momentum_weight):
    import dasklearn.simulation.motef.simulation as implementation
    initial, parts = logistic
    with torch.no_grad():
        for p in initial.parameters():
            p.add_(torch.arange(p.numel()).view_as(p) * .003 + .001)
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    s = config(tmp_path, compressor=compressor, topology=topology, synchronous=synchronous,
               momentum_weight=momentum_weight)
    sim = run(MoTEFSimulation, s, True)
    result = execute_local(sim)
    W = torch.tensor([[1 / (3 if topology == 'ring' else 4)
                      if topology == 'complete' or (i-j) % 4 in [0, 1, 3] else 0. for j in range(4)] for i in range(4)])
    def gradients(values):
        grads = []
        for i, part in enumerate(parts):
            model = model_from(initial, values[i])
            torch.nn.functional.cross_entropy(model(part.features), part.labels).backward()
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]) + .01 * values[i])
        return torch.stack(grads)
    x = torch.stack([vector(initial)] * 4)
    m = v = gradients(x)
    h = torch.zeros_like(x)
    g = torch.zeros_like(x)
    for i, c in enumerate(sim.clients):
        state = result[c.bootstrap_state.task][1]
        for key in ['h', 'g', 'h_memory', 'g_memory']:
            assert torch.count_nonzero(state[key]) == 0
        for key in ['m', 'v']:
            torch.testing.assert_close(state[key], m[i], rtol=2e-5, atol=3e-7)
        assert c.bootstrap_ready == c.get_train_time()
    for r in range(1, 6):
        next_x = x + .2 * (W @ h - h) - .1 * v
        grad = gradients(next_x)
        next_m = (1 - momentum_weight) * m + momentum_weight * grad
        next_v = v + .2 * (W @ g - g) + next_m - m
        qh, qg = [], []
        for i in range(4):
            seed = (s.seed + 1000003 * i + 9176 * r) % 2**63
            qh.append(independent_compress(next_x[i] - h[i], compressor, .5, (seed + 13) % 2**63))
            qg.append(independent_compress(next_v[i] - g[i], compressor, .5, (seed + 29) % 2**63))
        next_h, next_g = h + torch.stack(qh), g + torch.stack(qg)
        torch.testing.assert_close(next_v.mean(0), next_m.mean(0), rtol=2e-5, atol=3e-7)
        torch.testing.assert_close(next_x.mean(0), x.mean(0) - .1 * m.mean(0), rtol=2e-5, atol=3e-7)
        for i, c in enumerate(sim.clients):
            state = c.rounds[r]
            assert state['done']
            assert_model(result[state['model'].task][0], model_from(initial, next_x[i]))
            actual = result[state['state'].task][0]
            for key, expected in [('m', next_m), ('v', next_v), ('h', next_h), ('g', next_g),
                                  ('h_memory', W @ next_h), ('g_memory', W @ next_g)]:
                torch.testing.assert_close(actual[key], expected[i], rtol=3e-5, atol=5e-7)
            packet = result[state['packet'].task][2]
            for channel, expected in [('model', qh[i]), ('tracker', qg[i])]:
                decoded = torch.zeros_like(expected)
                if packet[channel]['indices'] is None:
                    decoded = packet[channel]['values']
                else:
                    decoded[packet[channel]['indices']] = packet[channel]['values']
                torch.testing.assert_close(decoded, expected, rtol=3e-5, atol=5e-7)
            predecessor = c.bootstrap_state if r == 1 else c.rounds[r - 1]['state']
            assert sim.workflow_dag.tasks[state['proposal'].task].data['state'] == predecessor
            assert sim.workflow_dag.tasks[state['packet'].task].data['model'] == state['proposal']
        x, m, v, h, g = next_x, next_m, next_v, next_h, next_g
    edges = 4 * (2 if topology == 'ring' else 3)
    expected_bytes = sum(edges * (sim.packet_size + s.control_message_bytes + len(pickle.dumps({'round': r}))) for r in range(1, 6))
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == expected_bytes
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == expected_bytes
    assert all(sum(c.incoming_counter.values()) == 5 * (edges // 4) for c in sim.clients)


@pytest.mark.asyncio
async def test_changing_oracle_uses_next_model_and_exact_previous_momentum(tmp_path, monkeypatch, logistic):
    from dasklearn import functions
    import dasklearn.simulation.motef.simulation as implementation
    initial, _ = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    noise = torch.randn(6, 4, 8, generator=torch.Generator().manual_seed(124)) * .1
    scales = torch.tensor([.5, 1., 2., 3.]).view(4, 1)
    def oracle(s, params):
        model = copy.deepcopy(params['model'])
        grad = scales[params['peer']] * vector(model) + noise[params['round'], params['peer']]
        model.gradient = [grad[:6].view(2, 3), grad[6:]]
        return [model]
    monkeypatch.setattr(functions, 'compute_gradient', oracle)
    sim = run(MoTEFSimulation, config(tmp_path, compressor='identity', learning=LearningSettings(.1, 0, 0, 64, 1)), True)
    result = execute_local(sim)
    W = torch.tensor([[1 / 3 if (i-j) % 4 in [0, 1, 3] else 0. for j in range(4)] for i in range(4)])
    x = torch.stack([vector(initial)] * 4)
    m = v = scales * x + noise[0]
    h, g = torch.zeros_like(x), torch.zeros_like(x)
    for r in range(1, 6):
        x = x + .2 * (W @ h - h) - .1 * v
        new_m = .8 * m + .2 * (scales * x + noise[r])
        v = v + .2 * (W @ g - g) + new_m - m
        m, h, g = new_m, x.clone(), v.clone()
        for i, c in enumerate(sim.clients):
            assert_model(result[c.rounds[r]['model'].task][0], model_from(initial, x[i]))
            state = result[c.rounds[r]['state'].task][0]
            torch.testing.assert_close(state['m'], m[i], rtol=2e-5, atol=3e-7)
            torch.testing.assert_close(state['v'], v[i], rtol=2e-5, atol=3e-7)


@pytest.mark.parametrize('compressor', ['topk', 'randomk', 'identity'])
@pytest.mark.parametrize('ratio', [.01, .4, 1.])
@pytest.mark.asyncio
async def test_global_compression_contract_and_exact_payload(tmp_path, compressor, ratio):
    value = torch.tensor([1., -9., 3., -7., 5., -2., 0.])
    packet = compress(value, compressor, ratio, 17)
    decoded = decompress(packet, len(value))
    torch.testing.assert_close(value, torch.tensor([1., -9., 3., -7., 5., -2., 0.]), rtol=0, atol=0)
    if compressor == 'topk':
        np.testing.assert_array_equal(np.sort(packet['indices'].numpy()), np.sort(np.argsort(-np.abs(value.numpy()))[:math.ceil(7*ratio)]))
        assert (value - decoded).square().sum() <= (1 - math.ceil(7*ratio)/7) * value.square().sum() + 1e-6
    if compressor == 'identity':
        torch.testing.assert_close(decoded, value, rtol=0, atol=0)
    else:
        assert packet['indices'].dtype == torch.int64 and len(packet['values']) == math.ceil(7*ratio)
    s = config(tmp_path, compressor=compressor, keep_ratio=ratio)
    sim = MoTEFSimulation(s)
    payload = compress(torch.zeros(sim.parameter_count), compressor, ratio, 17)
    assert sim.packet_size == 2 * sum(t.numel() * t.element_size() for t in payload.values() if t is not None)


@pytest.mark.asyncio
async def test_bootstrap_and_same_iteration_deliveries_are_charged(tmp_path):
    sim = run(MoTEFSimulation, config(tmp_path, rounds=2, network_latency_ms=20), True)
    for c in sim.clients:
        first = sim.workflow_dag.tasks[c.rounds[1]['proposal'].task]
        assert first.data['time'] >= c.bootstrap_ready > 0
        for r, state in c.rounds.items():
            assert state['completed_time'] >= state['gradient_ready']
            for peer in sim.neighbors(c.index):
                assert state['completed_time'] >= sim.clients[peer].rounds[r]['gradient_ready'] + 20000
            if r > 1:
                assert sim.workflow_dag.tasks[state['proposal'].task].data['time'] >= c.rounds[r - 1]['completed_time']
    assert sim.clients[0].rounds[1]['completed_time'] > sim.clients[0].rounds[1]['gradient_ready']


@pytest.mark.asyncio
async def test_early_duplicate_messages_and_completion_guards(tmp_path):
    sim = run(MoTEFSimulation, config(tmp_path))
    c = sim.clients[0]
    before = len(sim.workflow_dag.tasks)
    for peer in sim.neighbors(0):
        packet = c.rounds[1]['inbox'].messages[peer]
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL, {'from': peer, 'model': packet, 'metadata': {'round': 1}}))
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL, {'from': peer, 'model': TaskRef('early'), 'metadata': {'round': 9}}))
    c.finish_gradient(Event(sim.current_time, 0, 'finish_gradient', {'params': {'round': 1}}))
    c.finish_bootstrap(Event(sim.current_time, 0, 'finish_bootstrap', {}))
    assert not c.rounds[9]['done'] and not c.rounds[9]['inbox'].completed
    assert len(sim.workflow_dag.tasks) == before
    with pytest.raises(ValueError, match='Unexpected sender'):
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL, {'from': 2, 'model': TaskRef('bad'), 'metadata': {'round': 9}}))


@pytest.mark.asyncio
@pytest.mark.parametrize('n,topology', [(1, 'ring'), (2, 'ring'), (5, 'ring'), (5, 'complete'), (6, 'kreg')])
async def test_small_graphs_streaming_snapshots_and_duration_stop(tmp_path, n, topology):
    sim = run(MoTEFSimulation, config(tmp_path, participants=n, topology=topology, degree=2, test_period=MICROSECONDS))
    execute_local(sim, retain_results=False)
    sim.merge_accuracies_files()
    frame = pd.read_csv(sim.data_dir + '/accuracies.csv')
    assert len(frame) == 4 * n and frame.groupby('time').peer.nunique().eq(n).all()
    assert frame.loc[frame.time == 4, 'round'].eq(5).all()
    assert np.isfinite(frame[['accuracy', 'loss']]).all().all()
    if n == 1:
        assert sim.clients[0].bw_scheduler.total_bytes_sent == 0


@pytest.mark.asyncio
async def test_single_worker_matches_ema_sgd_not_shared_momentum(tmp_path, monkeypatch, logistic):
    import dasklearn.simulation.motef.simulation as implementation
    initial, parts = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    sim = run(MoTEFSimulation, config(tmp_path, participants=1, learning=LearningSettings(.1, 0, 0, 64, 1)))
    result = execute_local(sim)
    model = copy.deepcopy(initial)
    def grad(m):
        m.zero_grad()
        torch.nn.functional.cross_entropy(m(parts[0].features), parts[0].labels).backward()
        return torch.cat([p.grad.flatten() for p in m.parameters()])
    momentum = grad(model)
    for r in range(1, 6):
        model = model_from(initial, vector(model) - .1 * momentum)
        momentum = .8 * momentum + .2 * grad(model)
        assert_model(result[sim.clients[0].rounds[r]['model'].task][0], model)


@pytest.mark.asyncio
async def test_seeded_reproducibility_and_saved_state_replay(tmp_path):
    from dasklearn.tasks.dag import WorkflowDAG
    frames, dags = [], []
    for repeat in range(2):
        Task.COUNTER.clear()
        sim = run(MoTEFSimulation, config(tmp_path / str(repeat), compressor='randomk', test_period=MICROSECONDS), True)
        dags.append(sim.workflow_dag.serialize())
        sim.workflow_dag = WorkflowDAG.load_from_file(sim.data_dir + '/workflow_dag')
        sim.workflow_dag.build_task_indices()
        sim.workflow_dag.check_validity()
        execute_local(sim, retain_results=False)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv'))
    assert dags[0] == dags[1]
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


def test_immutable_inputs_and_preserved_local_buffers(tmp_path, monkeypatch):
    from dasklearn import functions
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    model(torch.randn(8, 3))
    original = copy.deepcopy(model.state_dict())
    def oracle(s, params):
        result = copy.deepcopy(params['model'])
        result.gradient = [torch.ones_like(p) for p in result.parameters()]
        return [result]
    monkeypatch.setattr(functions, 'compute_gradient', oracle)
    s = config(tmp_path)
    params = {'model': model, 'peer': 0, 'round': 0, 'time': 0}
    local, old = bootstrap(s, params)
    snapshot = copy.deepcopy(old)
    proposal, qh = propose(s, {**params, 'model': local, 'state': old, 'round': 1})
    qh_copy = copy.deepcopy(qh)
    local, partial, packet = gradient(s, {**params, 'model': proposal, 'state': old, 'model_packet': qh, 'round': 1})
    result = receive(s, {'state': old, 'local_state': partial, 'packets': [packet], 'weights': [1.]})[0]
    for key in old:
        torch.testing.assert_close(old[key], snapshot[key], rtol=0, atol=0)
    for key, value in qh_copy.items():
        if value is not None:
            torch.testing.assert_close(qh[key], value, rtol=0, atol=0)
    for name, value in original.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    for name, value in model.named_buffers():
        torch.testing.assert_close(dict(local.named_buffers())[name], value, rtol=0, atol=0)
    assert all(not t.requires_grad for t in result.values())


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
async def test_cpu_cuda_all_state_and_compressed_values_match(tmp_path):
    sim = run(MoTEFSimulation, config(tmp_path), True)
    cpu = execute_local(sim)
    sim.settings.torch_device_name = 'cuda:0'
    gpu = execute_local(sim)
    def check(a, b):
        if isinstance(a, torch.nn.Module):
            assert_model(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                check(a[key], b[key])
        elif a is not None:
            torch.testing.assert_close(a, b, rtol=4e-5, atol=5e-7)
    for name, values in cpu.items():
        for a, b in zip(values, gpu[name]):
            check(a, b)


@pytest.mark.asyncio
async def test_duration_stop_ignores_round_cap_and_hides_incomplete_proposal(tmp_path):
    sim = run(MoTEFSimulation, config(tmp_path, rounds=1, stop='duration', duration=MICROSECONDS))
    for c in sim.clients:
        assert sum(state['done'] for state in c.rounds.values()) > 1
        r, model = sim.latest_models[c.index]
        assert c.rounds[r]['done'] and c.rounds[r]['model'] == model
        assert c.rounds[r]['completed_time'] <= MICROSECONDS
    execute_local(sim, retain_results=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [dict(topology='unknown'), dict(topology='kreg', degree=4),
    dict(churn='synthetic'), dict(lora=True), dict(test_method='global'), dict(compressor='bad'),
    dict(keep_ratio=0), dict(keep_ratio=float('nan')), dict(keep_ratio=1.1),
    dict(momentum_weight=0), dict(momentum_weight=float('inf')), dict(momentum_weight=1.1),
    dict(consensus_stepsize=0), dict(consensus_stepsize=1.1),
    dict(learning=LearningSettings(.1, .9, 0, 32, 1)), dict(learning=LearningSettings(.1, 0, 0, 32, 5)),
    dict(learning=LearningSettings(0, 0, 0, 32, 1)), dict(learning=LearningSettings(.1, 0, -.1, 32, 1))])
async def test_unsupported_settings_fail_early(tmp_path, extra):
    with pytest.raises(ValueError):
        MoTEFSimulation(config(tmp_path, **extra))


def test_cli_and_comparison_alias():
    from args import get_args
    from dasklearn.settings_factory import session_options
    from dasklearn.simulation.peer_baselines import baseline_spec
    args = get_args(['--algorithm', 'motef', '--local-steps', '1', '--momentum', '0',
        '--motef-topology', 'kreg', '--motef-degree', '2', '--motef-compressor', 'randomk',
        '--motef-keep-ratio', '.3', '--motef-consensus-stepsize', '.2', '--motef-momentum-weight', '.05'])
    s = MoTEFSettings(**session_options(args, 'cpu'))
    assert (s.topology, s.degree, s.compressor, s.keep_ratio, s.consensus_stepsize, s.momentum_weight) == ('kreg', 2, 'randomk', .3, .2, .05)
    assert baseline_spec('motef', 4, 4) == (MoTEFSimulation, MoTEFSettings, {})
