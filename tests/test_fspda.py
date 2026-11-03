"""Independent incidence-matrix oracle, randomized protocol, and wire accounting."""
import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn.optimizer.fspda import edge_seed, finish, pack
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.events import Event, INCOMING_MODEL
from dasklearn.simulation.fspda.settings import FSPDASettings
from dasklearn.simulation.fspda.simulation import FSPDASimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.references import TaskRef
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings
from test_classic_baselines import logistic, run, assert_model
from test_sgp import model_from, vector


def config(tmp_path, **kwargs):
    values = dict(algorithm='fspda', participants=4, rounds=8, duration=4 * MICROSECONDS,
                  test_period=0, learning=LearningSettings(.1, 0, .01, 64, 1), network_latency_ms=3,
                  keep_ratio=.5, consensus_stepsize=.2, dual_weight=.03, dual_stepsize=.4)
    values.update(kwargs)
    return settings(FSPDASettings, tmp_path, **values)


@pytest.mark.asyncio
@pytest.mark.parametrize('sampling', ['bernoulli', 'one-edge'])
@pytest.mark.parametrize('topology', ['ring', 'complete'])
@pytest.mark.parametrize('ratio', [.5, 1.])
@pytest.mark.parametrize('synchronous', [False, True])
async def test_independent_incidence_logistic_oracle(tmp_path, monkeypatch, logistic, sampling, topology, ratio, synchronous):
    import dasklearn.simulation.fspda.simulation as implementation
    initial, parts = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    s = config(tmp_path, sampling=sampling, topology=topology, keep_ratio=ratio, synchronous=synchronous)
    sim = run(FSPDASimulation, s, True)
    result = execute_local(sim)
    x = torch.stack([vector(initial)] * 4)
    dual = torch.zeros_like(x)
    traffic = 0
    for r in range(1, 9):
        laplacian = torch.zeros_like(x)
        for i, j in sim.edges(r):
            a = result[sim.clients[i].rounds[r]['outgoing'][j].task][0]
            b = result[sim.clients[j].rounds[r]['outgoing'][i].task][0]
            indices = a['indices'] if ratio < 1 else torch.arange(x.shape[1])
            if ratio < 1:
                assert torch.equal(indices, b['indices'])
                assert indices.dtype == torch.int64 and len(indices) == 4
            for packet, source in [(a, i), (b, j)]:
                torch.testing.assert_close(packet['values'], x[source, indices])
                assert sum(t.numel()*t.element_size() for t in packet.values() if t is not None) == sim.packet_size
            # Incidence form B^T C B X, constructed independently from node updates.
            incidence = torch.zeros(4)
            incidence[i], incidence[j] = 1., -1.
            edge_difference = incidence @ x
            projection = torch.zeros_like(edge_difference)
            projection[indices] = edge_difference[indices]
            laplacian += incidence[:, None] * projection
        grads = []
        for i, part in enumerate(parts):
            model = model_from(initial, x[i])
            torch.nn.functional.cross_entropy(model(part.features), part.labels).backward()
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]) + .01*x[i])
        grad = torch.stack(grads)
        next_x = x - .1*grad - .03*dual - .2*laplacian
        dual = dual + .4*laplacian
        torch.testing.assert_close(dual.sum(0), torch.zeros_like(dual[0]), atol=2e-7, rtol=0)
        torch.testing.assert_close(next_x.mean(0), x.mean(0) - .1*grad.mean(0), atol=2e-7, rtol=2e-5)
        for i, c in enumerate(sim.clients):
            state = c.rounds[r]
            assert state['done']
            actual, actual_dual = result[state['model'].task]
            assert_model(actual, model_from(initial, next_x[i]))
            torch.testing.assert_close(actual_dual, dual[i], rtol=3e-5, atol=3e-7)
            assert state['completed_time'] >= state['gradient_ready']
            for peer in sim.neighbors(i, r):
                assert state['inbox'].messages[peer] == sim.clients[peer].rounds[r]['outgoing'][i]
                assert state['completed_time'] >= sim.clients[peer].rounds[r]['started_time'] + 3000
            if r > 1:
                prev = c.rounds[r-1]
                assert state['base'] == prev['model'] and state['previous'] == prev['dual']
                assert state['started_time'] >= prev['completed_time']
            for ref in state['outgoing'].values():
                assert sim.workflow_dag.tasks[ref.task].data['model'] == state['base']
        traffic += 2 * len(sim.edges(r)) * (sim.packet_size + s.control_message_bytes + len(pickle.dumps({'round': r})))
        x = next_x
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == traffic
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == traffic


def test_isolation_keeps_dual_force_and_uses_old_model_for_ascent(tmp_path):
    s = config(tmp_path, keep_ratio=1)
    model = torch.nn.Linear(3, 2)
    x = vector(model)
    dual, grad = torch.arange(8)*.1, torch.ones(8)*.7
    original = copy.deepcopy(model.state_dict())
    local, next_dual = finish(s, dict(base=model, local=model, dual=dual, gradient=grad, packets=[]))
    torch.testing.assert_close(vector(local), x - .1*grad - .03*dual)
    torch.testing.assert_close(next_dual, dual, rtol=0, atol=0)
    packet = {'indices': torch.tensor([0, 3]), 'values': x[[0, 3]] + 1}
    local, next_dual = finish(s, dict(base=model, local=model, dual=dual, gradient=grad, packets=[packet]))
    expected = dual.clone()
    expected[[0, 3]] -= .4
    torch.testing.assert_close(next_dual, expected)
    for key, value in original.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(dual, torch.arange(8)*.1, rtol=0, atol=0)


@pytest.mark.asyncio
@pytest.mark.parametrize('latency', [0, 20])
async def test_exchange_overlaps_gradient_and_waits_for_both(tmp_path, latency):
    sim = run(FSPDASimulation, config(tmp_path, participants=2, rounds=1, edge_probability=1,
                                     keep_ratio=1, network_latency_ms=latency))
    for c in sim.clients:
        state = c.rounds[1]
        assert state['started_time'] == 0
        # One transfer per direction: full duplex, payload plus round/header.
        size = sim.packet_size + sim.settings.control_message_bytes + len(pickle.dumps({'round': 1}))
        transfer_time = int(size / 100000 * MICROSECONDS) + latency * 1000
        assert state['completed_time'] == max(c.get_train_time(), transfer_time)


@pytest.mark.asyncio
async def test_early_duplicate_invalid_and_isolated_rounds(tmp_path):
    sim = run(FSPDASimulation, config(tmp_path, sampling='one-edge'))
    c = sim.clients[0]
    before = len(sim.workflow_dag.tasks)
    r = next(r for r in range(1, 9) if sim.neighbors(0, r))
    for peer in sim.neighbors(0, r):
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
            {'from': peer, 'model': c.rounds[r]['inbox'].messages[peer], 'metadata': {'round': r}}))
    c.finish_gradient(Event(sim.current_time, 0, 'finish_gradient', {'params': {'round': r}}))
    future = next(r for r in range(10, 20) if sim.neighbors(0, r))
    peer = sim.neighbors(0, future)[0]
    c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
        {'from': peer, 'model': TaskRef('early'), 'metadata': {'round': future}}))
    assert not c.rounds[future]['done'] and len(sim.workflow_dag.tasks) == before
    with pytest.raises(ValueError, match='Unexpected sender'):
        c.on_incoming_model(Event(sim.current_time, 0, INCOMING_MODEL,
            {'from': 0, 'model': TaskRef('bad'), 'metadata': {'round': future}}))
    isolated = [s for client in sim.clients for r, s in client.rounds.items() if not sim.neighbors(client.index, r)]
    assert isolated and all(s['done'] and s['completed_time'] == s['gradient_ready'] for s in isolated)
    for bad in [0, -1, True, 1.2]:
        with pytest.raises(ValueError, match='Invalid FSPDA round'):
            c.on_incoming_model(Event(0, 0, INCOMING_MODEL, {'metadata': {'round': bad}}))


@pytest.mark.asyncio
@pytest.mark.parametrize('n,topology', [(1, 'ring'), (2, 'ring'), (5, 'complete'), (6, 'kreg')])
async def test_streaming_snapshots_and_small_graphs(tmp_path, n, topology):
    sim = run(FSPDASimulation, config(tmp_path, participants=n, topology=topology, degree=2,
                                      test_period=MICROSECONDS))
    execute_local(sim, retain_results=False)
    sim.merge_accuracies_files()
    frame = pd.read_csv(sim.data_dir + '/accuracies.csv')
    assert len(frame) == 4*n and frame.groupby('time').peer.nunique().eq(n).all()
    assert frame.loc[frame.time == 4, 'round'].eq(8).all()
    assert np.isfinite(frame[['accuracy', 'loss']]).all().all()
    if n == 1:
        assert sim.clients[0].bw_scheduler.total_bytes_sent == 0
        assert sim.clients[0].rounds[8]['completed_time'] == 8*sim.clients[0].get_train_time()


@pytest.mark.asyncio
async def test_duration_stop_snapshots_only_publish_completed_updates(tmp_path):
    sim = run(FSPDASimulation, config(tmp_path, rounds=1, stop='duration', duration=MICROSECONDS))
    for c in sim.clients:
        assert sum(s['done'] for s in c.rounds.values()) > 1
        r, model = sim.latest_models[c.index]
        assert c.rounds[r]['done'] and c.rounds[r]['model'] == model
        assert c.rounds[r]['completed_time'] <= MICROSECONDS
    execute_local(sim, retain_results=False)


@pytest.mark.asyncio
async def test_reproducible_schedule_and_serialized_replay(tmp_path):
    from dasklearn.tasks.dag import WorkflowDAG
    frames, dags = [], []
    for repeat in range(2):
        Task.COUNTER.clear()
        sim = run(FSPDASimulation, config(tmp_path / str(repeat), test_period=MICROSECONDS), True)
        dags.append(sim.workflow_dag.serialize())
        sim.workflow_dag = WorkflowDAG.load_from_file(sim.data_dir + '/workflow_dag')
        sim.workflow_dag.build_task_indices()
        sim.workflow_dag.check_validity()
        execute_local(sim, retain_results=False)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv'))
    assert dags[0] == dags[1]
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


def test_shared_masks_unscaled_positive_support_and_dense_payload(tmp_path):
    model = torch.nn.Linear(3, 2)
    s = config(tmp_path)
    counts = torch.zeros(8)
    for r in range(1, 101):
        a = pack(s, dict(model=model, peer=0, neighbor=1, round=r))[0]
        b = pack(s, dict(model=model, peer=1, neighbor=0, round=r))[0]
        assert torch.equal(a['indices'], b['indices'])
        torch.testing.assert_close(a['values'], vector(model)[a['indices']], rtol=0, atol=0)
        counts[a['indices']] += 1
    assert (counts > 20).all() and (counts < 80).all()
    assert edge_seed(3, 1, 0, 1, 'edge') != edge_seed(3, 1, 0, 1, 'coordinates')
    s.keep_ratio = 1
    packet = pack(s, dict(model=model, peer=0, neighbor=1, round=1))[0]
    assert packet['indices'] is None
    torch.testing.assert_close(packet['values'], vector(model), rtol=0, atol=0)


def test_fixed_graph_converges_to_uniform_quadratic_optimum(tmp_path):
    # Unequal Hessians make consensus at the correct optimizer distinguishable
    # from just averaging local optima, or discarding the dual as in DSGD.
    s = config(tmp_path, learning=LearningSettings(.05, 0, 0, 32, 1))
    models = [torch.nn.Linear(1, 1, bias=False).double() for _ in range(3)]
    for i, model in enumerate(models):
        model.weight.data.fill_(i + 2.)
    duals = [None] * 3
    a = torch.tensor([1., 2., 3.], dtype=torch.float64)
    b = torch.tensor([-2., 1., 3.], dtype=torch.float64)
    for _ in range(2000):
        values = [vector(m).clone() for m in models]
        results = [finish(s, dict(base=m, local=m, dual=duals[i], gradient=a[i]*(values[i]-b[i]),
            packets=[{'indices': None, 'values': values[j]} for j in range(3) if j != i]))
            for i, m in enumerate(models)]
        models, duals = map(list, zip(*results))
    expected = (a*b).sum()/a.sum()
    for model in models:
        torch.testing.assert_close(vector(model)[0], expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(sum(duals), torch.zeros_like(duals[0]), rtol=0, atol=1e-12)


def test_local_buffers_and_packet_inputs_are_immutable(tmp_path, monkeypatch):
    from dasklearn import functions
    from dasklearn.optimizer.fspda import gradient
    s = config(tmp_path)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    before = copy.deepcopy(model.state_dict())
    def oracle(settings, params):
        result = copy.deepcopy(params['model'])
        result(torch.ones(8, 3))
        result.gradient = [torch.ones_like(p) for p in result.parameters()]
        return [result]
    monkeypatch.setattr(functions, 'compute_gradient', oracle)
    local, grad = gradient(s, {'model': model})
    local_buffers = copy.deepcopy(dict(local.named_buffers()))
    packet = pack(s, dict(model=model, peer=0, neighbor=1, round=1))[0]
    packet_before = copy.deepcopy(packet)
    result, dual = finish(s, dict(base=model, local=local, dual=None, gradient=grad, packets=[packet]))
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    for key, value in local_buffers.items():
        torch.testing.assert_close(dict(result.named_buffers())[key], value, rtol=0, atol=0)
    for key, value in packet_before.items():
        torch.testing.assert_close(packet[key], value, rtol=0, atol=0)
    assert not dual.requires_grad and not grad.requires_grad


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
async def test_cpu_cuda_models_duals_and_packets(tmp_path):
    sim = run(FSPDASimulation, config(tmp_path), True)
    cpu = execute_local(sim)
    sim.settings.torch_device_name = 'cuda:0'
    gpu = execute_local(sim)
    for name, values in cpu.items():
        for a, b in zip(values, gpu[name]):
            if isinstance(a, torch.nn.Module):
                assert_model(a, b)
            elif isinstance(a, dict):
                for key, value in a.items():
                    if value is not None:
                        torch.testing.assert_close(value, b[key], rtol=4e-5, atol=5e-7)
            elif a is not None:
                torch.testing.assert_close(a, b, rtol=4e-5, atol=5e-7)


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [dict(topology='unknown'), dict(topology='kreg', degree=4),
    dict(topology='kreg', degree=0), dict(sampling='unknown'), dict(churn='synthetic'), dict(lora=True),
    dict(test_method='global'), dict(edge_probability=0), dict(edge_probability=float('nan')),
    dict(edge_probability=1.1), dict(keep_ratio=0), dict(keep_ratio=float('nan')), dict(keep_ratio=1.1),
    dict(dual_weight=0), dict(dual_weight=float('inf')), dict(dual_stepsize=-1), dict(dual_stepsize=float('nan')),
    dict(consensus_stepsize=0), dict(consensus_stepsize=float('inf')),
    dict(learning=LearningSettings(.1, .9, 0, 32, 1)), dict(learning=LearningSettings(.1, 0, 0, 32, 5)),
    dict(learning=LearningSettings(0, 0, 0, 32, 1)), dict(learning=LearningSettings(.1, 0, -.1, 32, 1))])
async def test_invalid_settings_fail_early(tmp_path, extra):
    with pytest.raises(ValueError):
        FSPDASimulation(config(tmp_path, **extra))


def test_cli_alias_and_broker_settings_roundtrip(tmp_path):
    from args import get_args
    from dasklearn.algorithms import settings_classes
    from dasklearn.settings_factory import session_options
    from dasklearn.simulation.peer_baselines import baseline_spec
    args = get_args(['--algorithm', 'fspda', '--local-steps', '1', '--momentum', '0',
        '--fspda-topology', 'kreg', '--fspda-degree', '2', '--fspda-sampling', 'one-edge',
        '--fspda-keep-ratio', '.3', '--fspda-consensus-stepsize', '.2', '--fspda-dual-weight', '.05',
        '--fspda-dual-stepsize', '.4'])
    s = FSPDASettings(**session_options(args, 'cpu'))
    assert (s.topology, s.degree, s.sampling, s.keep_ratio, s.consensus_stepsize, s.dual_weight, s.dual_stepsize) == ('kreg', 2, 'one-edge', .3, .2, .05, .4)
    broker_settings = config(tmp_path)
    assert settings_classes()['FSPDASettings'].from_json(broker_settings.to_json()) == broker_settings
    assert baseline_spec('fspda', 4, 4) == (FSPDASimulation, FSPDASettings, {})
