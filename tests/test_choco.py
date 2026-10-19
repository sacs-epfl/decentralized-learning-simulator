"""CHOCO's persistent hats, causal round exchange and ICML update equations."""
import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn import functions
from dasklearn.optimizer.choco import compress, decompress, payload_bytes, prepare, receive
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.choco.settings import ChocoSettings
from dasklearn.simulation.choco.simulation import ChocoSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings
from test_classic_baselines import logistic, local_sgd, run, assert_model


def config(tmp_path, **kwargs):
    values = dict(algorithm='choco', participants=4, rounds=5, duration=3*MICROSECONDS,
                  test_period=0, learning=LearningSettings(.1, 0, 0, 64, 1),
                  model_size_bytes=1024, network_latency_ms=3, keep_ratio=.5, consensus_stepsize=.2)
    values.update(kwargs)
    return settings(ChocoSettings, tmp_path, **values)


def flatten(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])


def set_flat(model, vector):
    result = copy.deepcopy(model)
    offset = 0
    with torch.no_grad():
        for p in result.parameters():
            p.copy_(vector[offset:offset+p.numel()].view_as(p))
            offset += p.numel()
    return result


@pytest.mark.parametrize('compressor', ['topk', 'randomk', 'identity'])
@pytest.mark.parametrize('ratio', [.01, .5, 1.])
def test_compression_payload_and_reconstruction(compressor, ratio):
    vector = torch.tensor([1., -8., 3., 2., 7., -.1, 5.])
    packet = compress(vector, [5, 2], compressor, ratio, seed=17)
    decoded = decompress(packet, len(vector))
    if compressor == 'identity':
        assert packet['indices'] is None
        assert torch.equal(decoded, vector)
    else:
        count = max(1, int(5*ratio)) + max(1, int(2*ratio))
        assert packet['indices'].dtype == torch.int64
        assert len(packet['values']) == count
        assert torch.equal(decoded[packet['indices']], vector[packet['indices']])
        assert torch.count_nonzero(decoded) == count
        if compressor == 'topk':
            for start, size in [(0, 5), (5, 2)]:
                order = np.argsort(-np.abs(vector.numpy()[start:start+size]))[:max(1, int(size*ratio))]
                assert set(packet['indices'].tolist()).intersection(range(start, start+size)) == set(start+order)
    model = torch.nn.Linear(5, 1)  # Two tensors with 5 and 1 values.
    actual = compress(flatten(model), [5, 1], compressor, ratio, 17)
    nbytes = sum(t.numel()*t.element_size() for t in actual.values() if t is not None)
    assert nbytes == payload_bytes(model, compressor, ratio)
    assert torch.equal(packet['values'], compress(vector, [5, 2], compressor, ratio, 17)['values'])


@pytest.mark.asyncio
@pytest.mark.parametrize('compressor', ['topk', 'randomk', 'identity'])
@pytest.mark.parametrize('steps', [1, 3])
@pytest.mark.parametrize('sync', [False, True])
async def test_multiround_independent_logistic_reference(tmp_path, monkeypatch, logistic, compressor, steps, sync):
    import dasklearn.simulation.choco.simulation as implementation
    initial, partitions = logistic
    # Break equal-magnitude pairs so independent NumPy/PyTorch Top-k choices
    # have one unique answer rather than different valid tie resolutions.
    with torch.no_grad():
        for p in initial.parameters():
            p.add_(torch.arange(p.numel()).reshape_as(p)*.003+.001)
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    s = config(tmp_path, compressor=compressor, synchronous=sync,
               learning=LearningSettings(.1, 0, 0, 64, steps))
    sim = run(ChocoSimulation, s, skew=True)
    actual = execute_local(sim)
    w = torch.tensor([[sim.weight(i,j) if j==i or sim.graph.has_edge(i,j) else 0.
                       for j in range(4)] for i in range(4)])
    torch.testing.assert_close(w.sum(0), torch.ones(4))
    torch.testing.assert_close(w, w.T)
    models = [copy.deepcopy(initial) for _ in range(4)]
    hats = torch.zeros(4, len(flatten(initial)))
    for r in range(1, 6):
        local = torch.stack([flatten(local_sgd(m, p, .1, steps)) for m,p in zip(models, partitions)])
        updated = local + .2 * (w @ hats - hats)
        # Consensus must preserve the mean of locally trained parameters even
        # though the public estimates are incomplete.
        torch.testing.assert_close(updated.mean(0), local.mean(0))
        next_hats = hats.clone()
        for peer in range(4):
            residual = updated[peer] - hats[peer]
            q = torch.zeros_like(residual)
            if compressor == 'identity':
                q = residual.clone()
            else:
                generator = torch.Generator().manual_seed((s.seed+1000003*peer+9176*r) % 2**63)
                offset = 0
                for p in initial.parameters():
                    count = max(1, int(p.numel()*.5))
                    if compressor == 'topk':
                        # Independent numpy selection; fixture avoids equal magnitudes.
                        indices = np.argsort(-np.abs(residual[offset:offset+p.numel()].numpy()))[:count]
                        indices = torch.tensor(indices.copy())
                    else:
                        indices = torch.randperm(p.numel(), generator=generator)[:count]
                    q[offset+indices] = residual[offset+indices]
                    offset += p.numel()
            next_hats[peer] += q
            state = sim.clients[peer].rounds[r]
            assert_model(actual[state['model'][0]][0], set_flat(initial, updated[peer]))
        hats = next_hats
        for peer in range(4):
            state = actual[sim.clients[peer].rounds[r]['state'][0]][0]
            torch.testing.assert_close(state['own'], hats[peer], rtol=2e-5, atol=2e-7)
            torch.testing.assert_close(state['memory'], (w @ hats)[peer], rtol=2e-5, atol=2e-7)
        models = [set_flat(initial, updated[peer]) for peer in range(4)]
    if compressor != 'identity':
        assert not torch.allclose(hats, torch.stack([flatten(m) for m in models]))
    edges = sim.graph.number_of_edges()*2
    expected_bytes = 0
    for r in range(1,6):
        metadata = {'round':r, 'phase':'choco-delta'}
        expected_bytes += edges*(sim.packet_size+s.control_message_bytes+len(pickle.dumps(metadata)))
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == expected_bytes
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == expected_bytes
    assert all(c.rounds[5]['done'] for c in sim.clients)


@pytest.mark.asyncio
async def test_identity_limit_is_old_model_mix_minus_own_gradient(tmp_path, monkeypatch, logistic):
    import dasklearn.simulation.choco.simulation as implementation
    initial, parts = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    sim = run(ChocoSimulation, config(tmp_path, compressor='identity', consensus_stepsize=1), True)
    result = execute_local(sim)
    states = [copy.deepcopy(initial) for _ in parts]
    for r in range(1,6):
        old = torch.stack([flatten(m) for m in states])
        own_updates = torch.stack([flatten(local_sgd(m,p,.1,1))-flatten(m) for m,p in zip(states,parts)])
        mixed = torch.stack([sum(sim.weight(i,j)*old[j] for j in [i]+sim.neighbors(i)) for i in range(4)])
        expected = mixed+own_updates
        for peer in range(4):
            assert_model(result[sim.clients[peer].rounds[r]['model'][0]][0], set_flat(initial,expected[peer]))
        states = [set_flat(initial,v) for v in expected]


def test_error_memory_persistence_and_inputs_are_immutable():
    model = torch.nn.Linear(4,2)
    initial = copy.deepcopy(model.state_dict())
    corrected, packet = prepare(model, None, 'topk', .25, .1, 0)
    state = receive(None,[packet],[1.],10)
    before = copy.deepcopy(state)
    second, packet2 = prepare(corrected,state,'topk',.25,.1,1)
    q = decompress(packet,10)
    residual = flatten(corrected)-q
    torch.testing.assert_close(decompress(packet2,10), decompress(compress(residual,[8,2],'topk',.25,1),10))
    for key in initial:
        assert torch.equal(model.state_dict()[key],initial[key])
    for key in state:
        assert torch.equal(state[key],before[key])
    assert all(not p.requires_grad for p in [state['own'],state['memory']])
    # A single worker performs no consensus correction.
    assert_model(second,model)


@pytest.mark.asyncio
@pytest.mark.parametrize('n,topology', [(1,'ring'),(2,'ring'),(4,'complete'),(6,'kreg')])
async def test_topologies_and_fixed_snapshots_execute(tmp_path,n,topology):
    s = config(tmp_path, participants=n, topology=topology, degree=2,
               test_period=MICROSECONDS, duration=3*MICROSECONDS, keep_ratio=.1)
    sim = run(ChocoSimulation,s)
    assert all(i not in sim.neighbors(i) for i in sim.graph)
    execute_local(sim,retain_results=False)
    sim.merge_accuracies_files()
    frame = pd.read_csv(sim.data_dir+'/accuracies.csv')
    assert len(frame)==n*3 and frame.groupby('time').peer.nunique().eq(n).all()
    assert np.isfinite(frame[['accuracy','loss']]).all().all()
    assert frame['round'].eq(5).all()


@pytest.mark.asyncio
async def test_reproducible_dag_and_metrics(tmp_path):
    dags,frames=[],[]
    for repeat in range(2):
        Task.COUNTER.clear()
        sim=run(ChocoSimulation,config(tmp_path/str(repeat),test_period=MICROSECONDS,compressor='randomk'),True)
        dags.append(sim.workflow_dag.serialize())
        execute_local(sim,retain_results=False);sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir+'/accuracies.csv'))
    assert dags[0]==dags[1]
    pd.testing.assert_frame_equal(frames[0],frames[1],check_exact=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA unavailable'))])
async def test_optimized_executor_preserves_models_packets_and_private_state(tmp_path,device):
    from dasklearn.tasks.training import TrainingExecutor
    s=config(tmp_path)
    s.torch_device_name=device
    sim=run(ChocoSimulation,s,True)
    s.gpu_residency=s.training_batching=False
    expected=execute_local(sim)
    s.gpu_residency=s.training_batching=True
    backend=TrainingExecutor(s,autotune=False)
    actual=execute_local(sim,training_backend=backend)
    for name,values in actual.items():
        for a,b in zip(values,expected[name]):
            if isinstance(a,torch.nn.Module):assert_model(a,b)
            else:
                for key in a:
                    if a[key] is None:assert b[key] is None
                    else:torch.testing.assert_close(a[key],b[key],rtol=2e-5,atol=2e-7)
    assert backend.stats['batched_tasks']>0


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [dict(keep_ratio=0),dict(keep_ratio=float('nan')),
    dict(consensus_stepsize=0),dict(consensus_stepsize=float('nan')),dict(compressor='bogus'),
    dict(churn='synthetic'),dict(test_method='global'),dict(topology='directed'),
    dict(topology='kreg',degree=4),dict(topology='kreg',degree=0)])
async def test_unsupported_settings_rejected(tmp_path,extra):
    with pytest.raises(ValueError):ChocoSimulation(config(tmp_path,**extra))


def test_convolution_parameters_compressed_but_buffers_and_frozen_parameters_stay_local():
    model=torch.nn.Sequential(torch.nn.Conv2d(1,2,3),torch.nn.BatchNorm2d(2),
                              torch.nn.Flatten(),torch.nn.Linear(8,2))
    model[0].bias.requires_grad_(False)
    model.train();model(torch.randn(4,1,4,4))
    before=copy.deepcopy(model.state_dict())
    count=len(flatten(model))
    state={'own':torch.zeros(count),'memory':torch.full((count,),.1)}
    corrected,packet=prepare(model,state,'topk',.2,.3,0)
    torch.testing.assert_close(flatten(corrected),flatten(model)+.03)
    for key in ['0.bias','1.running_mean','1.running_var','1.num_batches_tracked']:
        assert torch.equal(corrected.state_dict()[key],before[key])
    assert payload_bytes(model,'topk',.2)==sum(t.numel()*t.element_size() for t in packet.values() if t is not None)
    for key in before:assert torch.equal(model.state_dict()[key],before[key])


@pytest.mark.asyncio
async def test_private_state_waits_for_all_same_round_neighbors_under_skew(tmp_path):
    sim=run(ChocoSimulation,config(tmp_path),True)
    tasks=sim.workflow_dag.tasks
    delayed=False
    for client in sim.clients:
        for r, state in client.rounds.items():
            if not state['done']:continue
            received=tasks[state['state'][0]]
            assert len(received.data['packets'])==1+len(sim.neighbors(client.index))
            for ref in received.data['packets']:
                source=tasks[ref[0]]
                assert source.func=='choco_prepare' and source.data['round']==r
                assert source.data['time']<=state['completed_time']
                delayed |= source.data['time']<state['completed_time']
            if r>1:
                trained=next(t for t in tasks.values() if t.func=='train' and
                             t.data['peer']==client.index and t.data['round']==r)
                assert trained.data['time']>=client.rounds[r-1]['completed_time']
                assert received.data['state']==client.rounds[r-1]['state']
    assert delayed


@pytest.mark.asyncio
async def test_momentum_is_rejected_instead_of_silently_resetting_it(tmp_path):
    with pytest.raises(ValueError,match='momentum'):
        ChocoSimulation(config(tmp_path,learning=LearningSettings(.1,.9,0,32,1)))
