"""Independent Algorithm 2 trajectories, coupled sampling, and protocol costs."""
import copy
import json
import pickle
import random

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from dasklearn import functions
from dasklearn.optimizer.motef_vr import bootstrap, gradient
from dasklearn.optimizer.motef import propose, receive
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.motef_vr.settings import MoTEFVRSettings
from dasklearn.simulation.motef_vr.simulation import MoTEFVRSimulation
from dasklearn.tasks.dag import WorkflowDAG
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings
from test_classic_baselines import logistic, run, assert_model
from test_motef import independent_compress
from test_sgp import model_from, vector


def config(tmp_path, **kwargs):
    values = dict(algorithm='motef-vr', participants=4, rounds=5, duration=4*MICROSECONDS,
                  test_period=0, learning=LearningSettings(.1, 0, .01, 2, 1),
                  network_latency_ms=3, keep_ratio=.5, consensus_stepsize=.2, momentum_weight=.2)
    values.update(kwargs)
    return settings(MoTEFVRSettings, tmp_path, **values)


@pytest.mark.asyncio
@pytest.mark.parametrize('compressor', ['identity', 'topk', 'randomk'])
@pytest.mark.parametrize('topology', ['ring', 'complete'])
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.parametrize('lam', [.2, 1.])
async def test_changing_minibatch_reference_and_tracker_conservation(
        tmp_path, monkeypatch, logistic, compressor, topology, synchronous, lam):
    import dasklearn.simulation.motef.simulation as implementation
    initial, parts = logistic
    with torch.no_grad():
        for p in initial.parameters():
            p.add_(torch.arange(p.numel()).view_as(p)*.003+.001)
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    s = config(tmp_path, compressor=compressor, topology=topology, synchronous=synchronous, momentum_weight=lam)
    sim = run(MoTEFVRSimulation, s, True)
    results = execute_local(sim)
    W = torch.tensor([[1/(3 if topology == 'ring' else 4) if topology == 'complete' or
                       (i-j)%4 in [0, 1, 3] else 0. for j in range(4)] for i in range(4)])
    def grads(x, r):
        values = []
        for i, part in enumerate(parts):
            # Independent sampling and differentiation; never call the VR oracle.
            loader = DataLoader(part, batch_size=2, shuffle=True,
                                generator=torch.Generator().manual_seed(s.seed+1000003*i+9176*r))
            batch = next(iter(loader))
            model = model_from(initial, x[i])
            loss = torch.nn.functional.cross_entropy(model(batch['img']), batch['label'])
            loss.backward()
            values.append(torch.cat([p.grad.flatten() for p in model.parameters()]) + .01*x[i])
        return torch.stack(values)
    x = torch.stack([vector(initial)]*4)
    m = v = grads(x, 0)
    h, g = torch.zeros_like(x), torch.zeros_like(x)
    for i, c in enumerate(sim.clients):
        torch.testing.assert_close(results[c.bootstrap_state.task][1]['m'], m[i])
        assert c.bootstrap_ready == c.get_train_time()
    for r in range(1, 6):
        xn = x + .2*(W@h-h) - .1*v
        mn = grads(xn, r) + (1-lam)*(m-grads(x, r))
        vn = v + .2*(W@g-g) + mn-m
        qh, qg = [], []
        for i in range(4):
            seed = s.seed+1000003*i+9176*r
            qh.append(independent_compress(xn[i]-h[i], compressor, .5, seed+13))
            qg.append(independent_compress(vn[i]-g[i], compressor, .5, seed+29))
        hn, gn = h+torch.stack(qh), g+torch.stack(qg)
        torch.testing.assert_close(vn.mean(0), mn.mean(0), atol=5e-7, rtol=3e-5)
        torch.testing.assert_close(xn.mean(0), x.mean(0)-.1*m.mean(0), atol=5e-7, rtol=3e-5)
        for i, c in enumerate(sim.clients):
            state = c.rounds[r]
            assert_model(results[state['model'].task][0], model_from(initial, xn[i]))
            actual = results[state['state'].task][0]
            for key, expected in [('m',mn), ('v',vn), ('h',hn), ('g',gn),
                                  ('h_memory',W@hn), ('g_memory',W@gn)]:
                torch.testing.assert_close(actual[key], expected[i], atol=7e-7, rtol=4e-5)
            task = sim.workflow_dag.tasks[state['packet'].task]
            assert task.data['previous_model'] == state['base']
            assert task.data['model'] == state['proposal']
            start = sim.workflow_dag.tasks[state['proposal'].task].data['time']
            assert state['gradient_ready']-start == 2*c.get_train_time()
            for peer in sim.neighbors(i):
                assert state['completed_time'] >= sim.clients[peer].rounds[r]['gradient_ready']+3000
            packet = results[state['packet'].task][2]
            for channel, expected in [('model',qh[i]), ('tracker',qg[i])]:
                wire = packet[channel]
                decoded = torch.zeros_like(expected)
                if wire['indices'] is None:
                    decoded = wire['values']
                else:
                    decoded[wire['indices']] = wire['values']
                torch.testing.assert_close(decoded, expected, atol=7e-7, rtol=4e-5)
        x,m,v,h,g = xn,mn,vn,hn,gn
    edges = 4*(2 if topology == 'ring' else 3)
    traffic = sum(edges*(sim.packet_size+s.control_message_bytes+len(pickle.dumps({'round':r}))) for r in range(1,6))
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == traffic
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == traffic
    assert all(c.compute_time == 11*c.get_train_time() for c in sim.clients)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_paired_sampling_transforms_dropout_buffers_rng_and_immutability(tmp_path, monkeypatch, device):
    from types import SimpleNamespace
    from dasklearn.model_trainer import ModelTrainer
    calls, forwards = [], []
    class Partition:
        def __len__(self): return 17
        def __getitem__(self, i):
            calls.append(i)
            return {'img': torch.tensor([float(i), random.random(), np.random.random()]), 'label': i%2}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(3)
            self.linear = torch.nn.Linear(3, 2)
        def forward(self, x):
            mask = torch.nn.functional.dropout(torch.ones_like(x), .5, True)
            forwards.append((x.clone(), mask.clone(), random.random(), np.random.random()))
            return self.linear(self.bn(x)*mask)
    monkeypatch.setattr(ModelTrainer, 'prepare_partition', lambda self: setattr(self, 'partition', Partition()))
    monkeypatch.setattr(functions, 'dataset', SimpleNamespace())
    monkeypatch.setattr(functions, 'model_trainers', None)
    s = config(tmp_path, learning=LearningSettings(.1,0,.01,4,1), torch_device_name=device)
    model = Model()
    params = dict(model=model, peer=0, round=0, time=0)
    original = copy.deepcopy(model.state_dict())
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(0).clone() if device.startswith('cuda') else None
    local, old = bootstrap(s, params)
    assert len(calls) == 4
    old_copy = copy.deepcopy(old)
    proposal, qh = propose(s, dict(params, model=local, state=old, round=1))
    before = len(calls)
    result, partial, packet = gradient(s, dict(params, model=proposal, previous_model=local,
                                             state=old, model_packet=qh, round=1))
    assert len(calls)-before == 4  # once, not twice, including transforms
    a,b = forwards[-2:]
    for x,y in zip(a,b):
        if isinstance(x,torch.Tensor): torch.testing.assert_close(x,y,rtol=0,atol=0)
        else: assert x == y
    assert calls[:4] != calls[4:]
    assert result.bn.num_batches_tracked == local.bn.num_batches_tracked+1
    assert local.bn.num_batches_tracked == 1
    state = receive(s, {'state':old, 'local_state':partial, 'packets':[packet], 'weights':[1.]})[0]
    assert all(not t.requires_grad for t in state.values())
    for k in old: torch.testing.assert_close(old[k],old_copy[k],rtol=0,atol=0)
    for k in original: torch.testing.assert_close(model.state_dict()[k],original[k],rtol=0,atol=0)
    if cuda_state is not None:
        torch.testing.assert_close(torch.cuda.get_rng_state(0),cuda_state,rtol=0,atol=0)
    assert random.getstate() == py_state
    assert np.random.get_state()[0] == np_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    torch.testing.assert_close(torch.get_rng_state(),torch_state,rtol=0,atol=0)


@pytest.mark.asyncio
async def test_saved_dag_replay_streaming_completed_snapshots_and_reproducibility(tmp_path):
    frames, dags = [], []
    for repeat in range(2):
        Task.COUNTER.clear()
        sim = run(MoTEFVRSimulation, config(tmp_path/str(repeat), compressor='randomk',
                   keep_ratio=.01, stop='duration', rounds=1, duration=1000000, test_period=250000), True)
        for c in sim.clients:
            r, model = sim.latest_models[c.index]
            assert c.rounds[r]['done'] and c.rounds[r]['model'] == model
            assert c.rounds[r]['completed_time'] <= sim.settings.duration
        dags.append(sim.workflow_dag.serialize())
        sim.workflow_dag = WorkflowDAG.load_from_file(sim.data_dir+'/workflow_dag')
        sim.workflow_dag.build_task_indices()
        sim.workflow_dag.check_validity()
        execute_local(sim, retain_results=False)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir+'/accuracies.csv'))
        assert len(frames[-1]) == 16
        assert np.isfinite(frames[-1][['accuracy','loss']]).all().all()
        assert json.loads(open(sim.data_dir+'/motef-vr-mixing.json').read())['packet_size_bytes'] == sim.packet_size
    assert dags[0] == dags[1]
    pd.testing.assert_frame_equal(frames[0],frames[1],check_exact=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('n', [1,2])
async def test_small_graphs_and_single_client_vr(tmp_path,n):
    sim = run(MoTEFVRSimulation,config(tmp_path,participants=n))
    result = execute_local(sim)
    assert all(c.rounds[5]['done'] for c in sim.clients)
    if n == 1:
        assert sim.clients[0].bw_scheduler.total_bytes_sent == 0
        for r in range(1,6):
            c=sim.clients[0]
            torch.testing.assert_close(result[c.rounds[r]['state'].task][0]['m'],result[c.rounds[r]['state'].task][0]['v'])


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
async def test_cpu_cuda_states_match(tmp_path):
    sim=run(MoTEFVRSimulation,config(tmp_path),True)
    cpu=execute_local(sim)
    sim.settings.torch_device_name='cuda:0'
    gpu=execute_local(sim)
    def check(a,b):
        if isinstance(a,torch.nn.Module): assert_model(a,b)
        elif isinstance(a,dict):
            for k in a: check(a[k],b[k])
        elif a is not None: torch.testing.assert_close(a,b,rtol=4e-5,atol=7e-7)
    for name in cpu:
        for a,b in zip(cpu[name],gpu[name]): check(a,b)


def test_cli_serialization_and_runner(tmp_path):
    from args import get_args, create_parser
    from dasklearn.algorithms import class_paths, import_class
    from dasklearn.settings_factory import session_options
    from dasklearn.simulation.peer_baselines import baseline_spec
    from dasklearn.experiments import resolve_experiment
    args=get_args(['--algorithm','motef-vr','--local-steps','1','--momentum','0',
                   '--motef-momentum-weight','.3','--motef-compressor','randomk','--local',
                   '--synchronous','--compute-validation-loss-global-model','--profile',
                   '--log-bandwidth-utilization'])
    cls=import_class(class_paths('motef-vr')[0])
    s=cls(**session_options(args,'cpu'))
    restored=cls.from_json(s.to_json())
    assert restored.algorithm=='motef-vr' and restored.momentum_weight==.3
    assert restored.learning.local_steps==1 and restored.learning.momentum==0
    assert baseline_spec('motef-vr',4,4)==(MoTEFVRSimulation,MoTEFVRSettings,{})
    path=tmp_path/'experiment.json'
    path.write_text(json.dumps(dict(version=1,settings=dict(algorithm='motef-vr',local_steps=1,momentum=0,
                                      motef_compressor='randomk',motef_momentum_weight=.3,rounds=2))))
    experiment=resolve_experiment(path,create_parser())
    assert experiment.runs[0].options['algorithm']=='motef-vr'


@pytest.mark.asyncio
@pytest.mark.parametrize('learning', [LearningSettings(.1,.9,0,2,1), LearningSettings(.1,0,0,2,2)])
async def test_rejects_shared_momentum_and_multiple_steps(tmp_path,learning):
    with pytest.raises(ValueError):
        MoTEFVRSimulation(config(tmp_path,learning=learning))


@pytest.mark.asyncio
async def test_duplicate_early_messages_and_completed_model_dependencies(tmp_path):
    from dasklearn.simulation.events import Event, INCOMING_MODEL
    from dasklearn.tasks.references import TaskRef
    sim=run(MoTEFVRSimulation,config(tmp_path,keep_ratio=.01,test_period=MICROSECONDS))
    c=sim.clients[0]
    count=len(sim.workflow_dag.tasks)
    for peer in sim.neighbors(0):
        packet=c.rounds[1]['inbox'].messages[peer]
        for r,ref in [(1,packet),(9,TaskRef('early')),(9,TaskRef('early'))]:
            c.on_incoming_model(Event(sim.current_time,0,INCOMING_MODEL,
                                     {'from':peer,'model':ref,'metadata':{'round':r}}))
    c.finish_gradient(Event(sim.current_time,0,'finish_gradient',{'params':{'round':1}}))
    c.finish_bootstrap(Event(sim.current_time,0,'finish_bootstrap',{}))
    assert len(sim.workflow_dag.tasks)==count
    assert not c.rounds[9]['done']
    for task in sim.workflow_dag.tasks.values():
        if task.func=='test':
            client=sim.clients[task.data['peer']]
            r=task.data['round']
            if r:
                assert task.data['model']==client.rounds[r]['model']
                assert client.rounds[r]['completed_time'] <= task.data['time']


@pytest.mark.asyncio
async def test_snapshot_before_any_gradient_completion_evaluates_initialization(tmp_path):
    sim=run(MoTEFVRSimulation,config(tmp_path,duration=1000,test_period=500))
    assert not any(c.bootstrap_done for c in sim.clients)
    tasks=[t for t in sim.workflow_dag.tasks.values() if t.func=='test']
    assert tasks and all(t.data['round']==0 for t in tasks)
    assert all(sim.workflow_dag.tasks[t.data['model'].task].func=='initialize' for t in tasks)
    execute_local(sim,retain_results=False)


@pytest.mark.asyncio
async def test_local_saved_dag_runner_never_resimulates(tmp_path,monkeypatch):
    from args import get_args
    from dasklearn.local_runner import run_local
    sim=run(MoTEFVRSimulation,config(tmp_path,test_period=MICROSECONDS))
    original=sim.workflow_dag.serialize()
    args=get_args(['--local','--from-dir',sim.data_dir])
    assert args.local and args.from_dir==sim.data_dir
    sim.settings.from_dir=sim.data_dir
    def forbidden(*a,**kw):
        pytest.fail('Saved DAG replay regenerated the protocol')
    monkeypatch.setattr(sim,'simulate',forbidden)
    await run_local(sim)
    assert sim.workflow_dag.serialize()==original
    assert len(pd.read_csv(sim.data_dir+'/accuracies.csv'))==16


def test_rejects_devices_without_coupled_rng_support(tmp_path):
    with pytest.raises(ValueError,match='CPU or CUDA'):
        MoTEFVRSimulation(config(tmp_path,torch_device_name='mps'))
