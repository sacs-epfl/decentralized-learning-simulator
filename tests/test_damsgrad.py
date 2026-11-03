"""Independent Algorithm 3 trajectories, delayed denominator and causal exchange."""
import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn.optimizer.damsgrad import finish, gradient, initial_state, pack
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.damsgrad.settings import DAMSGradSettings
from dasklearn.simulation.damsgrad.simulation import DAMSGradSimulation
from dasklearn.simulation.events import Event, INCOMING_MODEL
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.references import TaskRef
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings
from test_classic_baselines import logistic, run, assert_model
from test_sgp import model_from, vector


def config(tmp_path, **kwargs):
    values = dict(algorithm='damsgrad', participants=4, rounds=6, duration=4*MICROSECONDS,
                  test_period=0, learning=LearningSettings(.01, 0, .01, 64, 1), network_latency_ms=3,
                  beta1=.7, beta2=.8, epsilon=.02)
    values.update(kwargs)
    return settings(DAMSGradSettings, tmp_path, **values)


@pytest.mark.asyncio
@pytest.mark.parametrize('topology', ['ring', 'complete'])
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.parametrize('betas', [(0., 0.), (.7, .8), (.9, .999)])
async def test_independent_logistic_models_all_moments_tracker_and_packets(
        tmp_path, monkeypatch, logistic, topology, synchronous, betas):
    import dasklearn.simulation.damsgrad.simulation as implementation
    initial, parts = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    s = config(tmp_path, topology=topology, synchronous=synchronous, beta1=betas[0], beta2=betas[1])
    sim = run(DAMSGradSimulation, s, True)
    result = execute_local(sim)
    W = np.array([[1/(3 if topology=='ring' else 4) if topology=='complete' or (i-j)%4 in [0,1,3]
                   else 0. for j in range(4)] for i in range(4)], dtype=np.float32)
    x = np.stack([vector(initial).numpy()]*4)
    m, v = np.zeros_like(x), np.zeros_like(x)
    vhat, u = np.full_like(x, .02), np.full_like(x, .02)
    for r in range(1, 7):
        grads = []
        for i, part in enumerate(parts):
            model = model_from(initial, torch.from_numpy(x[i]))
            torch.nn.functional.cross_entropy(model(part.features), part.labels).backward()
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]).numpy() + .01*x[i])
        grads = np.stack(grads)
        mixed_u = W @ u
        next_m = betas[0]*m + (1-betas[0])*grads
        next_v = betas[1]*v + (1-betas[1])*grads**2
        next_vhat = np.maximum(vhat, next_v)
        next_x = W @ x - .01 * next_m / np.sqrt(np.maximum(mixed_u, .02))
        next_u = mixed_u + next_vhat - vhat
        np.testing.assert_allclose(next_u.mean(0), next_vhat.mean(0), rtol=2e-5, atol=3e-7)
        np.testing.assert_allclose(next_x.mean(0), x.mean(0) - .01*(next_m/np.sqrt(np.maximum(mixed_u,.02))).mean(0),
                                   rtol=2e-5, atol=3e-7)
        for i, c in enumerate(sim.clients):
            state = c.rounds[r]
            actual_model, actual_state = result[state['model'].task]
            assert_model(actual_model, model_from(initial, torch.from_numpy(next_x[i])))
            for key, expected in [('m',next_m),('v',next_v),('vhat',next_vhat),('u',next_u)]:
                np.testing.assert_allclose(actual_state[key].numpy(), expected[i], rtol=3e-5, atol=4e-7)
            packet = result[state['packet'].task][0]
            for key, expected in [('x',x),('u',u)]:
                np.testing.assert_allclose(packet[key].numpy(), expected[i], rtol=3e-5, atol=4e-7)
            assert sum(t.numel()*t.element_size() for t in packet.values()) == sim.packet_size
            assert state['completed_time'] >= state['gradient_ready']
            for j in sim.neighbors(i):
                assert state['inbox'].messages[j] == sim.clients[j].rounds[r]['packet']
                assert state['completed_time'] >= sim.clients[j].rounds[r]['started_time'] + 3000
            if r > 1:
                prev = c.rounds[r-1]
                assert state['base'] == prev['model'] and state['previous'] == prev['state']
                assert state['started_time'] >= prev['completed_time']
            task = sim.workflow_dag.tasks[state['packet'].task]
            assert task.data['model'] == state['base'] and task.data['state'] == state['previous']
        x,m,v,vhat,u = next_x,next_m,next_v,next_vhat,next_u
    expected_bytes = sum(4*(2 if topology=='ring' else 3) *
        (sim.packet_size+s.control_message_bytes+len(pickle.dumps({'round':r}))) for r in range(1,7))
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == expected_bytes
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == expected_bytes


@pytest.mark.asyncio
async def test_single_client_delayed_denominator_floor_and_no_bias_correction(tmp_path, monkeypatch, logistic):
    from dasklearn import functions
    import dasklearn.simulation.damsgrad.simulation as implementation
    initial, _ = logistic
    monkeypatch.setattr(implementation, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    values = torch.tensor([1., 3., .5])
    def oracle(s, params):
        model = copy.deepcopy(params['model'])
        g = values[params['round']-1]
        model.gradient = [torch.full_like(p, g) for p in model.parameters()]
        return [model]
    monkeypatch.setattr(functions, 'compute_gradient', oracle)
    sim = run(DAMSGradSimulation, config(tmp_path, participants=1, rounds=3, beta1=.5, beta2=.5,
        epsilon=.04, learning=LearningSettings(.01,0,0,64,1)))
    result = execute_local(sim)
    x,m,v,vhat = vector(initial), 0., 0., .04
    for r,g in enumerate(values.tolist(),1):
        m = .5*m + .5*g
        v = .5*v + .5*g*g
        x = x - .01*m/max(vhat,.04)**.5  # Old maximum; no (1-beta^t) division.
        vhat = max(vhat,v)
        state = sim.clients[0].rounds[r]
        assert_model(result[state['model'].task][0], model_from(initial,x))
        torch.testing.assert_close(result[state['model'].task][1]['u'], torch.full_like(x,vhat))
    assert sim.clients[0].bw_scheduler.total_bytes_sent == 0


def test_floor_is_inside_sqrt_and_does_not_clamp_persistent_tracker(tmp_path):
    s = config(tmp_path, epsilon=.04)
    model = torch.nn.Linear(3,2)
    x = vector(model)
    old = initial_state(x,.04)
    partial = dict(m=torch.ones_like(x), v=torch.ones_like(x), vhat=torch.full_like(x,.06))
    # An artificial below-floor tracker distinguishes the raw tracker from the
    # clamped denominator. Real nonnegative mixing preserves positivity.
    packet = {'x': x.clone(), 'u': torch.full_like(x,.01)}
    actual,state = finish(s,dict(local=model,local_state=partial,state=old,packets=[packet],weights=[1.]))
    torch.testing.assert_close(vector(actual), x-.01/.2)
    torch.testing.assert_close(state['u'], torch.full_like(x,.03))
    assert torch.all(old['vhat']==.04) and torch.all(packet['u']==.01)


def test_tracker_reaches_consensus_with_unequal_local_second_moments(tmp_path, monkeypatch):
    from dasklearn import functions
    def oracle(s, params):
        model = copy.deepcopy(params['model'])
        model.gradient = [torch.full_like(p, params['peer']+1.) for p in model.parameters()]
        return [model]
    monkeypatch.setattr(functions, 'compute_gradient', oracle)
    s = config(tmp_path, beta2=0, epsilon=1e-6, learning=LearningSettings(.01,0,0,64,1))
    models = [torch.nn.Linear(3,2).double() for _ in range(4)]
    states = [None]*4
    for r in range(40):
        packets = [pack(s, dict(model=m,state=state))[0] for m,state in zip(models,states)]
        next_models,next_states = [],[]
        for i in range(4):
            local,partial = gradient(s,dict(model=models[i],state=states[i],peer=i))
            peers = [i,(i-1)%4,(i+1)%4]
            model,state = finish(s,dict(local=local,local_state=partial,state=states[i],
                packets=[packets[j] for j in peers],weights=[1/3]*3))
            next_models.append(model)
            next_states.append(state)
        models,states = next_models,next_states
        torch.testing.assert_close(torch.stack([v['u'] for v in states]).mean(0),
            torch.stack([v['vhat'] for v in states]).mean(0),rtol=1e-12,atol=1e-12)
    # Local maxima remain different; consensus is on their network mean.
    for i,state in enumerate(states):
        torch.testing.assert_close(state['vhat'],torch.full_like(state['vhat'],(i+1.)**2))
        torch.testing.assert_close(state['u'],torch.full_like(state['u'],7.5),rtol=1e-10,atol=1e-10)


@pytest.mark.asyncio
@pytest.mark.parametrize('latency', [0,20])
async def test_old_state_exchange_overlaps_gradient_and_waits_for_both(tmp_path, latency):
    sim = run(DAMSGradSimulation,config(tmp_path,participants=2,rounds=1,network_latency_ms=latency))
    for c in sim.clients:
        state = c.rounds[1]
        size = sim.packet_size+sim.settings.control_message_bytes+len(pickle.dumps({'round':1}))
        delivery = int(size/100000*MICROSECONDS)+latency*1000
        assert state['started_time']==0
        assert state['completed_time']==max(c.get_train_time(),delivery)


@pytest.mark.asyncio
async def test_early_duplicate_invalid_messages_and_completions(tmp_path):
    sim = run(DAMSGradSimulation,config(tmp_path))
    c = sim.clients[0]
    before = len(sim.workflow_dag.tasks)
    for peer in sim.neighbors(0):
        c.on_incoming_model(Event(sim.current_time,0,INCOMING_MODEL,
            {'from':peer,'model':c.rounds[1]['inbox'].messages[peer],'metadata':{'round':1}}))
        c.on_incoming_model(Event(sim.current_time,0,INCOMING_MODEL,
            {'from':peer,'model':TaskRef('early'),'metadata':{'round':9}}))
    c.finish_gradient(Event(sim.current_time,0,'finish_gradient',{'params':{'round':1}}))
    c.start_round(Event(sim.current_time,0,'start_round',{'round':1}))
    assert not c.rounds[9]['done'] and len(sim.workflow_dag.tasks)==before
    with pytest.raises(ValueError,match='Unexpected sender'):
        c.on_incoming_model(Event(0,0,INCOMING_MODEL,{'from':2,'model':TaskRef('bad'),'metadata':{'round':9}}))
    for bad in [0,-1,True,1.2]:
        with pytest.raises(ValueError,match='Invalid Decentralized AMSGrad round'):
            c.on_incoming_model(Event(0,0,INCOMING_MODEL,{'metadata':{'round':bad}}))


@pytest.mark.asyncio
@pytest.mark.parametrize('n,topology',[(1,'ring'),(2,'ring'),(5,'complete'),(6,'kreg')])
async def test_small_graphs_streaming_and_snapshots(tmp_path,n,topology):
    sim = run(DAMSGradSimulation,config(tmp_path,participants=n,topology=topology,degree=2,test_period=MICROSECONDS))
    execute_local(sim,retain_results=False)
    sim.merge_accuracies_files()
    frame = pd.read_csv(sim.data_dir+'/accuracies.csv')
    assert len(frame)==4*n and frame.groupby('time').peer.nunique().eq(n).all()
    assert frame.loc[frame.time==4,'round'].eq(6).all()
    assert np.isfinite(frame[['accuracy','loss']]).all().all()
    W = np.array([[sim.weight(i,j) if i==j or sim.graph.has_edge(i,j) else 0. for j in range(n)] for i in range(n)])
    np.testing.assert_allclose(W,W.T)
    np.testing.assert_allclose(W.sum(0),1.)
    np.testing.assert_allclose(W.sum(1),1.)
    if n>1:
        assert np.max(np.abs(np.linalg.eigvalsh(W)[:-1]))<1


@pytest.mark.asyncio
async def test_duration_stop_only_publishes_completed_updates(tmp_path):
    sim = run(DAMSGradSimulation,config(tmp_path,rounds=1,stop='duration',duration=MICROSECONDS))
    for c in sim.clients:
        assert sum(s['done'] for s in c.rounds.values())>1
        r,model = sim.latest_models[c.index]
        assert c.rounds[r]['done'] and c.rounds[r]['model']==model
        assert c.rounds[r]['completed_time']<=MICROSECONDS
    execute_local(sim,retain_results=False)


@pytest.mark.asyncio
async def test_seeded_serialized_replay(tmp_path):
    from dasklearn.tasks.dag import WorkflowDAG
    frames,dags = [],[]
    for repeat in range(2):
        Task.COUNTER.clear()
        sim = run(DAMSGradSimulation,config(tmp_path/str(repeat),test_period=MICROSECONDS),True)
        dags.append(sim.workflow_dag.serialize())
        sim.workflow_dag = WorkflowDAG.load_from_file(sim.data_dir+'/workflow_dag')
        sim.workflow_dag.build_task_indices()
        sim.workflow_dag.check_validity()
        execute_local(sim,retain_results=False)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir+'/accuracies.csv'))
    assert dags[0]==dags[1]
    pd.testing.assert_frame_equal(frames[0],frames[1],check_exact=True)


def test_inputs_are_immutable_and_local_buffers_follow_gradient(tmp_path,monkeypatch):
    from dasklearn import functions
    model = torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.BatchNorm1d(2))
    before = copy.deepcopy(model.state_dict())
    def oracle(s,params):
        result = copy.deepcopy(params['model'])
        result(torch.ones(8,3))
        result.gradient = [torch.ones_like(p) for p in result.parameters()]
        return [result]
    monkeypatch.setattr(functions,'compute_gradient',oracle)
    s = config(tmp_path)
    old = initial_state(vector(model),s.epsilon)
    snapshot = copy.deepcopy(old)
    packet = pack(s,dict(model=model,state=old))[0]
    packet_before = copy.deepcopy(packet)
    local,partial = gradient(s,dict(model=model,state=old))
    buffers = copy.deepcopy(dict(local.named_buffers()))
    actual,state = finish(s,dict(local=local,local_state=partial,state=old,packets=[packet],weights=[1.]))
    for key,value in before.items():
        torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    for key,value in buffers.items():
        torch.testing.assert_close(dict(actual.named_buffers())[key],value,rtol=0,atol=0)
    for key in old:
        torch.testing.assert_close(old[key],snapshot[key],rtol=0,atol=0)
    for key in packet:
        torch.testing.assert_close(packet[key],packet_before[key],rtol=0,atol=0)
    assert all(not t.requires_grad for t in state.values())


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
async def test_cpu_cuda_models_moments_tracker_and_packets(tmp_path):
    sim = run(DAMSGradSimulation,config(tmp_path),True)
    cpu = execute_local(sim)
    sim.settings.torch_device_name = 'cuda:0'
    gpu = execute_local(sim)
    for name,values in cpu.items():
        for a,b in zip(values,gpu[name]):
            if isinstance(a,torch.nn.Module):
                assert_model(a,b)
            elif isinstance(a,dict):
                for key in a:
                    torch.testing.assert_close(a[key],b[key],rtol=4e-5,atol=5e-7)


@pytest.mark.asyncio
@pytest.mark.parametrize('extra',[dict(topology='unknown'),dict(topology='kreg',degree=4),
    dict(topology='kreg',degree=0),dict(churn='synthetic'),dict(lora=True),dict(test_method='global'),
    dict(beta1=-.1),dict(beta1=1),dict(beta1=float('nan')),dict(beta2=-.1),dict(beta2=1),dict(beta2=float('inf')),
    dict(epsilon=0),dict(epsilon=-1),dict(epsilon=float('nan')),dict(epsilon=float('inf')),
    dict(epsilon=1e-99),dict(epsilon=1e99),
    dict(learning=LearningSettings(.01,.9,0,32,1)),dict(learning=LearningSettings(.01,0,0,32,5)),
    dict(learning=LearningSettings(0,0,0,32,1)),dict(learning=LearningSettings(float('nan'),0,0,32,1)),
    dict(learning=LearningSettings(.01,0,-.1,32,1))])
async def test_invalid_settings_fail_early(tmp_path,extra):
    with pytest.raises(ValueError):
        DAMSGradSimulation(config(tmp_path,**extra))


@pytest.mark.asyncio
@pytest.mark.parametrize('kind',['frozen','empty','mixed','complex'])
async def test_unsupported_parameter_layout(tmp_path,monkeypatch,kind):
    import dasklearn.simulation.damsgrad.simulation as implementation
    if kind=='empty':
        model = torch.nn.Identity()
    else:
        model = torch.nn.Linear(3,2)
        if kind=='frozen':model.weight.requires_grad_(False)
        if kind=='mixed':model.weight.data = model.weight.data.double()
        if kind=='complex':model.weight.data = model.weight.data.to(torch.complex64)
    monkeypatch.setattr(implementation,'create_model',lambda *a,**k:model)
    with pytest.raises(ValueError,match='unfrozen floating parameters'):
        DAMSGradSimulation(config(tmp_path))


def test_cli_alias_and_broker_settings(tmp_path):
    from args import get_args
    from dasklearn.algorithms import settings_classes
    from dasklearn.settings_factory import session_options
    from dasklearn.simulation.peer_baselines import baseline_spec
    args = get_args(['--algorithm','damsgrad','--local-steps','1','--momentum','0',
        '--damsgrad-topology','kreg','--damsgrad-degree','2','--damsgrad-beta1','.6',
        '--damsgrad-beta2','.95','--damsgrad-epsilon','.001'])
    s = DAMSGradSettings(**session_options(args,'cpu'))
    assert (s.topology,s.degree,s.beta1,s.beta2,s.epsilon)==('kreg',2,.6,.95,.001)
    broker = config(tmp_path)
    assert settings_classes()['DAMSGradSettings'].from_json(broker.to_json())==broker
    assert baseline_spec('damsgrad',4,4)==(DAMSGradSimulation,DAMSGradSettings,{})
