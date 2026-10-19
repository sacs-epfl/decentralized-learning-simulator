"""Independent full-batch trajectories for Algorithm 1 and momentum controls."""
import copy
import pickle
import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn.optimizer.qgm import flat,prepare,finish
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.qgm.settings import QGMSettings
from dasklearn.simulation.qgm.simulation import QGMSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.task import Task
from dasklearn.tasks.references import TaskRef
from dasklearn.util import MICROSECONDS
from helpers import settings
from test_classic_baselines import logistic,run,assert_model


def config(tmp_path,**kwargs):
    values=dict(algorithm='qgm',participants=4,rounds=5,duration=3*MICROSECONDS,
                test_period=0,learning=LearningSettings(.1,0,.01,64,1),
                model_size_bytes=1024,network_latency_ms=3,beta=.8,mu=.6)
    values.update(kwargs)
    return settings(QGMSettings,tmp_path,**values)


def model_from(template,vector):
    result=copy.deepcopy(template);offset=0
    with torch.no_grad():
        for p in result.parameters():
            p.copy_(vector[offset:offset+p.numel()].view_as(p));offset+=p.numel()
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['quasi','local','plain'])
@pytest.mark.parametrize('nesterov',[False,True])
@pytest.mark.parametrize('synchronous',[False,True])
async def test_independent_multiround_logistic_reference(tmp_path,monkeypatch,logistic,mode,nesterov,synchronous):
    import dasklearn.simulation.qgm.simulation as implementation
    initial,parts=logistic
    monkeypatch.setattr(implementation,'create_model',lambda *a,**k:copy.deepcopy(initial))
    sim=run(QGMSimulation,config(tmp_path,mode=mode,nesterov=nesterov,synchronous=synchronous),True)
    results=execute_local(sim)
    x=torch.stack([flat(initial)]*4);memory=torch.zeros_like(x)
    W=torch.tensor([[sim.weight(i,j) if i==j or sim.graph.has_edge(i,j) else 0
                     for j in range(4)] for i in range(4)])
    for r in range(1,6):
        gradients=[]
        for i in range(4):
            m=model_from(initial,x[i]);p=parts[i]
            torch.nn.functional.cross_entropy(m(p.features),p.labels).backward()
            gradients.append(torch.cat([v.grad.reshape(-1) for v in m.parameters()]))
        gradients=torch.stack(gradients)+.01*x
        beta=0 if mode=='plain' else .8
        local=beta*memory+gradients
        direction=gradients+beta*local if nesterov and mode!='plain' else local
        proposal=x-.1*direction
        next_x=W@proposal
        if mode=='quasi':next_memory=.6*memory+.4*(x-next_x)/.1
        elif mode=='local':next_memory=local
        else:next_memory=torch.zeros_like(memory)
        for i in range(4):
            state=sim.clients[i].rounds[r]
            actual_model,actual_memory=results[state['model'][0]]
            assert_model(actual_model,model_from(initial,next_x[i]))
            if mode=='plain':assert actual_memory is None
            else:torch.testing.assert_close(actual_memory,next_memory[i],rtol=2e-5,atol=2e-6)
            if r>1:
                assert sim.workflow_dag.tasks[state['model'][0]].data['momentum']==sim.clients[i].rounds[r-1]['state']
        torch.testing.assert_close(next_x.mean(0),proposal.mean(0))
        x,memory=next_x,next_memory
    per_round=sim.packet_size+sim.settings.control_message_bytes+len(pickle.dumps({'round':1,'phase':'momentum-model'}))
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients)==5*8*per_round
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients)==5*8*per_round


@pytest.mark.asyncio
@pytest.mark.parametrize('n,topology',[(1,'ring'),(2,'ring'),(4,'complete'),(6,'kreg')])
async def test_topologies_streaming_and_snapshots(tmp_path,n,topology):
    sim=run(QGMSimulation,config(tmp_path,participants=n,degree=2,topology=topology,test_period=MICROSECONDS))
    execute_local(sim,retain_results=False);sim.merge_accuracies_files()
    frame=pd.read_csv(sim.data_dir+'/accuracies.csv')
    assert len(frame)==3*n and frame.groupby('time').peer.nunique().eq(n).all()
    assert frame['round'].eq(5).all() and np.isfinite(frame[['accuracy','loss']]).all().all()
    assert all(i not in sim.neighbors(i) for i in sim.graph)


@pytest.mark.asyncio
async def test_reproducibility_and_worker_state_is_explicit(tmp_path):
    dags,frames=[],[]
    for repeat in range(2):
        Task.COUNTER.clear();sim=run(QGMSimulation,config(tmp_path/str(repeat),test_period=MICROSECONDS),True)
        dags.append(sim.workflow_dag.serialize());execute_local(sim,retain_results=False);sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir+'/accuracies.csv'))
    assert dags[0]==dags[1]
    pd.testing.assert_frame_equal(frames[0],frames[1],check_exact=True)


def test_no_input_mutation_and_buffers_stay_local():
    m=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.BatchNorm1d(2))
    m.train();m(torch.randn(4,3));m.gradient=[torch.ones_like(p) for p in m.parameters()]
    before=copy.deepcopy(m.state_dict());previous=torch.ones_like(flat(m))
    old=previous.clone()
    local,state,packet=prepare(m,previous,.1,.9,.01,True,'quasi')
    mixed,memory=finish(m,local,previous,state,[packet,packet+1],[.5,.5],.1,.9,'quasi')
    assert not hasattr(local,'gradient') and torch.equal(previous,old)
    for name,value in before.items():assert torch.equal(m.state_dict()[name],value)
    for name,value in local.named_buffers():assert torch.equal(dict(mixed.named_buffers())[name],value)
    assert not packet.requires_grad and not memory.requires_grad


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['quasi','local','plain'])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
async def test_cuda_and_cpu_full_states_match(tmp_path,mode):
    s=config(tmp_path,mode=mode);sim=run(QGMSimulation,s,True)
    cpu=execute_local(sim);s.torch_device_name='cuda:0';gpu=execute_local(sim)
    for name,values in cpu.items():
        for a,b in zip(values,gpu[name]):
            if isinstance(a,torch.nn.Module):assert_model(a,b)
            elif a is None:assert b is None
            else:torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)


@pytest.mark.asyncio
@pytest.mark.parametrize('extra',[dict(beta=1),dict(beta=float('nan')),dict(mu=-1),dict(mode='bogus'),
    dict(churn='synthetic'),dict(test_method='global'),dict(topology='directed'),dict(topology='kreg',degree=4),
    dict(learning=LearningSettings(.1,0,0,32,5)),dict(learning=LearningSettings(0,0,0,32,1)),
    dict(learning=LearningSettings(.1,.9,0,32,1))])
async def test_unsupported_settings_rejected(tmp_path,extra):
    with pytest.raises(ValueError):QGMSimulation(config(tmp_path,**extra))


@pytest.mark.asyncio
async def test_validation_curves_and_final_test_never_share_cached_metrics(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from dasklearn import functions
    from dasklearn.datasets.synthetic import TensorPartition
    from dasklearn.simulation.simulation import Simulation
    from dasklearn.simulation.client import BaseClient
    model=torch.nn.Linear(16,4)
    with torch.no_grad():model.weight.zero_();model.bias.copy_(torch.tensor([2.,0.,0.,0.]))
    val=TensorPartition(torch.zeros(20,16),torch.zeros(20,dtype=torch.long))
    test=TensorPartition(torch.zeros(20,16),torch.ones(20,dtype=torch.long))
    data=SimpleNamespace(load_split=lambda split:val if split=='validation' else test)
    monkeypatch.setattr(functions,'create_dataset',lambda s:data)
    monkeypatch.setattr(functions,'initialize',lambda *a:[copy.deepcopy(model)])
    sim=Simulation(config(tmp_path,participants=1,evaluation_split='validation'))
    sim.setup_directories();client=BaseClient(sim,0);sim.clients=[client]
    initial=client.create_compute_task('initialize',{'peer':0,'round':0,'time':0},name='initial')
    for i,split in enumerate(['validation','test','validation']):
        client.create_compute_task('test',{'model':TaskRef('initial',0),'peer':0,
            'round':1,'time':(i+1)*MICROSECONDS,'evaluation_split':split,'output_prefix':f'metric{i}'},name=f'test-{i}')
    execute_local(sim,retain_results=False)
    rows=[pd.read_csv(sim.data_dir+f'/metric{i}_0.csv').iloc[0] for i in range(3)]
    assert [r.accuracy for r in rows]==[100,0,100]
    assert rows[1].loss>rows[0].loss
    assert sim.settings.evaluation_split=='validation'
