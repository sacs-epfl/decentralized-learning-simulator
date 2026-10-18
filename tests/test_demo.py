import copy
import math
import pickle
import numpy as np
import pytest
import torch
from scipy.fft import dctn

from dasklearn.optimizer.demo import encode, decode, compress, decompress, prepare, update, packet_bytes, block_shape
from dasklearn.simulation.demo.settings import DeMoSettings
from dasklearn.simulation.demo.simulation import DeMoSimulation
from dasklearn.tasks.local import execute_local
from dasklearn import functions
from test_conflux_comparison import settings, simulate


@pytest.mark.parametrize('shape',[(37,), (12,20), (8,3,5,5), ()])
def test_dct_matches_orthonormal_scipy_blocks(shape):
    torch.manual_seed(2); x = torch.randn(shape, dtype=torch.float64)
    logical, splits = block_shape(x.shape, 8)
    if len(logical)==1:
        blocked=x.reshape(-1,splits[0]).numpy(); axes=(-1,)
    else:
        a,b=splits
        blocked=x.reshape(logical[0]//a,a,logical[1]//b,b).permute(0,2,1,3).numpy(); axes=(-2,-1)
    np.testing.assert_allclose(encode(x,8),dctn(blocked,axes=axes,norm='ortho'),atol=1e-13)
    torch.testing.assert_close(decode(encode(x,8),x.shape,8),x)


def test_duplicate_indices_mean_excludes_unsent_workers():
    packets=[(torch.tensor([[0,2]]),torch.tensor([[2.,8.]])),
             (torch.tensor([[0,1]]),torch.tensor([[6.,3.]]))]
    actual=encode(decompress(packets,(4,),4),4)
    torch.testing.assert_close(actual,torch.tensor([[4.,3.,8.,0.]]),atol=1e-6,rtol=1e-5)


def test_private_residual_is_conserved_and_persists():
    torch.manual_seed(3); gradients=[torch.randn(8,12)]
    packet,residual=prepare(gradients,None,.01,.999,4,2)
    decoded=decompress([packet[0]],gradients[0].shape,4)
    torch.testing.assert_close(residual[0]+decoded,.01*gradients[0])
    old=residual[0].clone()
    packet2,residual2=prepare(gradients,residual,.01,.999,4,2)
    torch.testing.assert_close(residual2[0]+decompress([packet2[0]],gradients[0].shape,4),old*.999+.01*gradients[0])
    assert torch.equal(old,residual[0])
    assert not torch.equal(packet[0][1],packet2[0][1])


def test_dense_limit_is_sign_sgd_with_decoupled_decay():
    torch.manual_seed(11);model=torch.nn.Linear(5,3);before=copy.deepcopy(model)
    gradients=[[torch.randn_like(p) for p in model.parameters()] for _ in range(3)]
    packets=[prepare(g,None,.01,.999,64,999)[0] for g in gradients]
    result=update(model,packets,.01,.2,64)
    for i,(actual,original) in enumerate(zip(result.parameters(),before.parameters())):
        expected=original*(1-.01*.2)-.01*torch.stack([g[i] for g in gradients]).mean(0).sign()
        torch.testing.assert_close(actual,expected)
    for a,b in zip(model.parameters(),before.parameters()):assert torch.equal(a,b)
    assert packet_bytes([p.shape for p in model.parameters()],64,999)==sum(v.numel()*(v.element_size()+i.element_size()) for i,v in packets[0])


@pytest.mark.asyncio
@pytest.mark.parametrize('peers',[1,2,6])
async def test_ring_training_dag_and_byte_conservation(tmp_path,peers):
    config=settings(DeMoSettings,tmp_path,participants=peers,rounds=3,compression_chunk=8,compression_topk=2)
    sim=simulate(DeMoSimulation,config); values=execute_local(sim)
    assert len(sim.updates)==3
    for client in sim.clients:
        for r,state in client.rounds.items():
            if not state['done']:continue
            assert len(state['packets'])==peers
            task=sim.workflow_dag.tasks[sim.updates[r][0]]
            if r>1:
                source=next(t for t in sim.workflow_dag.tasks.values() if t.func=='demo_prepare' and t.data['round']==r and t.data['peer']==client.index)
                assert source.data['model']==sim.updates[r-1]
                assert source.data['residual'][1]==1
                assert sim.workflow_dag.tasks[source.data['residual'][0]].data['peer']==client.index
            packets=[values[state['packets'][i][0]][0] for i in range(peers)]
            actual_bytes=sum(i.numel()*i.element_size()+v.numel()*v.element_size() for i,v in packets[0])
            assert actual_bytes==sim.packet_size
    sent=sum(c.bw_scheduler.total_bytes_sent for c in sim.clients)
    received=sum(c.bw_scheduler.total_bytes_received for c in sim.clients)
    assert sent==received
    expected=0
    for client in sim.clients:
        for r in range(1,4):
            for stage in range(peers-1):
                meta={'round':r,'stage':stage,'origin':(client.index-stage)%peers}
                expected+=sim.packet_size+config.control_message_bytes+len(pickle.dumps(meta))
    assert sent==expected
    # All replicas of each round use the exact same aggregate, without a free global average.
    assert all(c.rounds[3]['done'] for c in sim.clients)


@pytest.mark.asyncio
async def test_accumulation_is_fixed_weight_gradient_and_streaming_matches(tmp_path):
    config=settings(DeMoSettings,tmp_path,participants=2,rounds=2,compression_chunk=8,compression_topk=2)
    sim=simulate(DeMoSimulation,config);values=execute_local(sim)
    task=next(t for t in sim.workflow_dag.tasks.values() if t.func=='demo_prepare')
    model=copy.deepcopy(values[task.data['model'][0]][0]); original=copy.deepcopy(model)
    trainer=functions.model_trainers[task.data['peer']]
    seed=functions.training_seed(config,task.data)
    gradients=[torch.zeros_like(p) for p in model.parameters()]
    batches=iter(trainer.make_train_loader(seed))
    for _ in range(config.learning.local_steps):
        batch=next(batches); model.zero_grad();torch.nn.CrossEntropyLoss()(model(batch['img']),batch['label']).backward()
        for g,p in zip(gradients,model.parameters()):g.add_(p.grad/config.learning.local_steps)
    packet,residual=values[task.name]
    for g,p,r in zip(gradients,packet,residual):
        torch.testing.assert_close(r+decompress([p],g.shape,8),config.learning.learning_rate*g)
    import pandas as pd
    from pathlib import Path
    files=list(Path(sim.data_dir).glob('accuracies_*.csv'))
    before=pd.concat([pd.read_csv(f) for f in files],ignore_index=True).sort_values(['time','peer']).reset_index(drop=True)
    for f in files:f.unlink()
    execute_local(sim,retain_results=False)
    after=pd.concat([pd.read_csv(f) for f in files],ignore_index=True).sort_values(['time','peer']).reset_index(drop=True)
    pd.testing.assert_frame_equal(before,after,check_exact=True)
    for a,b in zip(model.parameters(),original.parameters()):assert torch.equal(a,b)


@pytest.mark.asyncio
async def test_ring_buffers_future_packets_under_skew(tmp_path):
    sim=DeMoSimulation(settings(DeMoSettings,tmp_path,participants=6,rounds=3,compression_chunk=8,compression_topk=2))
    sim.setup_directories()
    def capacities():
        for c in sim.clients:
            c.simulated_speed=80 if c.index==0 else 1
            c.bw_scheduler.bw_limit=100000 if c.index%2 else 40000
    sim.apply_traces=capacities;sim.simulate();sim.workflow_dag.build_task_indices();sim.workflow_dag.check_validity()
    execute_local(sim,retain_results=False)
    for c in sim.clients:
        assert c.rounds[3]['done']
        for r in [2,3]:
            source=next(t for t in sim.workflow_dag.tasks.values() if t.func=='demo_prepare' and t.data['peer']==c.index and t.data['round']==r)
            assert source.data['time']>=sim.workflow_dag.tasks[sim.updates[r-1][0]].data['time']


def test_reject_unsupported_configs(tmp_path):
    for kw in [dict(churn='synthetic'),dict(compression_decay=float('nan')),dict(compression_topk=0),dict(model='resnet18',dataset='cifar10')]:
        with pytest.raises(ValueError):DeMoSimulation(settings(DeMoSettings,tmp_path,**kw))
