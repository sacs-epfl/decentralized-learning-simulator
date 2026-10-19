import numpy as np
import pytest
import torch

from dasklearn.simulation.dspodfl.settings import DSpodFLSettings
from dasklearn.simulation.dspodfl.simulation import DSpodFLSimulation
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.tasks.local import execute_local
from helpers import settings,simulate


@pytest.mark.asyncio
@pytest.mark.parametrize('steps',[1,5])
@pytest.mark.parametrize('sgd,communication',[(1,1),(0,1),(1,0),(.5,.5),(0,0)])
async def test_dspodfl_matches_old_model_plus_local_delta(tmp_path,steps,sgd,communication):
    config=settings(DSpodFLSettings,tmp_path,participants=6,degree=2,rounds=3,
        probability_mode='constant',sgd_probability=sgd,communication_probability=communication,
        model_size_bytes=12001)
    config.learning.local_steps=steps
    sim=simulate(DSpodFLSimulation,config)
    values=execute_local(sim)
    flat=lambda ref:ChunkManager.get_flat_params(values[ref[0]][ref[1]])
    for client in sim.clients:
        for r,s in client.iterations.items():
            if not s['updated']:continue
            expected=flat(s['local']).clone()
            for neighbor in s['neighbors']:
                expected+=sim.weight(client.index,neighbor)*(flat(s['incoming'][neighbor])-flat(s['base']))
                assert client.index in sim.neighbors(neighbor,r)
                source=sim.workflow_dag.tasks[s['incoming'][neighbor][0]]
                assert source.data['round']<=r-1
                assert s['incoming'][neighbor]==sim.clients[neighbor].iterations[r]['base']
            if s['neighbors']:
                task=next(t for t in sim.workflow_dag.tasks.values() if t.func=='aggregate' and t.data['peer']==client.index and t.data['round']==r)
                torch.testing.assert_close(flat((task.name,0)),expected,rtol=1e-5,atol=1e-7)
                assert sum(task.data['weights'])==pytest.approx(1)
            if s['sgd'] and s['neighbors']:
                assert task.data['weights'][1]<0 # local gradient must not be diluted by gossip
    assert sim.completed_iterations and max(r[0] for r in sim.completed_iterations)==3
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients)==sum(c.bw_scheduler.total_bytes_received for c in sim.clients)
    if communication==0:assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients)==0
    if sgd==0:assert not any(t.func=='train' for t in sim.workflow_dag.tasks.values())


@pytest.mark.asyncio
async def test_dspodfl_same_iteration_mixing_is_doubly_stochastic(tmp_path):
    sim=simulate(DSpodFLSimulation,settings(DSpodFLSettings,tmp_path,degree=4,rounds=3))
    for r in range(1,4):
        matrix=np.eye(6)
        for i in range(6):
            for j in sim.neighbors(i,r):
                w=sim.weight(i,j);matrix[i,j]=w;matrix[i,i]-=w
        np.testing.assert_allclose(matrix,matrix.T)
        np.testing.assert_allclose(matrix.sum(axis=0),1)
        np.testing.assert_allclose(matrix.sum(axis=1),1)
        assert matrix.min()>=0


@pytest.mark.asyncio
async def test_dspodfl_buffered_arrivals_and_reproducibility_with_skew(tmp_path):
    records=[]
    for iteration in range(2):
        sim=DSpodFLSimulation(settings(DSpodFLSettings,tmp_path/str(iteration),degree=2,
            sgd_probability=1,communication_probability=1,probability_mode='constant'))
        sim.setup_directories()
        def profile():
            for c in sim.clients:c.simulated_speed=100 if c.index==0 else 1
        sim.apply_traces=profile;sim.simulate();sim.workflow_dag.build_task_indices();sim.workflow_dag.check_validity()
        for c in sim.clients:
            for r,s in c.iterations.items():
                if s['updated']:
                    train=next(t for t in sim.workflow_dag.tasks.values() if t.func=='train' and t.data['round']==r and t.data['peer']==c.index)
                    aggregate=next(t for t in sim.workflow_dag.tasks.values() if t.func=='aggregate' and t.data['round']==r and t.data['peer']==c.index)
                    assert aggregate.data['time']>=train.data['time']+c.get_train_time()
        records.append(sim.completed_iterations)
    assert records[0]==records[1]


@pytest.mark.asyncio
async def test_dspodfl_rejects_unsupported_configuration(tmp_path):
    for kw in [dict(churn='synthetic'),dict(degree=1),dict(sgd_probability=float('nan'))]:
        with pytest.raises(ValueError):DSpodFLSimulation(settings(DSpodFLSettings,tmp_path,**kw))
