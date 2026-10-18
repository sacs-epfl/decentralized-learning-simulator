import copy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import linprog

from dasklearn.simulation.moshpit.simulation import MoshpitSimulation, balanced_weights
from dasklearn.simulation.moshpit.settings import MoshpitSettings
from dasklearn.simulation.teleportation.simulation import TeleportationSimulation
from dasklearn.simulation.teleportation.settings import TeleportationSettings
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.tasks.local import execute_local
from test_conflux_comparison import settings, simulate


@pytest.mark.parametrize("bandwidths", [[1,1,1], [1,1,8,8], [1,2,3,4,5], [1,100,1000], [7,7]])
def test_moshpit_allocation_solves_published_lp(bandwidths):
    m = len(bandwidths)
    weights = balanced_weights(bandwidths)
    assert sum(weights) == pytest.approx(1) and min(weights) >= 0
    objective = max((1+(m-2)*w)/b for w,b in zip(weights, bandwidths))
    constraints = np.zeros((m,m+1))
    for i,b in enumerate(bandwidths):
        constraints[i,i] = m-2
        constraints[i,-1] = -b
    optimum = linprog([0]*m+[1], A_ub=constraints, b_ub=[-1]*m,
        A_eq=[[1]*m+[0]], b_eq=[1], bounds=[(0,None)]*(m+1), method="highs")
    assert optimum.success
    assert objective == pytest.approx(optimum.fun, rel=1e-8)
    assert objective <= max((1+(m-2)/m)/b for b in bandwidths)+1e-9


def test_variable_chunks_preserve_all_state_and_remainders():
    model = torch.nn.Linear(7,5)
    for weights in [[0,0,1], [.1,.2,.7], [1,1,1]]:
        sizes = ChunkManager.partition_sizes(71,weights)
        assert sum(sizes) == 71 and min(sizes) >= 0
        chunks = ChunkManager.chunk_model(model,3,weights)
        restored = ChunkManager.reconstruct_model([[c] for c in chunks],copy.deepcopy(model))
        assert torch.equal(ChunkManager.get_flat_params(restored),ChunkManager.get_flat_params(model))


@pytest.mark.asyncio
@pytest.mark.parametrize("topology,size", [("ring",1),("ring",2),("ring",4),("hypercube",4)])
async def test_teleportation_same_round_training_mix_and_token_handoff(tmp_path,topology,size):
    sim = simulate(TeleportationSimulation,settings(TeleportationSettings,tmp_path,
        sample_size=size,topology=topology,model_size_bytes=12000,network_latency_ms=10))
    results = execute_local(sim)
    trained = {(t.data["round"],t.data["peer"]):t for t in sim.workflow_dag.tasks.values() if t.func=="train"}
    mixes = [t for t in sim.workflow_dag.tasks.values() if t.name.startswith("teleport_mix")]
    assert len(mixes)==3*size and len(trained)==3*size
    for task in mixes:
        r, peer = task.data["round"],task.data["peer"]
        sample = sim.sample(r); token = sample.index(peer)
        owners = [peer]+[sample[i] for i in sim.neighbors(token)]
        assert task.data["models"] == [(trained[r,p].name,0) for p in owners]
        actual = ChunkManager.get_flat_params(results[task.name][0])
        expected = torch.stack([ChunkManager.get_flat_params(results[trained[r,p].name][0]) for p in owners]).mean(0)
        torch.testing.assert_close(actual,expected)
        if r<3:
            recipient = sim.sample(r+1)[token]
            assert trained[r+1,recipient].data["model"] == (task.name,0)
    sim.merge_accuracies_files()
    frame = pd.read_csv(sim.data_dir+"/accuracies.csv")
    assert frame.groupby("time").peer.nunique().eq(6).all()
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients)==sum(c.bw_scheduler.total_bytes_received for c in sim.clients)


@pytest.mark.asyncio
async def test_teleportation_waits_for_own_training_under_extreme_compute_skew(tmp_path):
    config = settings(TeleportationSettings,tmp_path,sample_size=4,rounds=2,model_size_bytes=1000)
    sim = TeleportationSimulation(config);sim.setup_directories()
    def capacities():
        for i,c in enumerate(sim.clients):
            c.simulated_speed = 1 if i else 100
    sim.apply_traces=capacities;sim.simulate();sim.workflow_dag.build_task_indices();sim.workflow_dag.check_validity()
    for t in sim.workflow_dag.tasks.values():
        if t.name.startswith("teleport_mix"):
            assert len(t.data["models"])==3
            for ref in t.data["models"]:
                source=sim.workflow_dag.tasks[ref[0]]
                assert source.func=="train" and source.data["round"]==t.data["round"]


@pytest.mark.asyncio
@pytest.mark.parametrize("count,group,steps",[(9,3,2),(8,2,3),(6,3,2)])
async def test_moshpit_full_grid_exact_consensus(tmp_path,count,group,steps):
    sim=simulate(MoshpitSimulation,settings(MoshpitSettings,tmp_path,participants=count,
        group_size=group,averaging_steps=steps,rounds=1,model_size_bytes=12001))
    results=execute_local(sim)
    trained=[results[t.name][0] for t in sim.workflow_dag.tasks.values() if t.func=="train"]
    expected=torch.stack([ChunkManager.get_flat_params(m) for m in trained]).mean(0)
    assert len(sim.latest_models)==count
    for r,ref in sim.latest_models.values():
        assert r==1
        torch.testing.assert_close(ChunkManager.get_flat_params(results[ref[0]][0]),expected)
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients)==sum(c.bw_scheduler.total_bytes_received for c in sim.clients)


@pytest.mark.asyncio
async def test_moshpit_single_phase_groups_and_mean_preservation(tmp_path):
    sim=simulate(MoshpitSimulation,settings(MoshpitSettings,tmp_path,participants=9,
        group_size=3,rounds=3,model_size_bytes=12000))
    results=execute_local(sim)
    for r in range(1,4):
        trains={t.data["peer"]:results[t.name][0] for t in sim.workflow_dag.tasks.values() if t.func=="train" and t.data["round"]==r}
        mixed=[t for t in sim.workflow_dag.tasks.values() if t.func=="reconstruct_from_chunks" and t.data["round"]==r]
        assert len(trains)==len(mixed)==9
        for task in mixed:
            members,_=sim.layout(task.data["peer"],r,0)
            expected=torch.stack([ChunkManager.get_flat_params(trains[p]) for p in members]).mean(0)
            torch.testing.assert_close(ChunkManager.get_flat_params(results[task.name][0]),expected)
        assert all(set(sim.layout(p,r,0)[0]).intersection(sim.layout(p,r+1,0)[0])=={p} for p in range(9))


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,setting,extra",[(TeleportationSimulation,TeleportationSettings,{"sample_size":3}),
                                           (MoshpitSimulation,MoshpitSettings,{"group_size":3})])
async def test_baseline_dag_and_metrics_are_reproducible(tmp_path,cls,setting,extra):
    dags=[];frames=[]
    for i in range(2):
        from dasklearn.tasks.task import Task
        Task.COUNTER.clear()
        sim=simulate(cls,settings(setting,tmp_path/str(i),model_size_bytes=12000,**extra))
        dags.append(sim.workflow_dag.serialize())
        execute_local(sim,retain_results=False);sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir+"/accuracies.csv"))
    assert dags[0]==dags[1]
    pd.testing.assert_frame_equal(frames[0],frames[1],check_exact=True)


def test_unimplemented_optimizer_state_is_rejected(tmp_path):
    config=settings(TeleportationSettings,tmp_path,sample_size=3)
    config.learning.momentum=.9
    with pytest.raises(ValueError,match="momentum"):
        TeleportationSimulation(config)
