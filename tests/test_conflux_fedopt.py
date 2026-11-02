"""Numerical reference, state ownership, quantized handoff and wire-byte checks."""
import copy
from types import SimpleNamespace
import pytest
import torch
from dasklearn import functions
from dasklearn.models import create_model
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.simulation.conflux_v2.optimizer import (quantize,dequantize,pack_state,
    state_payload_bytes,packet_payload_bytes,update_state,parameter_mask)
from dasklearn.simulation.conflux_v2.settings import ConfluxV2Settings
from dasklearn.simulation.conflux_v2.simulation import ConfluxV2Simulation
from dasklearn.tasks.local import execute_local
from helpers import settings,simulate


def protocol(tmp_path,optimizer,epoch=0,bits=32,fraction=1,compute_ms=0):
    config=settings(ConfluxV2Settings,tmp_path,participants=8,sample_size=4,chunks_in_sample=8,
        relay_count=4,rounds=4,duration=100_000_000,test_period=0,immediate_uploads=True,
        success_fraction=fraction,server_optimizer=optimizer,server_learning_rate=.03,
        reducer_epoch_rounds=epoch,state_bits=bits,state_block_size=17,
        model_size_bytes=12001,network_latency_ms=10,server_step_ms=compute_ms)
    sim=simulate(ConfluxV2Simulation,config)
    return sim,execute_local(sim)


def models(sim,results):
    return {(t.data['round'],t.data['peer']):ChunkManager.get_flat_params(results[t.name][0])
        for t in sim.workflow_dag.tasks.values() if t.func=='reconstruct_from_chunks'}


@pytest.mark.asyncio
@pytest.mark.parametrize('optimizer',['fedavgm','fedadam'])
@pytest.mark.parametrize('epoch',[0,1,2])
@pytest.mark.parametrize('fraction',[1,.5])
async def test_sharded_optimizer_matches_independent_full_vector_reference(tmp_path,optimizer,epoch,fraction):
    sim,results=protocol(tmp_path,optimizer,epoch=epoch,fraction=fraction)
    s=sim.settings
    base=ChunkManager.get_flat_params(functions.initialize(s,{})[0])
    m=torch.zeros_like(base);v=torch.full_like(base,s.server_tau**2)
    actual=models(sim,results)
    assert len(actual)==16
    for r in range(1,5):
        local=[results[t.name][0] for t in sim.workflow_dag.tasks.values()
               if t.func=='train' and t.data['round']==r and t.data['peer'] in sim.contributor_sets[r]]
        # Full-vector math is independent of chunk reconstruction/optimizer helpers.
        avg=torch.stack([ChunkManager.get_flat_params(value) for value in local]).mean(0)
        g=base-avg
        if optimizer=='fedavgm':
            m=s.server_beta1*m+g;base=base-s.server_learning_rate*m
        else:
            m=s.server_beta1*m+(1-s.server_beta1)*g
            v=s.server_beta2*v+(1-s.server_beta2)*g.square()
            base=base-s.server_learning_rate*m/(v.sqrt()+s.server_tau)
        for (round_nr,_),model in actual.items():
            if round_nr==r:torch.testing.assert_close(model,base,rtol=1e-6,atol=1e-7)
    if epoch==0:assert not sim.optimizer_handoffs
    else:assert sim.optimizer_handoffs
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients)==sum(c.bw_scheduler.total_bytes_received for c in sim.clients)


@pytest.mark.asyncio
async def test_migration_quantization_preserves_common_models_and_reduces_actual_payload(tmp_path):
    exact,expected=protocol(tmp_path/'exact','fedadam',epoch=1,bits=32)
    compressed,actual=protocol(tmp_path/'compressed','fedadam',epoch=1,bits=8)
    assert exact.optimizer_handoffs and len(exact.optimizer_handoffs)==len(compressed.optimizer_handoffs)
    assert sum(r[-1] for r in compressed.optimizer_handoffs)<sum(r[-1] for r in exact.optimizer_handoffs)
    exact_sent=sum(c.bw_scheduler.total_bytes_sent for c in exact.clients)
    compressed_sent=sum(c.bw_scheduler.total_bytes_sent for c in compressed.clients)
    assert exact_sent-compressed_sent==sum(r[-1] for r in exact.optimizer_handoffs)-sum(
        r[-1] for r in compressed.optimizer_handoffs)
    outputs=models(compressed,actual)
    for r in range(1,5):
        values=[v for (round_nr,_),v in outputs.items() if round_nr==r]
        assert all(torch.equal(values[0],v) for v in values)
    for task in compressed.workflow_dag.tasks.values():
        if task.func=='conflux_opt_pack':
            state=actual[task.name][0]
            assert state['m']['values'].dtype==torch.int8
            assert state['v']['values'].dtype==torch.uint8
            assert packet_payload_bytes(state)==state_payload_bytes(state['x'].numel(),'fedadam',8,17)
    assert torch.isfinite(torch.stack(list(outputs.values()))).all()


@pytest.mark.asyncio
async def test_charged_optimizer_compute_delays_completion_without_changing_models(tmp_path):
    fast,expected=protocol(tmp_path/'fast','fedadam',compute_ms=0)
    slow,actual=protocol(tmp_path/'slow','fedadam',compute_ms=20)
    assert max(t for _,_,t in slow.completed_rounds)>max(t for _,_,t in fast.completed_rounds)
    for key,value in models(fast,expected).items():assert torch.equal(value,models(slow,actual)[key])


@pytest.mark.parametrize('unsigned',[False,True])
def test_block_quantization_zero_outliers_and_wire_size(unsigned):
    values=torch.tensor([0.,1e-8,1.,100.,.1,0.,3.],dtype=torch.float32)
    if not unsigned:values[3]*=-1
    packet=quantize(values,3,unsigned)
    restored=dequantize(packet)
    assert restored.shape==values.shape and torch.isfinite(restored).all()
    assert torch.all((restored-values).abs()<=packet['scales'].repeat_interleave(3)[:len(values)]*.501)
    zeros=quantize(torch.zeros(7),3,unsigned)
    assert torch.equal(dequantize(zeros),torch.zeros(7))
    if unsigned:assert torch.all(restored>=0)


def test_buffers_are_averaged_not_optimized():
    # ResNet running statistics and integer counters must not receive Adam steps.
    s=SimpleNamespace(server_optimizer='fedadam',server_beta1=.9,server_beta2=.99,
        server_learning_rate=.01,server_tau=.001,dataset='cifar10',model='resnet18')
    mask=parameter_mask(s.dataset,s.model)
    base=torch.ones(mask.numel());avg=base+1
    state=dict(x=base,m=torch.zeros_like(base),v=torch.ones_like(base)*1e-6,step=0)
    updated,new=update_state(s,dict(state=state,chunks=[avg],round=1,n=1,chunk_index=0))
    assert (~mask).any()
    assert torch.equal(updated[~mask],avg[~mask])
    assert torch.equal(new['m'][~mask],torch.zeros_like(new['m'][~mask]))


@pytest.mark.parametrize('changes',[
    dict(server_beta1=1),dict(server_tau=0),dict(server_learning_rate=float('nan')),
    dict(state_bits=4),dict(reducer_epoch_rounds=-1),dict(state_block_size=0),
    dict(server_step_ms=-1),dict(server_optimizer='fedavg',server_learning_rate=.1)])
def test_invalid_optimizer_settings_rejected(tmp_path,changes):
    with pytest.raises(ValueError):
        ConfluxV2Simulation(settings(ConfluxV2Settings,tmp_path,sample_size=4,**changes))
