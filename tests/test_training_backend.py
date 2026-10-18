import argparse
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn import functions
from dasklearn.datasets.synthetic import SyntheticDataset, TensorPartition
from dasklearn.datasets.tiny_imagenet import TinyPartition
from dasklearn.models import create_model
from dasklearn.session_settings import SessionSettings, LearningSettings
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.training import TrainingExecutor, add_training_arguments
from dasklearn.tasks.task import Task
from dasklearn.simulation.shatter.simulation import ShatterSimulation
from dasklearn.simulation.shatter.settings import ShatterSettings
from dasklearn.simulation.conflux_v2.simulation import ConfluxV2Simulation
from dasklearn.simulation.conflux_v2.settings import ConfluxV2Settings
from test_conflux_comparison import settings, simulate


def config(**kwargs):
    values=dict(algorithm="test",seed=7,work_dir="",dataset="synthetic",participants=8,
        learning=LearningSettings(.01,.9,.001,16,5),torch_threads=1,training_batch_size=4)
    values.update(kwargs)
    return SessionSettings(**values)


def reset(dataset, monkeypatch):
    monkeypatch.setattr(functions,"dataset",dataset)
    monkeypatch.setattr(functions,"model_trainers",None)


def assert_states(actual, expected, exact=False):
    for name,value in actual.state_dict().items():
        other=expected.state_dict()[name]
        if exact or not value.is_floating_point():assert torch.equal(value,other),name
        else:torch.testing.assert_close(value,other,rtol=1e-4,atol=1e-5,msg=name)
        assert value.device.type=="cpu"


def inputs(s, count=4, round_nr=1):
    model=functions.initialize(s,{})[0]
    result=[]
    for peer in range(count):
        private=copy.deepcopy(model)
        with torch.no_grad():next(private.parameters()).view(-1)[0].add_(peer*.01)
        result.append({"model":private,"peer":peer,"round":round_nr,"time":0})
    return result


def test_options_are_opt_out():
    p=argparse.ArgumentParser();add_training_arguments(p)
    defaults=p.parse_args([])
    assert defaults.gpu_residency and defaults.training_batching
    assert defaults.training_batch_size==2
    disabled=p.parse_args(["--no-gpu-residency","--no-training-batching"])
    assert not disabled.gpu_residency and not disabled.training_batching
    assert config().gpu_residency and config().training_batching


@pytest.mark.parametrize("resident",[False,True])
def test_independent_batched_sgd_and_fresh_momentum(monkeypatch,resident):
    torch.set_num_threads(1)
    s=config(gpu_residency=resident);dataset=SyntheticDataset(s);data=inputs(s)
    reset(dataset,monkeypatch)
    expected=[functions.train(s,d)[0] for d in data]
    reset(dataset,monkeypatch);backend=TrainingExecutor(s,autotune=False)
    before=[copy.deepcopy(d["model"].state_dict()) for d in data]
    actual=backend.train_many(data,[None]*4)
    for a,b in zip(actual,expected):assert_states(a,b)
    for d,state in zip(data,before):
        for k,v in d["model"].state_dict().items():assert torch.equal(v,state[k])
    # Reusing the bank must not carry optimizer momentum or BN state into a new task.
    next_data=[{**d,"round":2,"model":a} for d,a in zip(data[:2],actual)]
    expected2=[functions.train(s,d)[0] for d in next_data]
    actual2=backend.train_many(next_data,[None]*2)
    for a,b in zip(actual2,expected2):assert_states(a,b)
    assert backend.stats["batched_tasks"]==6
    assert all(p.grad is None for m in actual for p in m.parameters())
    backend.close()


def test_resident_workspace_preserves_scalar_results_and_inputs(monkeypatch):
    s=config(training_batching=False);dataset=SyntheticDataset(s);data=inputs(s,count=2)
    reset(dataset,monkeypatch);expected=[functions.train(s,d)[0] for d in data]
    reset(dataset,monkeypatch);backend=TrainingExecutor(s)
    a=backend.train_many(data[:1],["shared"])[0]
    pointer=next(backend.workspace.parameters()).data_ptr()
    b=backend.train_many(data[1:],[None])[0]
    assert pointer==next(backend.workspace.parameters()).data_ptr()
    assert_states(a,expected[0],exact=True);assert_states(b,expected[1],exact=True)
    assert backend.cache_bytes<=backend.cache_limit
    backend.drop_input("shared");assert backend.cache_bytes==0
    backend.close()


def test_unequal_minibatches_are_not_padded_or_dropped(monkeypatch):
    s=config();g=torch.Generator().manual_seed(1)
    parts=[TensorPartition(torch.randn(n,16,generator=g),torch.arange(n)%4) for n in [19,19,11,7]]
    dataset=SimpleNamespace(load_partition=lambda peer,split:parts[peer]);data=inputs(s)
    reset(dataset,monkeypatch);expected=[functions.train(s,d)[0] for d in data]
    reset(dataset,monkeypatch);backend=TrainingExecutor(s,autotune=False)
    actual=backend.train_many(data,[None]*4)
    for a,b in zip(actual,expected):assert_states(a,b)
    assert backend.stats["batched_tasks"]==2 and backend.stats["serial_tasks"]==2
    backend.close()


@pytest.mark.parametrize("error",[NotImplementedError("test vmap limitation"),torch.OutOfMemoryError("test memory limit")])
def test_batch_failure_restarts_from_immutable_inputs(monkeypatch,error):
    s=config();dataset=SyntheticDataset(s);data=inputs(s,count=2)
    reset(dataset,monkeypatch);expected=[functions.train(s,d)[0] for d in data]
    reset(dataset,monkeypatch);backend=TrainingExecutor(s,autotune=False)
    def fail(*args):raise error
    monkeypatch.setattr(backend,"vectorized",fail)
    actual=backend.train_many(data,[None]*2)
    for a,b in zip(actual,expected):assert_states(a,b,exact=True)
    assert not backend.batch_enabled and backend.stats["fallback_reasons"]
    backend.close()


def test_unexpected_errors_are_not_hidden(monkeypatch):
    s=config();reset(SyntheticDataset(s),monkeypatch);backend=TrainingExecutor(s,autotune=False)
    def fail(*args):raise RuntimeError("invalid training target")
    monkeypatch.setattr(backend,"vectorized",fail)
    with pytest.raises(RuntimeError,match="invalid training target"):
        backend.train_many(inputs(s,count=2),[None]*2)
    backend.close()


def test_startup_calibration_publishes_reference_values(monkeypatch):
    s=config();dataset=SyntheticDataset(s);data=inputs(s,count=2)
    reset(dataset,monkeypatch);expected=[functions.train(s,d)[0] for d in data]
    reset(dataset,monkeypatch);backend=TrainingExecutor(s)
    actual=backend.train_many(data,[None]*2)
    for a,b in zip(actual,expected):assert_states(a,b,exact=True)
    assert backend.tuned and "calibration" in backend.stats
    assert backend.stats["calibration_tasks"]==2 and backend.stats["batched_tasks"]==0
    backend.close()


def test_input_cache_is_bounded_and_independent():
    backend=TrainingExecutor(config(gpu_state_cache_mb=1))
    for i in range(8):
        model=torch.nn.Linear(128,512)
        state=backend.input_state(model,i)
        assert state["weight"].data_ptr()!=model.weight.data_ptr()
        assert backend.cache_bytes<=1024*1024
    assert len(backend.cache)<8
    huge=torch.nn.Linear(512,1024)
    backend.input_state(huge,"oversized")
    assert "oversized" not in backend.cache
    backend.close();assert backend.cache_bytes==0


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,setting,extra",[
    (ShatterSimulation,ShatterSettings,{"k":3,"r":2}),
    (ConfluxV2Simulation,ConfluxV2Settings,{"sample_size":3,"chunks_in_sample":3})])
async def test_prefetch_preserves_dag_and_every_value(tmp_path,cls,setting,extra):
    results,frames=[],[]
    for accelerate in (False,True):
        Task.COUNTER.clear()
        sim=simulate(cls,settings(setting,tmp_path/str(accelerate),model_size_bytes=12000,**extra))
        original=copy.deepcopy(sim.workflow_dag.serialize())
        backend=TrainingExecutor(sim.settings,autotune=False) if accelerate else None
        results.append(execute_local(sim,training_backend=backend))
        assert sim.workflow_dag.serialize()==original
        if accelerate:assert sim.execution_training_stats["batched_tasks"]>0
        if accelerate:assert sim.execution_training_stats["prefetch_peak_models"]<=sim.settings.training_batch_size
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir+"/accuracies.csv").sort_values(["time","peer"]).reset_index(drop=True))
    assert results[0].keys()==results[1].keys()
    for name,values in results[0].items():
        for a,b in zip(results[1][name],values):
            if isinstance(a,torch.nn.Module):assert_states(a,b)
            else:torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
    pd.testing.assert_frame_equal(*frames,rtol=1e-5,atol=1e-5)


@pytest.mark.asyncio
async def test_executor_releases_training_resources_on_error(tmp_path):
    sim=simulate(ShatterSimulation,settings(ShatterSettings,tmp_path,k=3,r=2))
    class FailingBackend:
        width=2
        closed=False
        def can_train(self,data):return True
        def can_batch(self,data):return False
        def train_many(self,*args):raise RuntimeError("intentional training failure")
        def close(self):self.closed=True
    backend=FailingBackend()
    with pytest.raises(RuntimeError,match="intentional training failure"):
        execute_local(sim,training_backend=backend)
    assert backend.closed


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA host validation")
@pytest.mark.parametrize("batching",[False,True])
@pytest.mark.parametrize("resident",[False,True])
def test_cuda_resnet_independent_bn_and_sgd(monkeypatch,request,batching,resident):
    torch.set_num_threads(2)
    old=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(True)
    request.addfinalizer(lambda:torch.use_deterministic_algorithms(old))
    s=config(dataset="tiny_imagenet",dataset_base_path="native-tensor-test",model="resnet18",
        torch_device_name="cuda:0",training_batching=batching,gpu_residency=resident,
        learning=LearningSettings(.01,.9,.001,8,5))
    g=torch.Generator().manual_seed(6)
    parts=[TinyPartition(torch.randn(24,3,64,64,generator=g),torch.arange(24)%200,np.arange(24)) for _ in range(2)]
    dataset=SimpleNamespace(load_partition=lambda peer,split:parts[peer]);data=inputs(s,count=2)
    with torch.no_grad():
        for i,item in enumerate(data):
            for m in item["model"].modules():
                if isinstance(m,torch.nn.BatchNorm2d):
                    m.running_mean.fill_(i*.1);m.running_var.fill_(1+i*.1);m.num_batches_tracked.fill_(i*10)
    before=[copy.deepcopy(d["model"].state_dict()) for d in data]
    reset(dataset,monkeypatch);expected=[functions.train(s,d)[0] for d in data]
    reset(dataset,monkeypatch);backend=TrainingExecutor(s,autotune=False)
    actual=backend.train_many(data,[None]*2)
    for a,b in zip(actual,expected):assert_states(a,b,exact=True)
    for d,state in zip(data,before):
        for k,v in d["model"].state_dict().items():assert torch.equal(v,state[k])
    next_data=[{**d,"round":2,"model":m} for d,m in zip(data,actual)]
    expected_next=[functions.train(s,d)[0] for d in next_data]
    actual_next=backend.train_many(next_data,[None]*2)
    for a,b in zip(actual_next,expected_next):assert_states(a,b,exact=True)
    backend.close()
