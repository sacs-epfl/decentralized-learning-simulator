from dasklearn.simulation.moshpit.simulation import MoshpitSimulation
from dasklearn.simulation.moshpit.settings import MoshpitSettings
import copy

import pandas as pd
import pytest
import torch

from dasklearn import functions
from dasklearn.models import (create_model, create_model_structure, copy_model_state,
                              serialize_model, unserialize_model)
from dasklearn.simulation.shatter.settings import ShatterSettings
from dasklearn.simulation.shatter.simulation import ShatterSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.task import Task
from dasklearn.tasks.references import TaskRef
from dasklearn.session_settings import SessionSettings, LearningSettings
from helpers import settings, simulate


def wire_copy(model, dataset, architecture=None):
    return unserialize_model(serialize_model(model), dataset, architecture)


@pytest.mark.parametrize("dataset,architecture", [
    ("synthetic", None), ("digits", None), ("cifar10", "gnlenet"), ("tiny_imagenet", "resnet18")])
def test_direct_state_copy_matches_wire_and_is_independent(dataset, architecture):
    torch.set_num_threads(1)
    model = create_model(dataset, architecture).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.fill_(.25)
                module.running_var.fill_(1.75)
                module.num_batches_tracked.fill_(35)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    expected = wire_copy(model, dataset, architecture)
    rng = torch.random.get_rng_state().clone()
    actual = copy_model_state(model, dataset, architecture)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert [m.training for m in actual.modules()] == [m.training for m in expected.modules()]
    for key, value in actual.state_dict().items():
        assert torch.equal(value, expected.state_dict()[key]), key
        assert value.dtype == expected.state_dict()[key].dtype
        assert value.data_ptr() != model.state_dict()[key].data_ptr()
    assert all(p.grad is None for p in actual.parameters())
    before = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        next(actual.parameters()).add_(1)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    # Mutating a returned structure must not poison the cached prototype.
    fresh = copy_model_state(model, dataset, architecture)
    assert all(torch.equal(v, before[k]) for k, v in fresh.state_dict().items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA equivalence checked on GPU host")
def test_cuda_to_cpu_state_copy_matches_wire():
    model = create_model("tiny_imagenet", "resnet18").cuda().eval()
    expected = wire_copy(model, "tiny_imagenet", "resnet18")
    actual = copy_model_state(model, "tiny_imagenet", "resnet18")
    assert all(v.device.type == "cpu" and torch.equal(v, expected.state_dict()[k])
               for k, v in actual.state_dict().items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA equivalence checked on GPU host")
def test_cuda_resnet_training_matches_reference_bit_for_bit(tmp_path, monkeypatch, request):
    from dasklearn.datasets.tiny_imagenet import TinyPartition
    import numpy as np
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    request.addfinalizer(lambda: torch.use_deterministic_algorithms(previous))
    config = SessionSettings(algorithm="test", seed=4, work_dir=str(tmp_path),
        data_dir=str(tmp_path), dataset="tiny_imagenet", model="resnet18", participants=1,
        learning=LearningSettings(.01, .9, 0, 32, 3), torch_device_name="cuda:0")
    generator = torch.Generator().manual_seed(42)
    partition = TinyPartition(torch.randn(32,3,64,64,generator=generator), torch.arange(32), np.arange(32))
    from types import SimpleNamespace
    dataset = SimpleNamespace(load_partition=lambda *args: partition)
    model = functions.initialize(config, {})[0]
    before = copy.deepcopy(model.state_dict())
    values = []
    for optimized in (False, True):
        config.release_cuda_cache = not optimized
        with monkeypatch.context() as patch:
            patch.setattr(functions, "dataset", dataset)
            patch.setattr(functions, "model_trainers", None)
            if not optimized:
                patch.setattr(functions, "copy_model_state", wire_copy)
            values.append(functions.train(config, {"model":model,"peer":0,"round":1,"time":0})[0])
    for key, value in values[1].state_dict().items():
        assert torch.equal(value, values[0].state_dict()[key]), key
        assert torch.equal(model.state_dict()[key], before[key])


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,config,kwargs", [
    (ShatterSimulation, ShatterSettings, {"k":3, "r":2}),
    (MoshpitSimulation, MoshpitSettings, {"group_size":3})])
async def test_optimized_execution_matches_reference_bit_for_bit(tmp_path, monkeypatch, cls, config, kwargs):
    runs, frames = [], []
    for optimized in (False, True):
        Task.COUNTER.clear()
        sim = simulate(cls, settings(config, tmp_path / str(optimized), model_size_bytes=12000, **kwargs))
        assert sim.model_size == 12000
        sim.settings.release_cuda_cache = not optimized
        with monkeypatch.context() as patch:
            if not optimized:
                patch.setattr(functions, "copy_model_state", wire_copy)
                patch.setattr(functions, "create_model_structure", create_model)
            result = execute_local(sim, share_readonly_inputs=optimized)
        runs.append(result)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + "/accuracies.csv").sort_values(["time", "peer"]).reset_index(drop=True))
    assert runs[0].keys() == runs[1].keys()
    for name in runs[0]:
        for expected, actual in zip(runs[0][name], runs[1][name]):
            if isinstance(actual, torch.nn.Module):
                for key, value in actual.state_dict().items():
                    assert torch.equal(value, expected.state_dict()[key]), (name, key)
            else:
                assert torch.equal(actual, expected), name
    pd.testing.assert_frame_equal(*frames, check_exact=True)


@pytest.mark.asyncio
async def test_invalid_payload_size_rejected(tmp_path):
    with pytest.raises(ValueError, match="model_size_bytes must be positive"):
        simulate(MoshpitSimulation, settings(MoshpitSettings, tmp_path,
            group_size=3, model_size_bytes=0))


@pytest.mark.asyncio
async def test_gradient_only_copy_preserves_gradients_and_predecessor(tmp_path, monkeypatch):
    config = settings(MoshpitSettings, tmp_path, group_size=3)
    model = functions.initialize(config, {})[0]
    before = copy.deepcopy(model.state_dict())
    values = []
    for optimized in (False, True):
        functions.dataset = None
        functions.model_trainers = None
        with monkeypatch.context() as patch:
            if not optimized:
                patch.setattr(functions, "copy_model_state", wire_copy)
            values.append(functions.compute_gradient(config, {"model":model, "round":1, "time":0, "peer":0})[0])
    for expected, actual in zip(values[0].gradient, values[1].gradient):
        assert torch.equal(actual, expected)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,config,kwargs", [
    (ShatterSimulation, ShatterSettings, {"k":3, "r":2}),
    (MoshpitSimulation, MoshpitSettings, {"group_size":3})])
async def test_reconstruction_memo_preserves_all_snapshots(tmp_path, monkeypatch, cls, config, kwargs):
    frames = []
    reconstruct = functions.reconstruct_from_chunks
    for budget in (0, 1, 64*1024*1024):
        sim = simulate(cls, settings(config, tmp_path/str(budget), model_size_bytes=12000, **kwargs))
        tasks = [task for task in sim.workflow_dag.tasks.values() if task.func == "reconstruct_from_chunks"]
        unique = len({repr(task.data["chunks"]) for task in tasks})
        calls = []
        def counted(*args):
            calls.append(1)
            return reconstruct(*args)
        status = []
        with monkeypatch.context() as patch:
            patch.setattr(functions, "reconstruct_from_chunks", counted)
            execute_local(sim, retain_results=False, reconstruction_cache_bytes=budget, progress=status.append)
        assert len(calls) == (unique if budget > 1 else len(tasks))
        assert status[-1]["reconstruction_cache_peak_bytes"] <= budget
        assert status[-1]["reconstruction_cache_bytes"] == 0
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir+"/accuracies.csv").sort_values(["peer","time"]).reset_index(drop=True))
    for actual in frames[1:]:
        pd.testing.assert_frame_equal(frames[0], actual, check_exact=True)


@pytest.mark.parametrize("models_in_budget,expected_calls", [(1,3), (2,2)])
def test_reconstruction_cache_eviction_and_release(monkeypatch, models_in_budget, expected_calls):
    from types import SimpleNamespace
    config = SessionSettings(algorithm="test", seed=4, work_dir="", dataset="synthetic", participants=1,
        learning=LearningSettings(.1,0,0,16,1), torch_threads=1)
    tasks = [Task("init", "initialize", {})]
    for label in ("a", "b"):
        tasks.append(Task(label, "chunk", {"model":TaskRef("init",0), "n":2}))
    for i,label in enumerate(("a","b","a","b")):
        tasks.append(Task("recon"+str(i), "reconstruct_from_chunks",
            {"chunks":[[TaskRef(label,0)],[TaskRef(label,1)]], "time":i+1}))
    by_name = {task.name:task for task in tasks}
    def connect(task, value):
        if isinstance(value, tuple) and value[0] in by_name:
            source = by_name[value[0]]
            task.inputs.append((source,value[1]));source.outputs.append((task,value[1]))
        elif isinstance(value, dict):
            for item in value.values(): connect(task,item)
        elif isinstance(value, list):
            for item in value: connect(task,item)
    for task in tasks: connect(task,task.data)
    sim = SimpleNamespace(settings=config, workflow_dag=SimpleNamespace(tasks=by_name))
    size = sum(v.numel()*v.element_size() for v in create_model("synthetic").state_dict().values())
    calls, status = [], []
    reconstruct = functions.reconstruct_from_chunks
    def counted(*args):
        calls.append(1)
        return reconstruct(*args)
    monkeypatch.setattr(functions,"reconstruct_from_chunks",counted)
    execute_local(sim,retain_results=False, reconstruction_cache_bytes=models_in_budget*size, progress=status.append)
    # The last A use needs no cache insertion, leaving B available for its hit.
    assert len(calls) == expected_calls
    assert status[-1]["reconstruction_cache_peak_bytes"] <= models_in_budget*size
    assert status[-1]["reconstruction_cache_bytes"] == 0
