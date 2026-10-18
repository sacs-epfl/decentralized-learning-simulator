from types import SimpleNamespace
import numpy as np
import torch
from dasklearn import functions
from dasklearn.datasets.tiny_imagenet import TinyPartition
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.task import Task


def test_tensor_chunk_consumers_copy_only_the_slice(monkeypatch):
    source = torch.arange(100000, dtype=torch.float32)
    monkeypatch.setattr(functions, "memory_source", lambda settings, data: list(source.split(1000)), raising=False)
    def inspect(settings, data):
        piece = data["piece"]
        assert piece.untyped_storage().nbytes() == piece.numel() * piece.element_size()
        return [piece.sum()]
    monkeypatch.setattr(functions, "memory_probe", inspect, raising=False)
    a = Task("source", "memory_source", {})
    b = Task("probe", "memory_probe", {"piece": ("source", 3)})
    a.outputs.append((b, 3)); b.inputs.append((a, 3))
    sim = SimpleNamespace(settings=SimpleNamespace(torch_threads=1), workflow_dag=SimpleNamespace(tasks={a.name:a,b.name:b}))
    result = execute_local(sim)
    assert result["probe"][0] == source[3000:4000].sum()


def test_resnet_chunks_roundtrip_parameters_and_batchnorm_buffers():
    from torchvision.models import resnet18
    torch.set_num_threads(1)
    model = resnet18(num_classes=200)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.fill_(.25)
                module.running_var.fill_(1.75)
                module.num_batches_tracked.fill_(35)
    chunks = ChunkManager.chunk_model(model, 16)
    restored = ChunkManager.reconstruct_model([[x] for x in chunks], resnet18(num_classes=200))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
        assert restored.state_dict()[key].dtype == value.dtype


def test_tiny_evaluation_subset_is_balanced_and_fixed():
    labels = torch.arange(200).repeat_interleave(10)
    data = TinyPartition(torch.zeros(2000, 1), labels, np.arange(2000))
    a, b = data.evaluation_subset(400), data.evaluation_subset(400)
    assert a is b and len(a) == 400
    assert np.all(np.bincount(labels[a.indices].numpy()) == 2)
