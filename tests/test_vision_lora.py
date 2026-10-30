"""Offline vision LoRA: real ViT/processor, shared frozen base, and protocol DAGs."""
import copy
import pickle

from datasets import Dataset, DatasetDict
import numpy as np
from PIL import Image
import pytest
import torch

pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from dasklearn import functions
from dasklearn.datasets import create_dataset
from dasklearn.lora import shared_runtime
from dasklearn.model_trainer import ModelTrainer
from dasklearn.model_evaluator import ModelEvaluator
from dasklearn.models import create_model, serialize_model, unserialize_model
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.dpsgd.settings import DPSGDSettings
from dasklearn.simulation.dpsgd.simulation import DPSGDSimulation
from dasklearn.tasks.local import execute_local
from helpers import settings, simulate


@pytest.fixture
def vision_config(tmp_path):
    torch.set_num_threads(1)
    checkpoint = tmp_path / "vit"
    transformers.ViTForImageClassification(transformers.ViTConfig(image_size=32, patch_size=8,
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=48,
        num_labels=3, hidden_dropout_prob=0, attention_probs_dropout_prob=0)).save_pretrained(checkpoint)
    transformers.ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(checkpoint)
    def split(count):
        return Dataset.from_dict({"image": [Image.fromarray(np.full((40, 40, 3), i % 3 * 100, dtype=np.uint8))
                                            for i in range(count)], "label": np.arange(count) % 3})
    data = tmp_path / "data"
    DatasetDict(train=split(96), validation=split(12), test=split(12)).save_to_disk(str(data))
    return settings(DPSGDSettings, tmp_path / "run", algorithm="dpsgd", dataset="vision", dataset_name=str(data), model="vit",
        participants=4, lora=True, lora_task="image-classification", base_model=str(checkpoint), num_labels=3,
        lora_rank=2, learning=LearningSettings(.1, 0, 0, 8, 1), topology="kreg", k=3,
        rounds=2, test_interval=1, torch_threads=1)


def test_vision_training_updates_only_adapter_and_head(vision_config):
    config = vision_config
    model = create_model("vision", settings=config)
    runtime = shared_runtime(model.spec)
    frozen = {name: parameter.clone() for name, parameter in runtime.model.named_parameters()}
    peer = copy.deepcopy(model)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    data = create_dataset(config)
    trainer = ModelTrainer(data, config, 0)
    batch = next(iter(trainer.make_train_loader(1)))
    assert batch["pixel_values"].shape == (8, 3, 32, 32)
    assert any("classifier" in name for name in model.parameter_names)
    evaluator = ModelEvaluator(data, config)
    loss_before = evaluator.evaluate_accuracy(model)[1]
    trainer.train(model, 30, seed=1)
    assert evaluator.evaluate_accuracy(model)[1] < loss_before
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(value, before[name]) for name, value in peer.state_dict().items())
    assert all(torch.equal(parameter, frozen[name]) and parameter.grad is None
               for name, parameter in runtime.model.named_parameters())
    restored = unserialize_model(serialize_model(model), "vision", settings=config)
    assert all(torch.equal(a, b) for a, b in zip(restored.parameters(), model.parameters()))
    assert len(pickle.dumps(model)) < len(pickle.dumps(runtime.model.state_dict())) / 3


@pytest.mark.parametrize("partitioner", ["iid", "dirichlet"])
@pytest.mark.asyncio
async def test_vision_lora_executes_three_regular_dpsgd_dag(vision_config, partitioner):
    config = vision_config
    config.partitioner, config.alpha = partitioner, .1
    sim = simulate(DPSGDSimulation, config)
    assert all(degree == 3 for _, degree in sim.topologies[0].out_degree())
    assert execute_local(sim)
    assert functions.last_test_metrics is not None
