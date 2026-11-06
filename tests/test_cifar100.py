"""Native fine-label CIFAR-100 and the buffer-free small-image ResNet."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from dasklearn.datasets import create_dataset
from dasklearn.datasets.huggingface import load_splits
from dasklearn.models import create_model
from dasklearn.model_trainer import ModelTrainer
from dasklearn.session_settings import LearningSettings, SessionSettings


@pytest.fixture
def cifar_source(monkeypatch):
    features = Features(img=Image(), fine_label=ClassLabel(num_classes=100),
                        coarse_label=Value('int64'), example_id=Value('int64'))
    def split(count, offset):
        return Dataset.from_dict(dict(
            img=[PILImage.fromarray(np.full((32, 32, 3), index % 256, dtype=np.uint8)) for index in range(count)],
            fine_label=[index % 100 for index in range(count)], coarse_label=[0] * count,
            example_id=list(range(offset, offset + count))), features=features)
    source = DatasetDict(train=split(1000, 0), test=split(200, 10000))
    monkeypatch.setattr('dasklearn.datasets.huggingface.load_source', lambda settings: source)
    return source


def config(tmp_path):
    return SessionSettings(algorithm='dpsgd', dataset='cifar100', model='resnet18_gn',
        work_dir=str(tmp_path), seed=100, participants=4, partitioner='iid',
        validation_set_fraction=.1, learning=LearningSettings(.01, 0, 0, 2, 1))


def test_cifar100_uses_fine_labels_and_disjoint_fixed_holdout(cifar_source, tmp_path):
    settings = config(tmp_path)
    splits = load_splits(settings)
    assert [len(splits[key]) for key in ['train', 'validation', 'test']] == [900, 100, 200]
    identities = [set(splits[key]['example_id']) for key in ['train', 'validation', 'test']]
    assert not any(identities[i] & identities[j] for i, j in [(0, 1), (0, 2), (1, 2)])
    assert len(set(splits['train']['label'])) == 100
    assert splits['validation']['example_id'] == load_splits(replace(settings, seed=101))['validation']['example_id']
    data = create_dataset(settings)
    assert sum(map(len, data.partitions)) == 900
    assert sorted(index for part in data.partitions for index in part.indices) == list(range(900))
    assert data.test[0]['img'].shape == (3, 32, 32)
    assert data.test[0]['label'] == 0
    torch.testing.assert_close(data.test[0]['img'][:, 0, 0],
        -torch.tensor([.5071, .4867, .4408]) / torch.tensor([.2675, .2565, .2761]))
    subset = data.test.evaluation_subset(100)
    assert np.array_equal(np.bincount(subset.labels[subset.indices], minlength=100), np.ones(100))
    assert set(subset.indices) <= set(data.test.indices)
    with pytest.raises(ValueError):
        data.test.evaluation_subset(101)


def test_resnet18_gn_trains_at_native_resolution_without_buffers(cifar_source, tmp_path):
    torch.set_num_threads(2)
    settings = config(tmp_path)
    model = create_model(settings.dataset, settings.model)
    assert sum(parameter.numel() for parameter in model.parameters()) == 11220132
    assert model.conv1.kernel_size == (3, 3) and model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)
    assert not list(model.buffers())
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    before = model.conv1.weight.detach().clone()
    data = create_dataset(settings)
    ModelTrainer(data, settings, 0).train(model, 1, seed=100)
    assert not torch.equal(before, model.conv1.weight)
    prediction = model(torch.randn(2, 3, 32, 32))
    assert prediction.shape == (2, 100) and torch.isfinite(prediction).all()
