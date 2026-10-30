"""Offline integration checks against real Hugging Face and Flower datasets."""
from dataclasses import replace
import json

from datasets import Dataset, DatasetDict
import numpy as np
from PIL import Image
import pytest
import torch

from dasklearn.datasets import create_dataset
from dasklearn.datasets.huggingface import allocate_indices, dataset_options, load_source, load_splits, partition_dataset
from dasklearn.model_evaluator import ModelEvaluator
from dasklearn.model_trainer import ModelTrainer
from dasklearn.models import create_model
from dasklearn.session_settings import SessionSettings, LearningSettings


def config(**kwargs):
    values = dict(algorithm="test", seed=7, work_dir="", dataset="text", participants=4,
                  learning=LearningSettings(.01, 0, 0, 8, 1))
    return SessionSettings(**(values | kwargs))


@pytest.mark.parametrize("partitioner", ["iid", "dirichlet", "shards", "natural"])
def test_flower_partitions_are_complete_disjoint_and_reproducible(partitioner):
    data = Dataset.from_dict({"label": np.arange(203) % 4, "client": np.arange(203) % 4})
    settings = config(partitioner=partitioner, partition_column="client" if partitioner == "natural" else None)
    parts = allocate_indices(data, settings)
    assert parts == allocate_indices(data, settings)
    assert sorted(index for part in parts for index in part) == list(range(len(data)))
    assert min(map(len, parts)) > 0
    if partitioner == "natural":
        assert all(len({data[index]["client"] for index in part}) == 1 for part in parts)
    else:
        assert parts != allocate_indices(data, replace(settings, seed=8))


def test_hub_source_passes_configuration_to_hugging_face(monkeypatch):
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return "loaded"
    monkeypatch.setattr("datasets.load_dataset", load)
    assert load_source(config(dataset_name="org/new-corpus", dataset_config="english", dataset_revision="abc123",
                              dataset_cache_dir="/tmp/data-cache")) == "loaded"
    assert calls == [(("org/new-corpus",), dict(name="english", revision="abc123", cache_dir="/tmp/data-cache"))]


@pytest.mark.parametrize("preset", ["synthetic", "digits"])
def test_offline_workloads_do_not_silently_ignore_a_requested_source(preset):
    with pytest.raises(ValueError, match="do not accept custom dataset sources"):
        create_dataset(config(dataset=preset, dataset_name="org/custom-data"))


def test_local_files_split_mapping_and_holdout(tmp_path):
    files = {}
    for split, count, offset in [("training", 40, 0), ("heldout", 8, 100)]:
        path = tmp_path / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps({"text": str(i), "label": i % 2})
                                  for i in range(offset, offset + count)))
        files[split] = str(path)
    settings = config(dataset_name="json", dataset_data_files=files, dataset_cache_dir=str(tmp_path / "cache"),
                      train_split="training", test_split="heldout", validation_set_fraction=.2)
    splits = load_splits(settings)
    train, validation, test = [set(splits[name]["text"]) for name in ("train", "validation", "test")]
    assert [len(train), len(validation), len(test)] == [32, 8, 8]
    assert not train & validation and not train & test and not validation & test
    assert validation == set(load_splits(replace(settings, seed=19))["validation"]["text"])
    restored = SessionSettings.from_json(settings.to_json())
    assert restored.dataset_data_files == files and restored.test_split == "heldout"


def image_data():
    def split(count, offset):
        return Dataset.from_dict({"picture": [Image.fromarray(np.full((32, 32, 3), i % 255, dtype=np.uint8))
                                               for i in range(count)],
                                  "target": np.arange(count) % 10,
                                  "id": np.arange(offset, offset + count), "unused": [None] * count})
    return DatasetDict(training=split(80, 0), tuning=split(20, 100), heldout=split(20, 200))


@pytest.mark.parametrize("native", [False, True])
def test_saved_images_train_validate_and_evaluate_with_custom_columns(tmp_path, native):
    torch.set_num_threads(1)
    image_data().save_to_disk(str(tmp_path))
    settings = config(dataset="cifar10", model="gnlenet", feature_column="picture", label_column="target",
                      train_split="training", validation_split="tuning", test_split="heldout",
                      compute_validation_loss_global_model=True,
                      **({"dataset_base_path": str(tmp_path)} if native else {"dataset_name": str(tmp_path)}))
    data = create_dataset(settings)
    model = create_model(settings.dataset, settings.model)
    trainer = ModelTrainer(data, settings, 0)
    batch = next(iter(trainer.make_train_loader(3)))
    assert batch["img"].shape == (8, 3, 32, 32)
    assert batch["label"].dtype == torch.int64
    result = trainer.train(model, 1, seed=3)
    assert result["samples"] == 8 and np.isfinite(result["validation_loss_global"])
    evaluator = ModelEvaluator(data, settings)
    accuracy, loss = evaluator.evaluate_accuracy(model, evaluation_examples=10)
    assert 0 <= accuracy <= 100 and np.isfinite(loss)
    assert data.load_split("validation") is not data.load_split("test")


def test_split_validation_and_natural_client_count_fail_clearly(monkeypatch):
    splits = DatasetDict(train=Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]}),
                         test=Dataset.from_dict({"text": ["c"], "label": [0]}))
    monkeypatch.setattr("dasklearn.datasets.huggingface.load_source", lambda _: splits)
    with pytest.raises(ValueError, match="different source splits"):
        load_splits(config(validation_split="test"))
    with pytest.raises(ValueError, match="nonempty"):
        load_splits(config(test_split="missing"))
    with pytest.raises(ValueError, match="found 2 clients"):
        partition_dataset(Dataset.from_dict({"client": [0, 1, 0, 1]}),
                          config(partitioner="natural", partition_column="client"))
    with pytest.raises(ValueError, match="finite and positive"):
        partition_dataset(splits["train"], config(participants=1, partitioner="dirichlet", alpha=float("nan")))


def test_dataset_arguments_are_available_to_both_runners(monkeypatch):
    from args import get_args
    monkeypatch.setattr("sys.argv", ["main.py", "--dataset", "text", "--dataset-name", "ag_news",
                                    "--dataset-config", "default", "--partitioner", "shards",
                                    "--shards-per-partition", "3", "--label-column", "category"])
    args = get_args()
    settings = config(**dataset_options(args), label_column=args.label_column)
    assert settings.dataset_name == "ag_news" and settings.dataset_config == "default"
    assert settings.shards_per_partition == 3 and settings.label_column == "category"


def test_tiny_native_cache_uses_distinct_validation_and_valid_test(tmp_path):
    data = image_data()
    data = DatasetDict({name: split.rename_columns({"picture": "image", "target": "label"})
                        for name, split in (("train", data["training"]), ("valid", data["heldout"]))})
    data.save_to_disk(str(tmp_path))
    settings = config(dataset="tiny_imagenet", model="resnet18", dataset_base_path=str(tmp_path), participants=2)
    loaded = create_dataset(settings)
    assert len(loaded.load_split("validation")) == 4
    assert len(loaded.load_split("test")) == 20
    assert loaded.load_split("test") is loaded.load_split("valid")
    assert sum(len(part) for part in loaded.partitions) == 76
    assert loaded.load_partition(0)[0]["image"].shape == (3, 32, 32)


def test_speech_silence_is_removed_before_natural_partitioning(monkeypatch):
    split = Dataset.from_dict({"audio": ["placeholder"] * 6, "label": [0, 1, 2, 0, 1, 2],
                               "speaker_id": ["a", "b", None, "a", "b", "None"]})
    monkeypatch.setattr("dasklearn.datasets.huggingface.load_source",
                        lambda _: DatasetDict(train=split, validation=split, test=split))
    settings = config(dataset="google_speech", participants=2, partitioner="natural")
    splits = load_splits(settings)
    allocator = partition_dataset(splits["train"], settings)
    assert len(splits["test"]) == 4
    assert [len(allocator.load_partition(peer)) for peer in range(2)] == [2, 2]


def test_generic_preparation_saves_reloadable_data_and_provenance(tmp_path, monkeypatch):
    from scripts.prepare_dataset import main
    from datasets import load_from_disk
    data = DatasetDict(train=Dataset.from_dict({"text": ["one", "two"]}),
                       test=Dataset.from_dict({"text": ["three"]}))
    monkeypatch.setattr("scripts.prepare_dataset.load_dataset", lambda *args, **kwargs: data)
    monkeypatch.setattr("sys.argv", ["prepare_dataset.py", "org/corpus", "--revision", "abc", "--data", str(tmp_path)])
    main()
    assert load_from_disk(str(tmp_path))["test"]["text"] == ["three"]
    source = json.loads((tmp_path / "source.json").read_text())
    assert source["dataset"] == "org/corpus" and source["revision"] == "abc"
