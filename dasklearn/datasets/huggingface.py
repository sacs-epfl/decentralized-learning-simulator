"""Shared Hugging Face loading and Flower partitioning for every real dataset."""
import json
import math
from pathlib import Path
from typing import Protocol


class FederatedData(Protocol):
    def load_partition(self, peer: int, split: str = "train"): ...
    def load_split(self, split: str): ...


SOURCES = {
    "cifar10": "uoft-cs/cifar10",
    "femnist": "coscotuff/femnist",
    "google_speech": "google/speech_commands",
    "tiny_imagenet": "zh-plus/tiny-imagenet",
    "text": "toy-text",
}


def add_dataset_arguments(parser):
    parser.add_argument("--dataset-name", help="Hugging Face ID, local directory, or builder (json, csv, imagefolder, ...)")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--dataset-data-files", type=json.loads, default=None,
                        help='JSON split-to-files mapping, e.g. {"train":"train.csv","test":"test.csv"}')
    parser.add_argument("--dataset-base-path", "--dataset-path", default=None,
                        help="Local dataset directory (including DatasetDict.save_to_disk output)")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default=None, help="Defaults to test (valid for Tiny ImageNet)")
    parser.add_argument("--validation-split", default=None)
    parser.add_argument("--feature-column", default=None)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--partition-column", default=None, help="Label column, or client ID column for natural partitioning")
    parser.add_argument("--shards-per-partition", type=int, default=2)
    parser.add_argument("--min-partition-size", type=int, default=1, help="Minimum Dirichlet client size (native image/digits caches require at least 8)")


def dataset_options(args):
    return {name: getattr(args, name) for name in (
        "dataset_name", "dataset_config", "dataset_revision", "dataset_cache_dir", "dataset_data_files",
        "dataset_base_path", "train_split", "test_split", "validation_split", "feature_column",
        "partition_column", "shards_per_partition", "min_partition_size")}


def load_source(settings):
    """Use the same loader for Hub IDs, local files, saved Arrow data and toy text."""
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    source = settings.dataset_name or (settings.text_dataset if settings.dataset == "text"
                                      else SOURCES.get(settings.dataset, settings.dataset))
    config = settings.dataset_config or (settings.text_dataset_config if settings.dataset == "text" else None)
    if config is None and settings.dataset == "google_speech" and settings.dataset_name is None:
        config = "v0.02"
    local = Path(settings.dataset_base_path or source).expanduser()
    if local.is_dir() and (local / "dataset_dict.json").exists():
        return load_from_disk(str(local))
    if settings.dataset_base_path or local.is_dir():
        source = str(local)
    if source == "toy-text":
        from dasklearn.datasets.text import toy_records
        return DatasetDict({name: Dataset.from_list(toy_records(count, offset)) for name, count, offset in (
            ("train", max(256, settings.participants * 32), 0),
            ("validation", 64, 100000), ("test", 64, 200000))})
    kwargs = {"name": config, "cache_dir": settings.dataset_cache_dir, "revision": settings.dataset_revision}
    if settings.dataset_data_files is not None:
        kwargs["data_files"] = settings.dataset_data_files
    if settings.dataset_base_path and settings.dataset_name in {"json", "csv", "parquet", "imagefolder", "audiofolder", "text"}:
        source = settings.dataset_name
        kwargs["data_dir"] = str(local)
    return load_dataset(source, **{key: value for key, value in kwargs.items() if value is not None})


def load_splits(settings):
    """Normalize split/column names; reserve validation before partitioning clients."""
    from datasets import ClassLabel, DatasetDict

    loaded = load_source(settings)
    train = settings.train_split
    test = settings.test_split or ("valid" if settings.dataset == "tiny_imagenet" else "test")
    validation = settings.validation_split or ("validation" if "validation" in loaded else None)
    if validation == test and settings.validation_split is None:
        validation = None  # A source validation split may be the final test set.
    names = [name for name in (train, test, validation) if name is not None]
    if len(set(names)) != len(names):
        raise ValueError("Train, validation and test must refer to different source splits")
    for name in names:
        if name not in loaded or not len(loaded[name]):
            raise ValueError(f"Dataset needs a nonempty {name!r} split; available splits: {list(loaded)}")
    splits = DatasetDict(train=loaded[train], test=loaded[test])
    if validation:
        splits["validation"] = loaded[validation]
    if settings.dataset == "google_speech":
        # Filter before client allocation so silent examples cannot create empty
        # natural-ID clients or contribute to label-based partitions.
        splits = DatasetDict({name: split.filter(lambda speaker: speaker not in (None, "None", ""), input_columns=["speaker_id"])
                              if "speaker_id" in split.column_names else split
                              for name, split in splits.items()})
    if not validation:
        fraction = settings.validation_set_fraction or (0.05 if settings.dataset == "tiny_imagenet" else 0.1)
        if not 0 < fraction < 1:
            raise ValueError("validation_set_fraction must be in (0, 1)")
        # A fixed holdout seed keeps tuning data the same across partition seeds.
        label = settings.label_column if settings.dataset == "text" else (
            settings.label_column if settings.label_column != "label" else column_names(settings)[1])
        stratify = label if isinstance(splits["train"].features.get(label), ClassLabel) else None
        heldout = splits["train"].train_test_split(test_size=fraction, seed=1729, stratify_by_column=stratify)
        splits["train"], splits["validation"] = heldout["train"], heldout["test"]
    if settings.dataset != "text":
        feature, label = column_names(settings)
        if settings.dataset == "google_speech":
            feature = "audio"
        source_feature = settings.feature_column or feature
        source_label = settings.label_column if settings.label_column != "label" else label
        renames = {old: new for old, new in ((source_feature, feature), (source_label, label)) if old != new}
        for name, split in splits.items():
            missing = {source_feature, source_label} - set(split.column_names)
            if missing:
                raise ValueError(f"Split {name!r} is missing columns {sorted(missing)}; configure --feature-column/--label-column")
            splits[name] = split.rename_columns(renames) if renames else split
    if any(not len(split) for split in splits.values()):
        raise ValueError("Train, validation and test splits must be nonempty after preprocessing")
    return splits


def column_names(settings):
    return ("x", "y") if settings.dataset == "femnist" else (
        "image" if settings.dataset in {"tiny_imagenet", "vision"} else "img", "label")


def partition_dataset(data, settings, *, min_size=1, label_column="label"):
    """Flower partitioners operate directly on a Hugging Face Dataset."""
    from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, NaturalIdPartitioner, ShardPartitioner

    peers = settings.participants
    if settings.min_partition_size < 1:
        raise ValueError("min_partition_size must be positive")
    min_size = max(min_size, settings.min_partition_size)
    if peers < 1 or min_size < 1 or len(data) < peers * min_size:
        raise ValueError(f"Need at least {min_size} training examples per participant")
    column = settings.partition_column or label_column
    if settings.partition_column == settings.label_column:
        column = label_column
    if settings.partitioner == "iid":
        partitioner = IidPartitioner(num_partitions=peers)
        data = data.shuffle(seed=settings.seed)
    elif settings.partitioner == "dirichlet":
        if not math.isfinite(settings.alpha) or settings.alpha <= 0:
            raise ValueError("Dirichlet alpha must be finite and positive")
        partitioner = DirichletPartitioner(num_partitions=peers, partition_by=column,
            alpha=settings.alpha, min_partition_size=min_size, seed=settings.seed)
    elif settings.partitioner == "natural":
        column = settings.partition_column or {"femnist": "writer_id", "google_speech": "speaker_id"}.get(settings.dataset)
        if not column:
            raise ValueError("Natural partitioning requires --partition-column with a client ID column")
        partitioner = NaturalIdPartitioner(partition_by=column)
    elif settings.partitioner == "shards":
        if settings.shards_per_partition < 1 or len(data) < peers * settings.shards_per_partition:
            raise ValueError("Need at least one example per shard and positive --shards-per-partition")
        # Let Flower distribute every shard, including the remainder. Fixing
        # num_shards_per_partition instead can silently discard leftover rows.
        partitioner = ShardPartitioner(num_partitions=peers, partition_by=column,
            shard_size=len(data) // (peers * settings.shards_per_partition),
            keep_incomplete_shard=True, seed=settings.seed)
    else:
        raise ValueError(f"Unknown partitioner: {settings.partitioner}")
    if settings.partitioner != "iid" and column not in data.column_names:
        raise ValueError(f"Partition column {column!r} is missing from training data")
    partitioner.dataset = data
    if partitioner.num_partitions != peers:
        raise ValueError(f"Natural partitioning found {partitioner.num_partitions} clients; set --peers to match")
    return partitioner


def allocate_indices(data, settings, *, min_size=1, label_column="label"):
    """Keep tensor/token caches in source order while Flower allocates row IDs."""
    column = "__dasklearn_index__"
    if column in data.column_names:
        raise ValueError(f"Dataset uses the reserved column {column!r}")
    allocator = partition_dataset(data.add_column(column, list(range(len(data)))), settings,
                                  min_size=min_size, label_column=label_column)
    parts = [list(allocator.load_partition(peer)[column]) for peer in range(settings.participants)]
    if min(map(len, parts)) < max(min_size, settings.min_partition_size):
        raise ValueError("A client partition is smaller than the required minimum size")
    return parts


class HuggingFaceDataset:
    def __init__(self, settings):
        self.splits = load_splits(settings)
        self.partitioner = partition_dataset(self.splits["train"], settings, label_column=column_names(settings)[1])

    def load_partition(self, peer, split="train"):
        if split != "train":
            raise ValueError("Only the training split is partitioned")
        return self.partitioner.load_partition(peer)

    def load_split(self, split):
        return self.splits["test" if split == "valid" else split]
