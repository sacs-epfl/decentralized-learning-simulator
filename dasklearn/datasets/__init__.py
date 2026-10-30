from __future__ import annotations

from dasklearn.session_settings import SessionSettings


def create_dataset(settings: SessionSettings) -> FederatedDataset:
    if settings.dataset == "text":
        from dasklearn.datasets.text import TextDataset
        return TextDataset(settings)
    if settings.dataset == "tiny_imagenet" and settings.dataset_base_path:
        from dasklearn.datasets.tiny_imagenet import TinyImagenetDataset
        return TinyImagenetDataset(settings)
    if settings.dataset == "cifar10" and settings.dataset_base_path:
        from dasklearn.datasets.cifar10 import Cifar10Dataset
        return Cifar10Dataset(settings)
    if settings.dataset == "digits":
        from dasklearn.datasets.digits import DigitsDataset
        return DigitsDataset(settings)
    if settings.dataset == "synthetic":
        from dasklearn.datasets.synthetic import SyntheticDataset
        return SyntheticDataset(settings)

    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, NaturalIdPartitioner

    # Create the partitioner
    # TODO add support for k-shards
    if settings.partitioner == "dirichlet":
        partitioner = DirichletPartitioner(num_partitions=settings.participants, partition_by="label", alpha=settings.alpha, seed=settings.seed)
    elif settings.partitioner == "natural":
        partition_columns = {"femnist": "writer_id", "google_speech": "speaker_id"}
        if settings.dataset not in partition_columns:
            raise RuntimeError("Natural partitioning is not supported for dataset %s" % settings.dataset)
        partitioner = NaturalIdPartitioner(partition_by=partition_columns[settings.dataset])
    else:
        partitioner = IidPartitioner(num_partitions=settings.participants)

    if settings.dataset == "cifar10":
        return FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": partitioner})
    elif settings.dataset == "femnist":
        return FederatedDataset(dataset="coscotuff/femnist", partitioners={"train": partitioner})
    elif settings.dataset == "google_speech":
        return FederatedDataset(dataset="speech_commands", subset = "v0.02", partitioners={"train": partitioner}, trust_remote_code=True)
    elif settings.dataset == "tiny_imagenet":
        return FederatedDataset(dataset="zh-plus/tiny-imagenet", partitioners={"train": partitioner})
    else:
        raise RuntimeError("Unknown dataset %s" % settings.dataset)
