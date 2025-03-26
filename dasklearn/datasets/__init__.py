from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, NaturalIdPartitioner

from dasklearn.session_settings import SessionSettings


def create_dataset(settings: SessionSettings) -> FederatedDataset:
    # Create the partitioner
    # TODO add support for k-shards
    if settings.partitioner == "dirichlet":
        partitioner = DirichletPartitioner(num_partitions=settings.participants, partition_by="label", alpha=settings.alpha)
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
