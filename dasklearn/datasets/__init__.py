from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

from dasklearn.session_settings import SessionSettings


def create_dataset(settings: SessionSettings) -> FederatedDataset:
    # Create the partitioner
    # TODO add support for k-shards
    if settings.partitioner == "dirichlet":
        partitioner = DirichletPartitioner(num_partitions=settings.participants, partition_by="label", alpha=settings.alpha)
    else:
        partitioner = IidPartitioner(num_partitions=settings.participants)

    if settings.dataset == "cifar10":
        return FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": partitioner})
    elif settings.dataset == "google_speech":
        return FederatedDataset(dataset="speech_commands", subset = "v0.02", partitioners={"train": partitioner}, trust_remote_code=True)
    else:
        raise RuntimeError("Unknown dataset %s" % settings.dataset)
