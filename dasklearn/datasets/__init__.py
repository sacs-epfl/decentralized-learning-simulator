from typing import Optional

from dasklearn.datasets.Dataset import Dataset
from dasklearn.mappings.Linear import Linear
from dasklearn.session_settings import SessionSettings


def create_dataset(settings: SessionSettings, participant_index: int = 0, train_dir: Optional[str] = None, test_dir: Optional[str] = None) -> Dataset:
    mapping = Linear(1, settings.participants)
    if settings.dataset == "cifar10":
        from dasklearn.datasets.CIFAR10 import CIFAR10
        return CIFAR10(participant_index, 0, mapping, settings.partitioner,
                       train_dir=train_dir, test_dir=test_dir, shards=settings.participants, alpha=settings.alpha,
                       validation_size=settings.validation_set_fraction, seed=settings.seed)
    elif settings.dataset == "femnist":
        from dasklearn.datasets.femnist import Femnist
        return Femnist(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir)
    elif settings.dataset == "movielens":
        from dasklearn.datasets.MovieLens import MovieLens
        return MovieLens(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir,
                         random_seed=settings.seed)
    else:
        raise RuntimeError("Unknown dataset %s" % settings.dataset)
