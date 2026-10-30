"""Offline handwritten digits with disjoint clients and a held-out test set."""
import numpy as np
import torch
from dasklearn.datasets.synthetic import TensorPartition
from dasklearn.datasets.huggingface import allocate_indices


class DigitsDataset:
    def __init__(self, settings):
        from sklearn.datasets import load_digits
        features, labels = load_digits(return_X_y=True)
        rng = np.random.default_rng(settings.seed + 10000)
        train, test = [], []
        for label in range(10):
            indices = rng.permutation(np.where(labels == label)[0])
            split = len(indices) // 4
            test.extend(indices[:split])
            train.extend(indices[split:])
        features = torch.tensor(features / 16, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        self.test = TensorPartition(features[test], labels[test])
        train = np.asarray(train)
        from datasets import Dataset
        partitions = allocate_indices(Dataset.from_dict({"label": labels[train].numpy()}), settings, min_size=8)
        self.partitions = [TensorPartition(features[train[indices]], labels[train[indices]]) for indices in partitions]

    def load_partition(self, peer, split="train"):
        return self.partitions[peer]

    def load_split(self, split):
        if split != "test":
            raise ValueError("The offline digits workload only provides a test split")
        return self.test
