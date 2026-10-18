"""Offline handwritten digits with disjoint clients and a held-out test set."""
import numpy as np
import torch
from dasklearn.datasets.synthetic import TensorPartition


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
        if settings.participants > len(train) // 8:
            raise ValueError("Digits validation requires at least eight training examples per client")
        if settings.partitioner == "iid":
            partitions = [list(indices) for indices in np.array_split(rng.permutation(train), settings.participants)]
        else:
            partitions = [[] for _ in range(settings.participants)]
            train = np.asarray(train)
            for label in range(10):
                indices = rng.permutation(train[labels[train].numpy() == label])
                probabilities = rng.dirichlet(np.full(settings.participants, settings.alpha))
                counts = rng.multinomial(len(indices), probabilities)
                for client, chunk in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
                    partitions[client].extend(chunk.tolist())
            # Guarantee trainable clients without duplicating any examples.
            for indices in partitions:
                while len(indices) < 8:
                    donor = max(partitions, key=len)
                    indices.append(donor.pop())
        self.partitions = [TensorPartition(features[indices], labels[indices]) for indices in partitions]

    def load_partition(self, peer, split="train"):
        return self.partitions[peer]

    def load_split(self, split):
        return self.test
