"""Cached CIFAR-10 tensors, disjoint clients, and a separate tuning split.

Normalization matches the existing CIFAR pipeline. No augmentation is used.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CifarPartition(Dataset):
    def __init__(self, features, labels, indices):
        self.features, self.labels = features, labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.subsets = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        return {"img": self.features[idx], "label": self.labels[idx]}

    def with_transform(self, transform):
        # Already normalized tensors; the legacy loader's PIL transforms would
        # normalize twice. Only the native 32x32 GNLeNet/LeNet path uses this.
        return self

    def evaluation_subset(self, count):
        if count >= len(self):
            return self
        if count < 10 or count % 10:
            raise ValueError('CIFAR evaluation subset must have an equal positive count per class')
        if count not in self.subsets:
            rng = np.random.default_rng(2026)
            labels = self.labels.numpy()
            chunks = [rng.permutation(self.indices[labels[self.indices] == label])[:count // 10]
                      for label in range(10)]
            if any(len(chunk) != count // 10 for chunk in chunks):
                raise ValueError('Not enough examples per CIFAR evaluation class')
            self.subsets[count] = CifarPartition(self.features, self.labels, np.concatenate(chunks))
        return self.subsets[count]


def partition_indices(labels, participants, partitioner, alpha, seed):
    rng = np.random.default_rng(seed)
    if participants < 1 or participants * 8 > len(labels):
        raise ValueError("CIFAR validation needs at least eight examples per client")
    if partitioner == "iid":
        return [part.tolist() for part in np.array_split(rng.permutation(len(labels)), participants)]
    if partitioner != "dirichlet" or alpha <= 0:
        raise ValueError("Native CIFAR supports IID or positive-alpha Dirichlet partitioning")
    parts = [[] for _ in range(participants)]
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        counts = rng.multinomial(len(indices), rng.dirichlet(np.full(participants, alpha)))
        for peer, chunk in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
            parts[peer].extend(chunk.tolist())
    # Move examples, never duplicate them. Record the resulting size histogram.
    for part in parts:
        while len(part) < 8:
            part.append(max(parts, key=len).pop())
    return parts


class Cifar10Dataset:
    def __init__(self, settings):
        from torchvision.datasets import CIFAR10
        if settings.model not in ["gnlenet", "lenet"]:
            raise ValueError("Native CIFAR tensor loading currently supports gnlenet/lenet")
        root = str(Path(settings.dataset_base_path).expanduser())
        archive = Path(root) / "cifar10-hf.npz"
        if archive.exists():
            with np.load(archive, allow_pickle=False) as data:
                train_images, labels = data["train_images"], data["train_labels"]
                test_images, test_labels = data["test_images"], data["test_labels"]
        else:
            train, test = CIFAR10(root, train=True, download=True), CIFAR10(root, train=False, download=True)
            train_images, labels = train.data, np.asarray(train.targets)
            test_images, test_labels = test.data, np.asarray(test.targets)
        rng = np.random.default_rng(1729)  # Fixed held-out data across training seeds.
        validation, training = [], []
        for label in range(10):
            ids = rng.permutation(np.flatnonzero(labels == label))
            validation.extend(ids[:500])
            training.extend(ids[500:])
        mean = torch.tensor([.4914, .4822, .4465]).view(1, 3, 1, 1)
        std = torch.tensor([.2470, .2435, .2616]).view(1, 3, 1, 1)

        def features(images):
            return (torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255) - mean) / std

        train_features, train_labels = features(train_images), torch.tensor(labels)
        self.validation = CifarPartition(train_features, train_labels, validation)
        self.test = CifarPartition(features(test_images), torch.tensor(test_labels), np.arange(len(test_labels)))
        training = np.asarray(training)
        indices = partition_indices(labels[training], settings.participants, settings.partitioner, settings.alpha, settings.seed)
        self.partitions = [CifarPartition(train_features, train_labels, training[part]) for part in indices]

    def load_partition(self, peer, split="train"):
        return self.partitions[peer]

    def load_split(self, split):
        return self.validation if split == "validation" else self.test
