"""Cached CIFAR-10 tensors, disjoint clients, and a separate tuning split.

Normalization matches the existing CIFAR pipeline. No augmentation is used.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from dasklearn.datasets.huggingface import allocate_indices, load_splits

class CifarPartition(Dataset):
    def __init__(self, features, labels, indices, num_classes=10):
        self.features, self.labels = features, labels
        self.num_classes = num_classes
        self.indices = np.asarray(indices, dtype=np.int64)
        self.subsets = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        return {"img": self.features[idx], "label": self.labels[idx]}

    def with_transform(self, transform):
        # Already normalized tensors; the legacy loader's PIL transforms would
        # normalize twice or resize the native 32x32 model inputs.
        return self

    def evaluation_subset(self, count):
        if count >= len(self):
            return self
        if count < self.num_classes or count % self.num_classes:
            raise ValueError('CIFAR evaluation subset must have an equal positive count per class')
        if count not in self.subsets:
            rng = np.random.default_rng(2026)
            labels = self.labels.numpy()
            chunks = [rng.permutation(self.indices[labels[self.indices] == label])[:count // self.num_classes]
                      for label in range(self.num_classes)]
            if any(len(chunk) != count // self.num_classes for chunk in chunks):
                raise ValueError('Not enough examples per CIFAR evaluation class')
            self.subsets[count] = CifarPartition(self.features, self.labels, np.concatenate(chunks), self.num_classes)
        return self.subsets[count]


class Cifar10Dataset:
    models = ('gnlenet', 'lenet')
    num_classes = 10
    mean = (.4914, .4822, .4465)
    std = (.2470, .2435, .2616)

    def __init__(self, settings):
        if settings.model not in self.models:
            raise ValueError(f"Native CIFAR tensor loading supports {self.models}")
        splits = load_splits(settings)
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)

        def tensors(split):
            images = np.stack([np.asarray(image.convert("RGB")) for image in split["img"]])
            features = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255)
            return features.sub_(mean).div_(std), torch.tensor(list(split["label"]))

        indices = allocate_indices(splits["train"], settings, min_size=8)
        features, labels = tensors(splits["train"])
        self.partitions = [CifarPartition(features, labels, part, self.num_classes) for part in indices]
        self.validation = CifarPartition(*tensors(splits["validation"]), np.arange(len(splits["validation"])), self.num_classes)
        self.test = CifarPartition(*tensors(splits["test"]), np.arange(len(splits["test"])), self.num_classes)

    def load_partition(self, peer, split="train"):
        return self.partitions[peer]

    def load_split(self, split):
        return {"validation": self.validation, "test": self.test}[split]
