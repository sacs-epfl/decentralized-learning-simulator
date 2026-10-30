"""Cached CIFAR-10 tensors, disjoint clients, and a separate tuning split.

Normalization matches the existing CIFAR pipeline. No augmentation is used.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from dasklearn.datasets.huggingface import allocate_indices, load_splits

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


class Cifar10Dataset:
    def __init__(self, settings):
        if settings.model not in ["gnlenet", "lenet"]:
            raise ValueError("Native CIFAR tensor loading currently supports gnlenet/lenet")
        splits = load_splits(settings)
        mean = torch.tensor([.4914, .4822, .4465]).view(1, 3, 1, 1)
        std = torch.tensor([.2470, .2435, .2616]).view(1, 3, 1, 1)

        def tensors(split):
            images = np.stack([np.asarray(image.convert("RGB")) for image in split["img"]])
            features = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255)
            return features.sub_(mean).div_(std), torch.tensor(list(split["label"]))

        indices = allocate_indices(splits["train"], settings, min_size=8)
        features, labels = tensors(splits["train"])
        self.partitions = [CifarPartition(features, labels, part) for part in indices]
        self.validation = CifarPartition(*tensors(splits["validation"]), np.arange(len(splits["validation"])))
        self.test = CifarPartition(*tensors(splits["test"]), np.arange(len(splits["test"])))

    def load_partition(self, peer, split="train"):
        return self.partitions[peer]

    def load_split(self, split):
        return {"validation": self.validation, "test": self.test}[split]
