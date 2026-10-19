"""Native 64x64 Tiny ImageNet tensors, disjoint clients, fixed tuning/eval splits."""
from pathlib import Path
import numpy as np
import torch
from dasklearn.datasets.cifar10 import partition_indices


class TinyPartition:
    def __init__(self, features, labels, indices):
        self.features, self.labels = features, labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.subsets = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        return {"image": self.features[idx], "label": self.labels[idx]}

    def with_transform(self, transform):
        # Already normalized, native 64x64 tensors. Avoid legacy 224x224 resizing.
        return self

    def evaluation_subset(self, count):
        if count >= len(self):
            return self
        if count < 200 or count % 200:
            raise ValueError("Tiny ImageNet evaluation subset must contain an equal count per class")
        if count not in self.subsets:
            rng = np.random.default_rng(2026)
            labels = self.labels.numpy()
            ids = np.concatenate([rng.permutation(self.indices[labels[self.indices] == label])[:count // 200]
                                  for label in range(200)])
            assert len(ids) == count
            self.subsets[count] = TinyPartition(self.features, self.labels, ids)
        return self.subsets[count]


class TinyImagenetDataset:
    def __init__(self, settings):
        root = Path(settings.dataset_base_path)
        mean = torch.tensor([.4802, .4481, .3975]).view(1, 3, 1, 1)
        std = torch.tensor([.2302, .2265, .2262]).view(1, 3, 1, 1)

        def load(split):
            # Copy into one normalized float tensor; the original image archive
            # stays memory mapped and is not duplicated in every client partition.
            images = np.load(root / f"{split}_images.npy", mmap_mode="r")
            features = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2).float()
            features.div_(255).sub_(mean).div_(std)
            labels = torch.from_numpy(np.load(root / f"{split}_labels.npy"))
            return features, labels

        features, labels = load("train")
        rng = np.random.default_rng(1729)
        tuning, training = [], []
        for label in range(200):
            ids = rng.permutation(np.flatnonzero(labels.numpy() == label))
            tuning.extend(ids[:25])
            training.extend(ids[25:])
        self.tuning = TinyPartition(features, labels, tuning)
        training = np.asarray(training)
        indices = partition_indices(labels[training].numpy(), settings.participants, settings.partitioner, settings.alpha, settings.seed)
        self.partitions = [TinyPartition(features, labels, training[part]) for part in indices]
        valid_features, valid_labels = load("valid")
        self.valid = TinyPartition(valid_features, valid_labels, np.arange(len(valid_labels)))

    def load_partition(self, peer, split="train"):
        return self.partitions[peer]

    def load_split(self, split):
        return self.tuning if split == "validation" else self.valid
