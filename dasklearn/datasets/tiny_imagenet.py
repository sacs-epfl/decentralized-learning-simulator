"""Native 64x64 Tiny ImageNet tensors, disjoint clients, fixed tuning/eval splits."""
import numpy as np
import torch
from dasklearn.datasets.huggingface import allocate_indices, load_splits


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
        splits = load_splits(settings)
        mean = torch.tensor([.4802, .4481, .3975]).view(1, 3, 1, 1)
        std = torch.tensor([.2302, .2265, .2262]).view(1, 3, 1, 1)

        def tensors(split):
            images = np.stack([np.asarray(image.convert("RGB")) for image in split["image"]])
            features = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255)
            return features.sub_(mean).div_(std), torch.tensor(list(split["label"]))

        indices = allocate_indices(splits["train"], settings, min_size=8)
        features, labels = tensors(splits["train"])
        self.partitions = [TinyPartition(features, labels, part) for part in indices]
        self.tuning = TinyPartition(*tensors(splits["validation"]), np.arange(len(splits["validation"])))
        self.valid = TinyPartition(*tensors(splits["test"]), np.arange(len(splits["test"])))

    def load_partition(self, peer, split="train"):
        return self.partitions[peer]

    def load_split(self, split):
        return {"validation": self.tuning, "test": self.valid, "valid": self.valid}[split]
