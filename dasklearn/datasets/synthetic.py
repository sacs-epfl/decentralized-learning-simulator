"""Small, offline classification workload for simulator validation, not a benchmark."""
import numpy as np
import torch
from torch.utils.data import Dataset


class TensorPartition(Dataset):
    def __init__(self, features, labels):
        self.features, self.labels = features, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"img": self.features[index], "label": self.labels[index]}


class SyntheticDataset:
    def __init__(self, settings):
        generator = torch.Generator().manual_seed(settings.seed + 10000)
        teacher = torch.randn(16, 4, generator=generator)
        total = max(16384, settings.participants * 1024) + 2048
        features = torch.randn(total, 16, generator=generator)
        labels = (features @ teacher + 0.2 * torch.randn(total, 4, generator=generator)).argmax(1)
        self.test = TensorPartition(features[-2048:], labels[-2048:])
        pools = [torch.where(labels[:-2048] == c)[0].numpy() for c in range(4)]
        rng = np.random.default_rng(settings.seed)
        pools = [rng.permutation(pool) for pool in pools]
        offsets = np.zeros(4, dtype=int)
        self.partitions = []
        used_indices = []
        for _ in range(settings.participants):
            probabilities = rng.dirichlet(np.full(4, settings.alpha)) if settings.partitioner == "dirichlet" else np.full(4, 0.25)
            classes = rng.choice(4, 256, p=probabilities)
            indices = []
            for c in classes:
                if offsets[c] >= len(pools[c]):
                    raise ValueError("Synthetic class pool exhausted")
                indices.append(pools[c][offsets[c]])
                offsets[c] += 1
            indices = np.asarray(indices)
            used_indices.extend(indices.tolist())
            self.partitions.append(TensorPartition(features[indices], labels[indices]))
        # Reuse previously unassigned examples without changing any client or
        # test data. Adaptive decisions must not consume the final test split.
        unused = np.setdiff1d(np.arange(total - 2048), used_indices)[:512]
        self.validation = TensorPartition(features[unused], labels[unused])

    def load_partition(self, peer, split="train"):
        return self.partitions[peer]

    def load_split(self, split):
        return self.validation if split == "validation" else self.test
