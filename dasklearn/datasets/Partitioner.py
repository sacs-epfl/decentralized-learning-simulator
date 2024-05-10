from random import Random

import numpy as np

""" Adapted from https://pytorch.org/tutorials/intermediate/dist_tuto.html """


class Partition(object):
    """
    Class for holding the data partition

    """

    def __init__(self, data, index):
        """
        Constructor. Caches the data and the indices

        Parameters
        ----------
        data : indexable
        index : list
            A list of indices

        """
        self.data = data
        self.index = index

    def __len__(self):
        """
        Function to retrieve the length

        Returns
        -------
        int
            Number of items in the data

        """
        return len(self.index)

    def __getitem__(self, index):
        """
        Retrieves the item in data with the given index

        Parameters
        ----------
        index : int

        Returns
        -------
        Data
            The data sample with the given `index` in the dataset

        """
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """
    Class to partition the dataset

    """

    def __init__(self, data, sizes=[1.0], seed=1234):
        """
        Constructor. Partitions the data according the parameters

        Parameters
        ----------
        data : indexable
            An indexable list of data items
        sizes : list(float)
            A list of fractions for each process
        seed : int, optional
            Seed for generating a random subset

        """
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, rank):
        """
        Get the partition for the process with the given `rank`

        Parameters
        ----------
        rank : int
            Rank of the current process

        Returns
        -------
        Partition
            The dataset partition of the current process

        """
        return Partition(self.data, self.partitions[rank])


class SimpleDataPartitioner(DataPartitioner):
    """
    Class to partition the dataset

    """

    def __init__(self, data, sizes=[1.0]):
        """
        Constructor. Partitions the data according the parameters

        Parameters
        ----------
        data : indexable
            An indexable list of data items
        sizes : list(float)
            A list of fractions for each process

        """
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]


class KShardDataPartitioner(DataPartitioner):
    """
    Class to partition the dataset

    """

    def __init__(self, data, sizes=[1.0], shards=1, seed=1234):
        """
        Constructor. Partitions the data according the parameters

        Parameters
        ----------
        data : indexable
            An indexable list of data items
        sizes : list(float)
            A list of fractions for each process
        shards : int
            Number of shards to allot to process
        seed : int, optional
            Seed for generating a random subset

        """
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng = Random()
        rng.seed(seed)

        for frac in sizes:
            self.partitions.append([])
            for _ in range(shards):
                start = rng.randint(0, len(indexes) - 1)
                part_len = int(frac * data_len) // shards
                if start + part_len > len(indexes):
                    self.partitions[-1].extend(indexes[start:])
                    self.partitions[-1].extend(
                        indexes[: (start + part_len - len(indexes))]
                    )
                    indexes = indexes[(start + part_len - len(indexes)) : start]
                else:
                    self.partitions[-1].extend(indexes[start : start + part_len])
                    index_start = indexes[:start]
                    index_start.extend(indexes[start + part_len :])
                    indexes = index_start


class DirichletDataPartitioner(DataPartitioner):
    """
    Class to partition the dataset using Dirichlet Function

    """

    def __init__(self, data, sizes=[1.0], seed=1235, alpha=0.1, validation_set=False, num_classes=10):
        """
        Constructor. Partitions the data according the parameters

        Parameters
        ----------
        data : indexable
            An indexable list of data items
        sizes : list(float)
            A list of fractions for each process
        shards : int
            Number of shards to allot to process
        seed : int, optional
            Seed for generating a random subset
        alpha : float
            Degree of heterogeneity. Lower is more heterogeneous.

        Code below modified from https://github.com/gong-xuan/FedKD/blob/master/dataset/data_cifar.py#L30
        """
        self.validation_set = validation_set
        self.data = data
        self.num_classes = num_classes
        num_peers = len(sizes)

        self.partitions = [None] * num_peers
        np.random.seed(43)
        split_arr = np.random.dirichlet([alpha] * num_peers, num_classes)
        for cls_idx in range(num_classes):
            idx = np.where(np.asarray(data.targets) == cls_idx)[0]
            totaln = idx.shape[0]
            idx_start = 0
            for i in range(num_peers):
                if i == num_peers - 1:
                    cur_idx = idx[idx_start:]
                else:
                    idx_end = idx_start + int(split_arr[cls_idx][i] * totaln)
                    cur_idx = idx[idx_start: idx_end]
                    idx_start = idx_end
                if cur_idx.size == 0:
                    continue
                if self.partitions[i] is None:
                    self.partitions[i] = cur_idx
                else:
                    self.partitions[i] = np.r_[(self.partitions[i], cur_idx)]

        for i in range(len(self.partitions)):
            self.partitions[i] = list(self.partitions[i])