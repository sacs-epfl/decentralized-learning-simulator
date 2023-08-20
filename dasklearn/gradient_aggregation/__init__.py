from abc import abstractmethod
from enum import IntEnum
from typing import List

from torch import nn


class GradientAggregationMethod(IntEnum):
    FEDAVG = 1


class GradientAggregation:

    @staticmethod
    @abstractmethod
    def aggregate(models: List[nn.Module], weights: List[float]):
        pass
