import copy
from typing import List

import torch
from torch import nn

from dlsim.core.gradient_aggregation import GradientAggregation


class FedAvg(GradientAggregation):

    @staticmethod
    def aggregate(models: List[nn.Module], weights: List[float]):
        if not weights:
            weights = [float(1. / len(models)) for _ in range(len(models))]
        else:
            assert len(weights) == len(models)

        with torch.no_grad():
            center_model = copy.deepcopy(models[0])
            for p in center_model.parameters():
                p.mul_(0)
            for m, w in zip(models, weights):
                for c1, p1 in zip(center_model.parameters(), m.parameters()):
                    c1.add_(w * p1)
            return center_model
