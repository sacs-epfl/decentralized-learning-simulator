import copy
from typing import List

import torch
from torch import nn

from dasklearn.gradient_aggregation import GradientAggregation


class FedAvg(GradientAggregation):

    @staticmethod
    def aggregate(models: List[nn.Module], weights: List[float]):
        if not weights:
            weights = [float(1. / len(models)) for _ in range(len(models))]
        else:
            assert len(weights) == len(models)

        with torch.no_grad():
            center_model = copy.deepcopy(models[0])
            for name, target in center_model.state_dict().items():
                values = [model.state_dict()[name] for model in models]
                if target.is_floating_point() or target.is_complex():
                    target.zero_()
                    for value, weight in zip(values, weights):
                        target.add_(value, alpha=weight)
                else:
                    target.copy_(torch.stack(values).max(dim=0).values)
            return center_model
