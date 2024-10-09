import logging
from typing import Dict, List, Optional

import torch.nn as nn

from flwr_datasets import FederatedDataset

from dasklearn.gradient_aggregation import GradientAggregationMethod
from dasklearn.gradient_aggregation.fedavg import FedAvg
from dasklearn.model_trainer import ModelTrainer
from dasklearn.session_settings import SessionSettings


class ModelManager:
    """
    This class manages the current ML model and training.
    """

    def __init__(self, dataset: FederatedDataset, settings: SessionSettings, participant_index: int):
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.logger = logging.getLogger(self.__class__.__name__)

        # Keeps track of the incoming trained models as aggregator
        self.incoming_trained_models: Dict[int, nn.Module] = {}

    def process_incoming_trained_model(self, peer_id: int, incoming_model: nn.Module):
        if peer_id in self.incoming_trained_models:
            # We already processed this model
            return

        self.incoming_trained_models[peer_id] = incoming_model

    def reset_incoming_trained_models(self):
        self.incoming_trained_models = {}

    def get_aggregation_method(self):
        if self.settings.gradient_aggregation == GradientAggregationMethod.FEDAVG:
            return FedAvg

    def aggregate_trained_models(self, weights: List[float] = None) -> Optional[nn.Module]:
        models = [model for model in self.incoming_trained_models.values()]
        return self.get_aggregation_method().aggregate(models, weights=weights)
