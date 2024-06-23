import logging
import os
from typing import Dict, Optional, List, Tuple

import torch.nn as nn

from dasklearn.gradient_aggregation import GradientAggregationMethod
from dasklearn.gradient_aggregation.fedavg import FedAvg
from dasklearn.model_trainer import ModelTrainer
from dasklearn.session_settings import SessionSettings


class ModelManager:
    """
    This class manages the current ML model and training.
    """

    def __init__(self, model: Optional[nn.Module], settings: SessionSettings, participant_index: int):
        self.model: nn.Module = model
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.logger = logging.getLogger(self.__class__.__name__)

        dataset_base_path: str = self.settings.dataset_base_path or os.environ["HOME"]
        if self.settings.dataset in ["cifar10", "mnist"]:
            self.data_dir = os.path.join(dataset_base_path, "dfl-data")
        else:
            # The LEAF dataset
            self.data_dir = os.path.join(dataset_base_path, "leaf", "data", self.settings.dataset)

        self.model_trainer: ModelTrainer = ModelTrainer(self.data_dir, self.settings, self.participant_index)

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

    def train(self, local_steps: int, gradient_only: bool = False) -> Dict:
        return self.model_trainer.train(self.model, local_steps, gradient_only, self.settings.torch_device_name)

    def gradient_update(self, gradient_model):
        self.model_trainer.gradient_update(self.model, gradient_model)
