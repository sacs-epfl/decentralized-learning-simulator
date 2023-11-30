import logging
import os
from typing import Dict

import torch

from dasklearn.model_evaluator import ModelEvaluator
from dasklearn.model_manager import ModelManager
from dasklearn.models import unserialize_model, serialize_model, create_model
from dasklearn.session_settings import SessionSettings


logger = logging.getLogger(__name__)
evaluator = None


def train(settings: SessionSettings, params: Dict):
    model = params["model"]
    round_nr = params["round"]
    peer_id = params["peer"]

    if not model:
        torch.manual_seed(settings.seed)
        copied_model = create_model(settings.dataset, architecture=settings.model)
    else:
        # Make a copy of the model so multiple workers are not training the same model
        copied_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)

    model_manager = ModelManager(copied_model, settings, peer_id)
    model_manager.train()

    logger.info("Peer %d training in round %d...", peer_id, round_nr)
    return copied_model


def aggregate(settings: SessionSettings, params: Dict):
    models = params["models"]
    round_nr = params["round"]
    peer_id = params["peer"]
    if peer_id is not None:
        logger.info("Peer %d aggregating %d models in round %d...", peer_id, len(models), round_nr)
    else:
        logger.info("Aggregating %d models in round %d...", len(models), round_nr)

    model_manager = ModelManager(None, settings, 0)
    for peer_id, model in models.items():
        model_manager.process_incoming_trained_model(peer_id, model)

    return model_manager.aggregate_trained_models()


def test(settings: SessionSettings, params: Dict):
    global evaluator
    model = params["model"]
    round_nr = params["round"]
    cur_time = params["time"]
    peer_id = params["peer"]
    logger.info("Testing model in round %d...", round_nr)

    dataset_base_path: str = settings.dataset_base_path or os.environ["HOME"]
    if settings.dataset in ["cifar10", "mnist", "fashionmnist"]:
        data_dir = os.path.join(dataset_base_path, "dfl-data")
    else:
        # The LEAF dataset
        data_dir = os.path.join(dataset_base_path, "leaf", settings.dataset)

    if not evaluator:
        evaluator = ModelEvaluator(data_dir, settings)
    accuracy, loss = evaluator.evaluate_accuracy(model, device_name=settings.torch_device_name)
    with open(os.path.join(settings.data_dir, "accuracies.csv"), "a") as accuracies_file:
        accuracies_file.write("%d,%d,%f,%f,%f\n" % (peer_id, round_nr, cur_time, accuracy, loss))
    logger.info("Model accuracy: %f, loss: %f", accuracy, loss)
    return model
