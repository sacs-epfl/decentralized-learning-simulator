import logging
import os
import time
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
    cur_time = params["time"]
    peer_id = params["peer"]

    if not model:
        torch.manual_seed(settings.seed)
        model = create_model(settings.dataset, architecture=settings.model)

    model_manager = ModelManager(model, settings, peer_id)
    train_info = model_manager.train()

    if train_info["validation_loss_global"] is not None:
        with open(os.path.join(settings.data_dir, "validation_losses.csv"), "a") as loss_file:
            loss_file.write("%d,%d,%f,%f\n" % (peer_id, round_nr, cur_time, train_info["validation_loss_global"]))

    detached_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)

    del model_manager.model
    del model_manager
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Peer %d training in round %d...", peer_id, round_nr)

    return detached_model


def aggregate(settings: SessionSettings, params: Dict):
    models = params["models"]
    round_nr = params["round"]
    peer_id = params["peer"]
    weights = params["weights"] if "weights" in params else None
    if peer_id is not None:
        logger.info("Peer %d aggregating %d models in round %d...", peer_id, len(models), round_nr)
    else:
        logger.info("Aggregating %d models in round %d...", len(models), round_nr)

    model_manager = ModelManager(None, settings, 0)
    for idx, model in enumerate(models):
        model_manager.process_incoming_trained_model(idx, model)

    start_time = time.time()
    agg_model = model_manager.aggregate_trained_models(weights)

    logger.info("Model aggregation took %f s.", time.time() - start_time)
    return agg_model


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
        data_dir = os.path.join(dataset_base_path, "leaf", "data", settings.dataset)

    if not evaluator:
        evaluator = ModelEvaluator(data_dir, settings)
    accuracy, loss = evaluator.evaluate_accuracy(model, device_name=settings.torch_device_name)
    with open(os.path.join(settings.data_dir, "accuracies.csv"), "a") as accuracies_file:
        accuracies_file.write("%d,%d,%f,%f,%f\n" % (peer_id, round_nr, cur_time, accuracy, loss))
    logger.info("Model accuracy: %f, loss: %f", accuracy, loss)

    detached_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)
    return detached_model
