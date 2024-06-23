import logging
import os
import time
import threading
from typing import Dict

import torch

from dasklearn.model_evaluator import ModelEvaluator
from dasklearn.model_manager import ModelManager
from dasklearn.models import unserialize_model, serialize_model, create_model
from dasklearn.session_settings import SessionSettings


logger = logging.getLogger(__name__)
model_managers = None
evaluator = None
lock = threading.Lock()


def train(settings: SessionSettings, params: Dict):
    global model_managers
    model = params["model"]
    round_nr = params["round"]
    cur_time = params["time"]
    peer_id = params["peer"]
    compute_gradient = params["compute_gradient"] if "compute_gradient" in params else False
    gradient_model = params["gradient_model"] if "gradient_model" in params else None
    local_steps = params["local_steps"] if "local_steps" in params else settings.learning.local_steps

    with lock:
        if model_managers is None:
            model_managers = [None] * settings.participants
    if not model:
        torch.manual_seed(settings.seed)
        model = create_model(settings.dataset, architecture=settings.model)

    if model_managers[peer_id] is None:
        model_managers[peer_id] = ModelManager(model, settings, peer_id)
    else:
        model_managers[peer_id].model = model
    if gradient_model:
        model_managers[peer_id].gradient_update(gradient_model)
    else:
        train_info = model_managers[peer_id].train(local_steps, compute_gradient)
        if train_info["validation_loss_global"] is not None:
            with open(os.path.join(settings.data_dir, "validation_losses.csv"), "a") as loss_file:
                loss_file.write("%d,%d,%f,%f\n" % (peer_id, round_nr, cur_time, train_info["validation_loss_global"]))

    if compute_gradient:
        grad = model.gradient
        detached_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)
        detached_model.gradient = []
        for g in grad:
            g = g.cpu()
            detached_model.gradient.append(g)
    else:
        detached_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)

    del model_managers[peer_id].model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Peer %d training in round %d...", peer_id, round_nr)

    return detached_model


def compute_gradient(settings: SessionSettings, params: Dict):
    params["compute_gradient"] = True
    return train(settings, params)


def gradient_update(settings: SessionSettings, params: Dict):
    return train(settings, params)


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
    with open(os.path.join(settings.data_dir, "accuracies_" + str(peer_id) + ".csv"), "a") as accuracies_file:
        accuracies_file.write("%d,%d,%f,%f,%f\n" % (peer_id, round_nr, cur_time, accuracy, loss))
    logger.info("Model accuracy: %f, loss: %f", accuracy, loss)

    detached_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)
    return detached_model
