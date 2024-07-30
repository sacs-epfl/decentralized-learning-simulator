import logging
import os
import time
import threading
from typing import Dict

import torch

from dasklearn.model_evaluator import ModelEvaluator
from dasklearn.model_manager import ModelManager
from dasklearn.models import unserialize_adapter, unserialize_model, serialize_model, create_model
from dasklearn.models.lora import LORALayer
from dasklearn.session_settings import SessionSettings


logger = logging.getLogger(__name__)
model_managers = None
pretrained_model = None
evaluator = None
lock = threading.Lock()


def train(settings: SessionSettings, params: Dict):
    global model_managers, pretrained_model
    adapter = None
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

    if settings.finetune:
        adapter = model
        if not pretrained_model:
            # Initialize the pre-trained model if we're finetuning
            torch.manual_seed(settings.seed)
            pretrained_model = create_model(settings.dataset, architecture=settings.model, pretrained=settings.finetune)

    if not settings.finetune and not model:
        torch.manual_seed(settings.seed)
        model = create_model(settings.dataset, architecture=settings.model)
    elif settings.finetune and not model:
        adapter = LORALayer(pretrained_model.fc)

    if settings.finetune:
        pretrained_model.fc = adapter
    
    if model_managers[peer_id] is None:
        model_managers[peer_id] = ModelManager(pretrained_model if settings.finetune else model, settings, peer_id)
    else:
        model_managers[peer_id].model = pretrained_model if settings.finetune else model

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
    elif settings.finetune:
        # TODO Here, we are serializing the ENTIRE LoRA layer - including the adaptable weights. We need to fix this!
        detached_adapter = unserialize_adapter(serialize_model(pretrained_model.fc), pretrained_model.fc.adapted_layer)
    else:
        detached_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)

    if not settings.finetune:  # We want to keep the pre-trained model
        del model_managers[peer_id].model
    else:
        # Reset the pre-trained model by removing the LoRA layer
        pretrained_model.fc = pretrained_model.fc.adapted_layer
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Peer %d training in round %d...", peer_id, round_nr)

    return detached_adapter if settings.finetune else detached_model


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
    global evaluator, pretrained_model
    model = params["model"]
    round_nr = params["round"]
    cur_time = params["time"]
    peer_id = params["peer"]
    logger.info("Testing model in round %d...", round_nr)

    # If we're fine-tuning, reconstruct the model
    if settings.finetune:
        pretrained_model.fc = model
        model = pretrained_model

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

    if settings.finetune:
        # TODO Here, we are serializing the ENTIRE LoRA layer - including the adaptable weights. We need to fix this!
        detached_adapter = unserialize_adapter(serialize_model(pretrained_model.fc), pretrained_model.fc.adapted_layer)
    else:
        detached_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)
    
    return detached_adapter if settings.finetune else detached_model
