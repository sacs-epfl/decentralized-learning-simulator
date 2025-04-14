import logging
import os
import time
import threading
from typing import Dict, List, Optional

import torch

from flwr_datasets import FederatedDataset

from dasklearn.datasets import create_dataset
from dasklearn.model_evaluator import ModelEvaluator
from dasklearn.model_manager import ModelManager
from dasklearn.model_trainer import ModelTrainer
from dasklearn.models import unserialize_model, serialize_model, create_model
from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.util import MICROSECONDS


logger = logging.getLogger(__name__)
model_trainers: Optional[List[ModelTrainer]] = None
dataset: Optional[FederatedDataset] = None
evaluator = None
lock = threading.Lock()


def train(settings: SessionSettings, params: Dict):
    global dataset, model_trainers
    model = params["model"]
    if isinstance(model, tuple):  # Happens for the first round, when the model is a tuple (None, 0)
        model = model[0]
    round_nr = params["round"]
    cur_time = params["time"]
    peer_id = params["peer"]
    compute_gradient = params["compute_gradient"] if "compute_gradient" in params else False
    gradient_model = params["gradient_model"] if "gradient_model" in params else None
    local_steps = params["local_steps"] if "local_steps" in params else settings.learning.local_steps

    if not dataset:
        dataset = create_dataset(settings)

    with lock:
        if model_trainers is None:
            model_trainers = [None] * settings.participants
    if not model:
        torch.manual_seed(settings.seed)
        model = create_model(settings.dataset, architecture=settings.model)

    if model_trainers[peer_id] is None:
        model_trainers[peer_id] = ModelTrainer(dataset, settings, peer_id)

    trainer: ModelTrainer = model_trainers[peer_id]
    if gradient_model:
        trainer.gradient_update(model, gradient_model)
    else:
        train_info = trainer.train(model, local_steps, compute_gradient, settings.torch_device_name)
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.debug("Peer %d training in round %d...", peer_id, round_nr)

    return [detached_model]


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
        logger.debug("Peer %d aggregating %d models in round %d...", peer_id, len(models), round_nr)
    else:
        logger.debug("Aggregating %d models in round %d...", len(models), round_nr)

    model_manager = ModelManager(None, settings, 0)
    for idx, model in enumerate(models):
        model_manager.process_incoming_trained_model(idx, model)

    start_time = time.time()
    agg_model = model_manager.aggregate_trained_models(weights)
    logger.debug("Model aggregation took %f s.", time.time() - start_time)
    return [agg_model]


def test(settings: SessionSettings, params: Dict):
    global dataset, evaluator
    model = params["model"]
    round_nr = params["round"]
    cur_time = params["time"]
    peer_id = params["peer"]
    logger.debug("Testing model in round %d...", round_nr)

    if not evaluator:
        evaluator = ModelEvaluator(dataset, settings)
    accuracy, loss = evaluator.evaluate_accuracy(model, device_name=settings.torch_device_name)

    accuracies_file_path: str = os.path.join(settings.data_dir, "accuracies_" + str(peer_id) + ".csv")
    if not os.path.exists(accuracies_file_path):
        with open(accuracies_file_path, "w") as accuracies_file:
            accuracies_file.write("algorithm,dataset,partitioner,alpha,learning_rate,peer,round,time,accuracy,loss\n")

    with open(accuracies_file_path, "a") as accuracies_file:
        accuracies_file.write("%s,%s,%s,%g,%g,%d,%d,%.2f,%f,%f\n" % (
            settings.algorithm, settings.dataset, settings.partitioner,
            settings.alpha, settings.learning.learning_rate, peer_id,
            round_nr, cur_time / MICROSECONDS, accuracy, loss))

    logger.info("[t=%.2f] Model accuracy (peer %d, round %d): %f, loss: %f", cur_time / MICROSECONDS, peer_id, round_nr, accuracy, loss)

    detached_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)
    return [detached_model]


def chunk(settings: SessionSettings, params: Dict) -> List[torch.Tensor]:
    model = params["model"]
    num_chunks = params["n"]
    chunks = ChunkManager.chunk_model(model, num_chunks)
    return chunks


def reconstruct_from_chunks(settings: SessionSettings, params: Dict) -> List[torch.nn.Module]:
    chunks = params["chunks"]
    model = create_model(settings.dataset, architecture=settings.model)
    model = ChunkManager.reconstruct_model(chunks, model)
    return [model]


def split_chunk(settings: SessionSettings, params: Dict) -> List[torch.Tensor]:
    chunk = params["chunk"].clone() / 2
    return [chunk]


def add_chunks(settings: SessionSettings, params: Dict) -> List[torch.Tensor]:
    chunks = params["chunks"]
    return [sum(chunks)]


def weighted_reconstruct_from_chunks(settings: SessionSettings, params: Dict) -> List[torch.nn.Module]:
    chunks = params["chunks"]
    weights = params["weights"]
    model = create_model(settings.dataset, architecture=settings.model)
    model = ChunkManager.weighted_reconstruct_model(chunks, model, weights)
    return [model]
