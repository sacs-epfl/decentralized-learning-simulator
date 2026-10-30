from __future__ import annotations

import logging
import copy
import os
import time
import threading
from typing import Dict, List, Optional

import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from flwr_datasets import FederatedDataset

from dasklearn.datasets import create_dataset
from dasklearn.model_evaluator import ModelEvaluator
from dasklearn.model_manager import ModelManager
from dasklearn.model_trainer import ModelTrainer
from dasklearn.models import copy_model_state, create_model, create_model_structure
from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.util import MICROSECONDS


logger = logging.getLogger(__name__)
model_trainers: Optional[List[ModelTrainer]] = None
dataset: Optional[FederatedDataset] = None
evaluator = None
last_test_metrics = None
lock = threading.Lock()


def initialize(settings: SessionSettings, params: Dict):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings.seed)
        return [create_model(settings.dataset, architecture=settings.model, settings=settings)]


def training_seed(settings, params):
    return (settings.seed + 1000003 * params["peer"] + 9176 * params["round"]) % (2 ** 63)


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
        model = create_model(settings.dataset, architecture=settings.model, settings=settings)
    else:
        # A model can feed several training tasks, possibly on one worker.
        # Never mutate the shared predecessor value in the workflow DAG.
        model = copy.deepcopy(model)

    if model_trainers[peer_id] is None:
        model_trainers[peer_id] = ModelTrainer(dataset, settings, peer_id)

    trainer: ModelTrainer = model_trainers[peer_id]
    if gradient_model:
        trainer.gradient_update(model, gradient_model)
    else:
        task_seed = training_seed(settings, params)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(task_seed)
            train_info = trainer.train(model, local_steps, compute_gradient, settings.torch_device_name, seed=task_seed)
        if train_info["validation_loss_global"] is not None:
            with open(os.path.join(settings.data_dir, "validation_losses.csv"), "a") as loss_file:
                loss_file.write("%d,%d,%f,%f\n" % (peer_id, round_nr, cur_time, train_info["validation_loss_global"]))

    if compute_gradient:
        grad = model.gradient
        detached_model = copy_model_state(model, settings.dataset, architecture=settings.model)
        detached_model.gradient = []
        for g in grad:
            g = g.cpu()
            detached_model.gradient.append(g)
    else:
        detached_model = copy_model_state(model, settings.dataset, architecture=settings.model)

    if settings.release_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.debug("Peer %d training in round %d...", peer_id, round_nr)

    return [detached_model]


def compute_gradient(settings: SessionSettings, params: Dict):
    params["compute_gradient"] = True
    return train(settings, params)


def gradient_update(settings: SessionSettings, params: Dict):
    return train(settings, params)


def aggregate(settings: SessionSettings, params: Dict):
    global dataset
    models = params["models"]
    round_nr = params["round"]
    peer_id = params["peer"]
    weights = params["weights"] if "weights" in params else None
    if params.get("weight_by_examples"):
        if dataset is None:
            dataset = create_dataset(settings)
        counts = [len(dataset.load_partition(peer, "train")) for peer in params["sample_peers"]]
        if not counts or min(counts) <= 0:
            raise ValueError("FedAvg requires nonempty selected client partitions")
        weights = [count / sum(counts) for count in counts]
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
    global dataset, evaluator, last_test_metrics
    model = params["model"]
    round_nr = params["round"]
    cur_time = params["time"]
    peer_id = params["peer"]
    logger.debug("Testing model in round %d...", round_nr)

    if dataset is None:
        dataset = create_dataset(settings)
    if not evaluator:
        evaluator = ModelEvaluator(dataset, settings)
    current_evaluator = evaluator
    if params.get('evaluation_split', settings.evaluation_split) != settings.evaluation_split:
        evaluation_settings = copy.copy(settings)
        evaluation_settings.evaluation_split = params['evaluation_split']
        current_evaluator = ModelEvaluator(dataset, evaluation_settings)
    metrics = params.get("cached_metrics")
    accuracy, loss = metrics if metrics is not None else current_evaluator.evaluate_accuracy(
        model, device_name=settings.torch_device_name, evaluation_examples=params.get("evaluation_examples"))
    last_test_metrics = (accuracy, loss)

    accuracies_file_path: str = os.path.join(settings.data_dir, params.get("output_prefix", "accuracies") + "_" + str(peer_id) + ".csv")
    if not os.path.exists(accuracies_file_path):
        with open(accuracies_file_path, "w") as accuracies_file:
            accuracies_file.write("algorithm,dataset,partitioner,alpha,learning_rate,peer,round,time,accuracy,loss\n")

    with open(accuracies_file_path, "a") as accuracies_file:
        accuracies_file.write("%s,%s,%s,%g,%g,%d,%d,%.2f,%f,%f\n" % (
            settings.algorithm, settings.dataset, settings.partitioner,
            settings.alpha, settings.learning.learning_rate, peer_id,
            round_nr, cur_time / MICROSECONDS, accuracy, loss))

    logger.info("[t=%.2f] Model accuracy (peer %d, round %d): %f, loss: %f", cur_time / MICROSECONDS, peer_id, round_nr, accuracy, loss)

    if not params.get("return_model", True):
        return []
    detached_model = copy_model_state(model, settings.dataset, architecture=settings.model)
    return [detached_model]


def evaluate(settings, params):
    """Return a validation report for protocol decisions, without CSV side effects."""
    global dataset
    split = params.get("evaluation_split", "validation")
    if split == "test":
        raise ValueError("Adaptive decisions require a validation split; use test for final reporting")
    if dataset is None:
        dataset = create_dataset(settings)
    if dataset.load_split(split) is dataset.load_split("test"):
        raise ValueError("Dataset must provide a validation split separate from its test split")
    evaluation_settings = copy.copy(settings)
    evaluation_settings.evaluation_split = split
    accuracy, loss = ModelEvaluator(dataset, evaluation_settings).evaluate_accuracy(
        params["model"], device_name=settings.torch_device_name,
        evaluation_examples=params.get("evaluation_examples"))
    return [{"accuracy": accuracy, "loss": loss, "evaluation_split": split,
             "peer": params["peer"], "round": params.get("round", 0),
             "evaluated_at": params["time"]}]


def qgm_train(settings, params):
    from dasklearn.optimizer.qgm import prepare
    gradient_model = compute_gradient(settings, dict(params))[0]
    return list(prepare(gradient_model, params['momentum'], settings.learning.learning_rate,
                        settings.beta, settings.learning.weight_decay, settings.nesterov, settings.mode))


def qgm_finish(settings, params):
    from dasklearn.optimizer.qgm import finish
    return list(finish(params['base'], params['local_model'], params['momentum'],
                       params['local_momentum'], params['packets'], params['weights'],
                       settings.learning.learning_rate, settings.mu, settings.mode))


def choco_prepare(settings, params):
    from dasklearn.optimizer.choco import prepare
    model, packet = prepare(params['model'], params['state'], settings.compressor,
                            settings.keep_ratio, settings.consensus_stepsize, training_seed(settings, params))
    return [model, packet]


def choco_receive(settings, params):
    from dasklearn.optimizer.choco import receive
    return [receive(params['state'], params['packets'], params['weights'], params['size'])]


def chunk(settings: SessionSettings, params: Dict) -> List[torch.Tensor]:
    model = params["model"]
    num_chunks = params["n"]
    chunks = ChunkManager.chunk_model(model, num_chunks, weights=params.get("weights"))
    return chunks


def reconstruct_from_chunks(settings: SessionSettings, params: Dict) -> List[torch.nn.Module]:
    chunks = params["chunks"]
    model = create_model_structure(settings.dataset, architecture=settings.model, settings=settings)
    model = ChunkManager.reconstruct_model(chunks, model)
    return [model]


def aggregate_chunk(settings: SessionSettings, params: Dict):
    return [torch.stack(params["chunks"]).mean(dim=0)]


def weighted_reconstruct_from_chunks(settings: SessionSettings, params: Dict) -> List[torch.nn.Module]:
    chunks = params["chunks"]
    result = []
    for sublist in chunks:
        weighted_sum = sum(tensor * weight for tensor, weight in sublist)
        result.append(weighted_sum)

    model = create_model_structure(settings.dataset, architecture=settings.model, settings=settings)
    model = ChunkManager.weighted_reconstruct_model(result, model)
    return [model]


def demo_prepare(settings, params):
    """Accumulate H microbatch gradients at fixed parameters, then compress.

H is compute matched to the local-SGD block in other baselines. It is gradient
accumulation, NOT H local DeMo steps without communicating. H=1 is canonical.
"""
    from dasklearn.optimizer.demo import prepare
    global dataset, model_trainers
    if dataset is None:
        dataset = create_dataset(settings)
    if model_trainers is None:
        model_trainers = [None] * settings.participants
    peer = params['peer']
    if model_trainers[peer] is None:
        model_trainers[peer] = ModelTrainer(dataset, settings, peer)
    trainer = model_trainers[peer]
    seed = training_seed(settings, params)
    device = settings.torch_device_name
    model = copy.deepcopy(params['model']).to(device).train()
    parameters = list(model.parameters())
    gradients = [torch.zeros_like(p) for p in parameters]
    lossf = torch.nn.NLLLoss() if settings.dataset == 'cifar10' and settings.model not in ['resnet8', 'resnet18', 'mobilenet_v3_large'] else torch.nn.CrossEntropyLoss()
    feature = 'x' if settings.dataset == 'femnist' else 'image' if settings.dataset == 'tiny_imagenet' else 'img'
    label = 'y' if settings.dataset == 'femnist' else 'label'
    cuda_devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        loader = trainer.make_train_loader(seed); batches = iter(loader)
        for _ in range(settings.learning.local_steps):
            try: batch = next(batches)
            except StopIteration:
                batches = iter(loader); batch = next(batches)
            model.zero_grad(set_to_none=True)
            if settings.dataset == "text":
                from dasklearn.datasets.text import text_forward
                text_forward(model, batch, device)[0].backward()
            else:
                lossf(model(batch[feature].to(device)), batch[label].to(device)).backward()
            for gradient, parameter in zip(gradients, parameters):
                gradient.add_(parameter.grad / settings.learning.local_steps)
    packet, residual = prepare(gradients, params['residual'], settings.learning.learning_rate,
        settings.compression_decay, settings.compression_chunk, settings.compression_topk)
    return [packet, residual]


def demo_update(settings, params):
    from dasklearn.optimizer.demo import update
    return [update(params['model'], params['packets'], settings.learning.learning_rate,
                   settings.learning.weight_decay, settings.compression_chunk)]
