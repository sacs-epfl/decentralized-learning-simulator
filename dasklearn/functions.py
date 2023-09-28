import os

import torch

from dasklearn.model_evaluator import ModelEvaluator
from dasklearn.model_manager import ModelManager
from dasklearn.models import unserialize_model, serialize_model, create_model


def train(params):
    model, round_nr, peer_id, settings = params

    if not model:
        torch.manual_seed(settings.seed)
        copied_model = create_model(settings.dataset, architecture=settings.model)
    else:
        # Make a copy of the model so multiple workers are not training the same model
        copied_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)

    model_manager = ModelManager(copied_model, settings, peer_id)
    model_manager.train()

    print("Peer %d training in round %d..." % (peer_id, round_nr))
    return copied_model


def aggregate(params):
    models, round_nr, peer_id, settings = params
    if peer_id is not None:
        print("Peer %d aggregating %d models in round %d..." % (peer_id, len(models), round_nr))
    else:
        print("Aggregating %d models in round %d..." % (len(models), round_nr))

    model_manager = ModelManager(None, settings, 0)
    for peer_id, model in models.items():
        model_manager.process_incoming_trained_model(peer_id, model)

    return model_manager.aggregate_trained_models()


def test(params):
    model, round_nr, cur_time, peer_id, settings = params
    print("Testing model in round %d..." % round_nr)
    data_dir = os.path.join(os.environ["HOME"], "dfl-data")
    evaluator = ModelEvaluator(data_dir, settings)
    accuracy, loss = evaluator.evaluate_accuracy(model)
    with open(os.path.join(settings.data_dir, "accuracies.csv"), "a") as accuracies_file:
        accuracies_file.write("%d,%d,%f,%f,%f\n" % (peer_id, round_nr, cur_time, accuracy, loss))
    print("Model accuracy: %f, loss: %f" % (accuracy, loss))
    return model
