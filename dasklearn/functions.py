import os

from dasklearn.model_evaluator import ModelEvaluator
from dasklearn.model_manager import ModelManager
from dasklearn.models import unserialize_model, serialize_model


def aggregate(params):
    models, round_nr, peer_id, settings = params
    print("Peer %d aggregating %d models in round %d..." % (peer_id, len(models), round_nr))

    model_manager = ModelManager(None, settings, 0)
    for peer_id, model in models.items():
        model_manager.process_incoming_trained_model(peer_id, model)

    return model_manager.aggregate_trained_models()


def train(params):
    model, round_nr, peer_id, settings = params

    # Make a copy of the model so multiple workers are not training the same model
    copied_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)
    model_manager = ModelManager(copied_model, settings, peer_id)
    model_manager.train()

    print("Peer %d training in round %d..." % (peer_id, round_nr))
    return model


def test(params):
    model, round_nr, peer_id, settings = params
    print("Peer %d training in round %d..." % (peer_id, round_nr))
    data_dir = os.path.join(os.environ["HOME"], "dfl-data")
    evaluator = ModelEvaluator(data_dir, settings)
    accuracy, loss = evaluator.evaluate_accuracy(model)
    print("Model accuracy: %f, loss: %f" % (accuracy, loss))
    return model
