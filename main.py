import argparse
import time
from multiprocessing import freeze_support

from dask.distributed import Client, LocalCluster

from dasklearn.model_manager import ModelManager
from dasklearn.models import create_model, unserialize_model, serialize_model
from dasklearn.session_settings import SessionSettings, LearningSettings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--peers', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=30)
    return parser.parse_args()


def get_task_name(round_nr, peer_id):
    return "t%d_%d" % (round_nr, peer_id)


if __name__ == "__main__":
    freeze_support()

    args = get_args()

    learning_settings = LearningSettings(
        learning_rate=0.002,
        momentum=0.9,
        weight_decay=0,
        batch_size=20,
        local_steps=5,
    )

    settings = SessionSettings(
        dataset="cifar10",
        work_dir="",
        learning=learning_settings,
        participants=args.peers,
        partitioner="iid",
    )

    # Start a local Dask cluster and connect to it
    cluster = LocalCluster(n_workers=args.workers)
    client = Client()
    print("Client URL dashboard: %s" % client.dashboard_link)

    def aggregate(params):
        models, round_nr = params
        print("Aggregating %d models in round %d..." % (len(models), round_nr))

        model_manager = ModelManager(None, settings, 0)
        for peer_id, model in models.items():
            model_manager.process_incoming_trained_model(peer_id, model)

        return model_manager.aggregate_trained_models()

    def train(params):
        model, round_nr, peer_id = params

        # Make a copy of the model so multiple workers are not training the same model
        copied_model = unserialize_model(serialize_model(model), settings.dataset, architecture=settings.model)
        model_manager = ModelManager(copied_model, settings, peer_id)
        model_manager.train()

        print("Training in round %d..." % round_nr)
        return model

    # Create the initial models
    initial_model = create_model("cifar10")
    tasks = {"a0": initial_model}

    for r in range(1, args.rounds + 1):
        # TODO for now, assume we do an all-reduce

        for peer_id in range(settings.participants):
            # Train on the previous aggregated model
            tasks[get_task_name(r, peer_id)] = (train, ['a%d' % (r - 1), r, peer_id])

        # Aggregate
        prev_models = {}
        for peer_id in range(settings.participants):
            prev_models[peer_id] = "t%d_%d" % (r, peer_id)
        tasks["a%d" % r] = (aggregate, [prev_models, r])

    # Submit the tasks
    print("Starting training...")
    start_time = time.time()
    result = client.get(tasks, 'a%d' % args.num_rounds)
    elapsed_time = time.time() - start_time

    print("Final result: %s (took %d s.)" % (result, elapsed_time))
