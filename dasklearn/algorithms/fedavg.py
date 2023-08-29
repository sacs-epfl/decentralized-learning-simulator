from random import Random
from typing import Dict

from dasklearn.functions import *
from dasklearn.models import create_model


def generate_task_graph(args, settings) -> Dict:
    # Create the initial models
    initial_model = create_model("cifar10", architecture=settings.model)
    tasks = {"a0": initial_model}
    all_peers = list(range(settings.participants))
    sample_size = int(settings.participants * args.work_fraction)
    print("Sample size: %d" % sample_size)
    rand = Random(args.seed)

    for r in range(1, args.rounds + 1):
        # Select peers
        selected_peers = rand.sample(all_peers, sample_size)

        # Train
        if r > 1 and (r - 1) % settings.test_interval == 0:
            prev_task = "test%d" % (r - 1)
        else:
            prev_task = 'a%d' % (r - 1)

        for peer_id in selected_peers:
            # Train on the previous aggregated model
            train_task_name = "t%d_%d" % (r, peer_id)
            tasks[train_task_name] = (train, [prev_task, r, peer_id, settings])

        # Aggregate
        prev_models = {}
        for selected_peer in selected_peers:
            prev_models[selected_peer] = "t%d_%d" % (r, selected_peer)

        tasks["a%d" % r] = (aggregate, [prev_models, r, None, settings])
        if r % settings.test_interval == 0:
            tasks["test%d" % r] = (test, ["a%d" % r, r, 0, settings])

    tasks["final"] = (test, ["a%d" % args.rounds, args.rounds, 0, settings])

    return tasks
