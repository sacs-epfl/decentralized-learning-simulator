from typing import Dict

from dasklearn.functions import *
from dasklearn.models import create_model


def generate_task_graph(G, args, settings) -> Dict:
    graph = {node: list(G[node]) for node in G}

    # Create the initial models
    initial_model = create_model("cifar10", architecture=settings.model)
    tasks = {"a0_0": initial_model}

    for r in range(1, args.rounds + 1):
        # Train
        for peer_id in range(settings.participants):
            # Train on the previous aggregated model
            agg_task = 'a%d_%d' % (r - 1, peer_id) if r > 1 else "a0_0"
            train_task_name = "t%d_%d" % (r, peer_id)
            tasks[train_task_name] = (train, [agg_task, r, peer_id, settings])

        # Aggregate
        for peer_id in range(settings.participants):
            prev_models = {}
            for neighbour_peer_id in graph[peer_id]:
                prev_models[neighbour_peer_id] = "t%d_%d" % (r, neighbour_peer_id)

            tasks["a%d_%d" % (r, peer_id)] = (aggregate, [prev_models, r, peer_id, settings])

    # Add one final aggregation step
    prev_models = {}
    for peer_id in range(settings.participants):
        prev_models[peer_id] = "a%d_%d" % (args.rounds, peer_id)
    tasks["final_aggregate"] = (aggregate, [prev_models, args.rounds + 1, 0, settings])
    tasks["final"] = (test, ["final_aggregate", args.rounds + 1, 0, settings])

    return tasks
