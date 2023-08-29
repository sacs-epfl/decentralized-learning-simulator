import argparse
import math
import time
from multiprocessing import freeze_support

from dask.distributed import Client, LocalCluster

from dasklearn.session_settings import SessionSettings, LearningSettings

import networkx as nx


graph = {}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--peers', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--model', type=str, default="gnlenet")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--work-fraction', type=float, default=1)
    parser.add_argument('--algorithm', type=str, default="fedavg")

    # Accuracy checking
    parser.add_argument('--test-interval', type=int, default=10)

    # Dask-related parameters
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--scheduler', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    freeze_support()

    args = get_args()

    learning_settings = LearningSettings(
        learning_rate=0.002,
        momentum=0.9,
        weight_decay=0,
        batch_size=32,
        local_steps=20,
    )

    settings = SessionSettings(
        dataset="cifar10",
        work_dir="",
        learning=learning_settings,
        participants=args.peers,
        partitioner="iid",
        model=args.model,
        test_interval=args.test_interval,
    )

    if args.scheduler:
        client = Client(args.scheduler)
    else:
        # Start a local Dask cluster and connect to it
        cluster = LocalCluster(n_workers=args.workers)
        client = Client(cluster)
        print("Client dashboard URL: %s" % client.dashboard_link)

    if args.algorithm == "fedavg":
        from dasklearn.algorithms.fedavg import generate_task_graph
        tasks = generate_task_graph(args, settings)
    elif args.algorithm == "dpsgd":
        from dasklearn.algorithms.dpsgd import generate_task_graph
        k = math.floor(math.log2(args.peers))
        G = nx.random_regular_graph(k, args.peers)
        tasks = generate_task_graph(G, args, settings)
    else:
        raise RuntimeError("Unknown learning algorithm %s" % args.algorithm)

    # Submit the tasks
    print("Starting training...")
    start_time = time.time()
    result = client.get(tasks, "final")
    elapsed_time = time.time() - start_time

    print("Final result: %s (took %d s.)" % (result, elapsed_time))
