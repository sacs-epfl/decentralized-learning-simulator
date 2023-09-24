import argparse
import logging
from multiprocessing import freeze_support

from dasklearn.session_settings import SessionSettings, LearningSettings
from dasklearn.simulation.simulation import Simulation

graph = {}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--peers', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--model', type=str, default="gnlenet")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--work-fraction', type=float, default=1)
    parser.add_argument('--algorithm', type=str, default="fedavg")

    # Traces
    parser.add_argument('--capability-traces', type=str, default="data/client_device_capacity")

    # Accuracy checking
    parser.add_argument('--test-interval', type=int, default=10)

    # Dask-related parameters
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--scheduler', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    freeze_support()

    logging.basicConfig(level=logging.INFO)

    args = get_args()

    learning_settings = LearningSettings(
        learning_rate=0.002,
        momentum=0.9,
        weight_decay=0,
        batch_size=32,
        local_steps=20,
    )

    # TODO add availability traces
    settings = SessionSettings(
        seed=args.seed,
        dataset="cifar10",
        work_dir="",
        learning=learning_settings,
        participants=args.peers,
        partitioner="iid",
        model=args.model,
        test_interval=args.test_interval,
        scheduler=args.scheduler,
        workers=args.workers,
        capability_traces=args.capability_traces,
        rounds=args.rounds,
    )

    simulation = Simulation(settings)
    simulation.run()
