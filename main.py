import argparse
import logging
from multiprocessing import freeze_support

from dasklearn.session_settings import SessionSettings, LearningSettings
from dasklearn.simulation.simulation import Simulation


def get_args():
    parser = argparse.ArgumentParser()

    # Learning settings
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--local-steps', type=int, default=5)

    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "femnist"])
    parser.add_argument('--dataset-base-path', type=str, default=None)
    parser.add_argument('--peers', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--model', type=str, default="gnlenet")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algorithm', type=str, default="dpsgd")

    # Traces
    parser.add_argument('--capability-traces', type=str, default="data/client_device_capacity")

    # Accuracy checking
    parser.add_argument('--test-interval', type=int, default=5)

    # Dask-related parameters
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--scheduler', type=str, default=None)

    args = parser.parse_args()
    if args.dataset == "femnist":
        args.learning_rate = 0.004
        args.momentum = 0

    return args


if __name__ == "__main__":
    freeze_support()

    logging.basicConfig(level=logging.INFO)

    args = get_args()

    learning_settings = LearningSettings(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        local_steps=args.local_steps,
    )

    # TODO add availability traces
    settings = SessionSettings(
        algorithm=args.algorithm,
        seed=args.seed,
        dataset=args.dataset,
        dataset_base_path=args.dataset_base_path,
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
