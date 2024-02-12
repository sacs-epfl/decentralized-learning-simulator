import argparse


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
    parser.add_argument('--peers', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--model', type=str, default="gnlenet")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algorithm', type=str, default="dpsgd")
    parser.add_argument('--log-level', type=str, default="INFO")
    parser.add_argument('--torch-threads', type=int, default=4)

    # Traces
    parser.add_argument('--capability-traces', type=str, default="data/client_device_capacity")

    # Accuracy checking
    parser.add_argument('--test-interval', type=int, default=5)

    # Dask-related parameters
    parser.add_argument('--brokers', type=int, default=1)
    parser.add_argument('--port', type=int, default=5555)

    args = parser.parse_args()
    if args.dataset == "femnist":
        args.learning_rate = 0.004
        args.momentum = 0

    return args


def get_broker_args():
    parser = argparse.ArgumentParser()

    # Learning settings
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument('--coordinator', type=str)
    parser.add_argument('--workers', type=int, default=1)

    args = parser.parse_args()

    return args
