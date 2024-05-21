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
    parser.add_argument('--partitioner', type=str, default="iid", choices=["iid", "shards", "dirichlet"])
    parser.add_argument('--dataset-base-path', type=str, default=None)
    parser.add_argument('--peers', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--model', type=str, default="gnlenet")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-level', type=str, default="INFO")
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--dry-run', action=argparse.BooleanOptionalAction)
    parser.add_argument('--alpha', type=float, default=0.1)

    # Algorithm-specific parameters
    parser.add_argument('--synchronous', action=argparse.BooleanOptionalAction)
    parser.add_argument('--algorithm', type=str, default="dpsgd")
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--duration', type=int, default=100)
    parser.add_argument('--gl-period', type=int, default=10)
    parser.add_argument('--agg', type=str, default="default", choices=["default", "average", "age"])
    parser.add_argument('--stop', type=str, default="rounds", choices=["rounds", "duration"])
    parser.add_argument('--wait', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--el', type=str, default="oracle", choices=["oracle", "local"])
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--no_weights', action=argparse.BooleanOptionalAction, default=False)

    # Traces
    parser.add_argument('--capability-traces', type=str, default="data/client_device_capacity")

    # Accuracy checking
    parser.add_argument('--validation-set-fraction', type=float, default=0.0)
    parser.add_argument('--compute-validation-loss-global-model', action=argparse.BooleanOptionalAction)
    parser.add_argument('--test-interval', type=int, default=5)
    parser.add_argument('--test-period', type=int, default=60)

    # Plots
    parser.add_argument('--compute-graph-plot-size', type=int, default=100)

    # Broker-related parameters
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
