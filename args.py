import argparse
from dasklearn.algorithms import ALGORITHMS
from dasklearn.tasks.training import add_training_arguments


def get_args():
    parser = argparse.ArgumentParser()

    # Learning settings
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--local-steps', type=int, default=5)

    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "femnist", "movielens", "google_speech", "tiny_imagenet", "synthetic", "digits"])
    parser.add_argument('--partitioner', type=str, default="iid", choices=["iid", "natural", "shards", "dirichlet"])
    parser.add_argument('--dataset-base-path', type=str, default=None)
    parser.add_argument('--peers', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=0)
    parser.add_argument('--model', type=str, default="gnlenet")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-level', type=str, default="INFO")
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--dry-run', action=argparse.BooleanOptionalAction)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--stragglers-proportion', type=float, default=0.0)
    parser.add_argument('--stragglers-ratio', type=float, default=0.1)
    parser.add_argument('--from-dir', type=str, default=None)  # Load a workflow DAG/settings from a previous run
    parser.add_argument('--dag-checkpoint-interval', type=int, default=0)  # Save the workflow DAG every N seconds

    from dasklearn.simulation.conflux_v2.settings import add_conflux_v2_arguments
    add_conflux_v2_arguments(parser)
    parser.add_argument('--relay-count', type=int, default=0)

    # Algorithm-specific parameters
    parser.add_argument('--synchronous', action=argparse.BooleanOptionalAction)
    parser.add_argument('--algorithm', default='dpsgd', choices=list(ALGORITHMS))
    from dasklearn.simulation.dspodfl.settings import add_dspodfl_arguments
    add_dspodfl_arguments(parser)
    from dasklearn.simulation.demo.settings import add_demo_arguments
    add_demo_arguments(parser)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--averaging-steps', type=int, default=1)
    parser.add_argument('--teleportation-topology', choices=['ring', 'hypercube', 'complete'], default='ring')
    parser.add_argument('--topology', type=str, default="kreg", choices=["ring", "kreg"])
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--chunks-in-sample', type=int, default=10)
    parser.add_argument('--success-fraction', type=float, default=1)
    parser.add_argument('--control-message-bytes', type=int, default=64)
    parser.add_argument('--network-latency-ms', type=float, default=0)
    parser.add_argument('--duration', type=int, default=100)
    parser.add_argument('--gl-period', type=int, default=10)
    parser.add_argument('--agg', type=str, default="default", choices=["default", "average", "age"])
    parser.add_argument('--stop', type=str, default="rounds", choices=["rounds", "duration"])
    parser.add_argument('--wait', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--el', type=str, default="oracle", choices=["oracle", "local"])
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--r', type=int, default=0)
    parser.add_argument('--no_weights', action=argparse.BooleanOptionalAction, default=False)

    # Traces
    parser.add_argument('--min-bandwidth', type=int, default=0)  # The minimum bandwidth a node must have to participate, in bytes/s.
    parser.add_argument('--traces', type=str, default="none", choices=["none", "fedscale", "diablo"])

    # Churn
    parser.add_argument('--churn', type=str, default="none", choices=["none", "synthetic", "fedscale"])

    # Accuracy checking
    parser.add_argument('--validation-set-fraction', type=float, default=0.0)
    parser.add_argument('--compute-validation-loss-global-model', action=argparse.BooleanOptionalAction)
    parser.add_argument('--test-interval', type=int, default=5)
    parser.add_argument('--test-period', type=int, default=60)
    parser.add_argument('--test-method', type=str, default="individual", choices=["individual", "global", "snapshot"])

    # Plots
    parser.add_argument('--compute-graph-plot-size', type=int, default=100)

    # Profiling
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction)
    parser.add_argument('--log-bandwidth-utilization', action=argparse.BooleanOptionalAction)

    # Broker-related parameters
    parser.add_argument('--brokers', type=int, default=1)
    parser.add_argument('--port', type=int, default=5555)

    add_training_arguments(parser)
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
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    return args
