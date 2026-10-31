import argparse
from dasklearn.algorithms import ALGORITHMS
from dasklearn.tasks.training import add_training_arguments


def create_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', help='YAML/JSON configuration; use run_experiment.py for sweeps')
    parser.add_argument('--local', action=argparse.BooleanOptionalAction, default=False,
                        help='Execute the simulator DAG locally without brokers')

    # Learning settings
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--local-steps', type=int, default=5)

    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "femnist", "google_speech", "tiny_imagenet", "synthetic", "digits", "text", "vision"], help="Model/preprocessing preset; use --dataset-name to select the data source")
    parser.add_argument('--partitioner', type=str, default="iid", choices=["iid", "natural", "shards", "dirichlet"])
    from dasklearn.datasets.huggingface import add_dataset_arguments
    add_dataset_arguments(parser)
    parser.add_argument('--peers', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=0)
    parser.add_argument('--model', type=str, default="gnlenet")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--work-dir', default="", help="Parent directory for run artifacts; use separate directories for comparisons")
    parser.add_argument('--log-level', type=str, default="INFO")
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda[:index], or mps')
    parser.add_argument('--dry-run', action=argparse.BooleanOptionalAction)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--stragglers-proportion', type=float, default=0.0)
    parser.add_argument('--stragglers-ratio', type=float, default=0.1)
    parser.add_argument('--from-dir', type=str, default=None)  # Load a workflow DAG/settings from a previous run
    parser.add_argument('--feedback-execution', choices=['local', 'distributed'], default='local')
    parser.add_argument('--feedback-replay', default=None, help='Replay observations from a trusted feedback.pkl')
    parser.add_argument('--dag-checkpoint-interval', type=int, default=0)  # Save the workflow DAG every N seconds

    # Algorithm-specific parameters
    parser.add_argument('--synchronous', action=argparse.BooleanOptionalAction)
    parser.add_argument('--algorithm', default='dpsgd', choices=list(ALGORITHMS))
    from dasklearn.simulation.dspodfl.settings import add_dspodfl_arguments
    add_dspodfl_arguments(parser)
    from dasklearn.simulation.demo.settings import add_demo_arguments
    add_demo_arguments(parser)
    from dasklearn.simulation.choco.settings import add_choco_arguments
    add_choco_arguments(parser)
    from dasklearn.simulation.qgm.settings import add_qgm_arguments
    add_qgm_arguments(parser)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--averaging-steps', type=int, default=1)
    parser.add_argument('--teleportation-topology', choices=['ring', 'hypercube', 'complete'], default='ring')
    parser.add_argument('--topology', type=str, default="kreg", choices=["ring", "kreg"])
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--fl-weighting', choices=['uniform', 'examples'], default='uniform',
                        help='Centralized FL client weights: historical uniform or partition example counts')
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
    parser.add_argument('--evaluation-split', choices=['test', 'validation', 'valid'], default='test')
    parser.add_argument('--evaluation-examples', type=int, default=0, help='Evaluation example limit; 0 uses the full split')
    parser.add_argument('--model-size-bytes', type=int, default=None, help='Explicit simulated payload size')
    parser.add_argument('--release-cuda-cache', action=argparse.BooleanOptionalAction, default=False)

    # Plots
    parser.add_argument('--compute-graph-plot-size', type=int, default=100)

    # Profiling
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction)
    parser.add_argument('--log-bandwidth-utilization', action=argparse.BooleanOptionalAction)

    # Broker-related parameters
    parser.add_argument('--brokers', type=int, default=1)
    parser.add_argument('--port', type=int, default=5555)

    from dasklearn.lora import add_lora_arguments
    add_lora_arguments(parser)
    add_training_arguments(parser)
    return parser


def get_args(argv=None):
    import sys
    from dasklearn.experiments import explicit_options, load_single_options
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = create_parser()
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument('--config')
    selected, _ = probe.parse_known_args(argv)
    file_values = {}
    if selected.config:
        try:
            file_values = load_single_options(selected.config, parser)
        except ValueError as exc:
            parser.error(str(exc))
        parser.set_defaults(**file_values)
    args = parser.parse_args(argv)
    cli_options = explicit_options(parser, argv)
    supplied = set(file_values) | cli_options
    if 'local' in cli_options and args.local and 'dry_run' not in cli_options:
        args.dry_run = False
    if 'dry_run' in cli_options and args.dry_run and 'local' not in cli_options:
        args.local = False
    if args.dataset == "femnist":
        if 'learning_rate' not in supplied:
            args.learning_rate = 0.004
        if 'momentum' not in supplied:
            args.momentum = 0
    if args.local and args.dry_run:
        parser.error('--local and --dry-run cannot be combined')
    if args.local and args.from_dir:
        parser.error('--local does not support --from-dir')
    if selected.config:
        from dasklearn.experiments import validate_run
        try:
            validate_run(vars(args))
        except ValueError as exc:
            parser.error(str(exc))

    return args


def get_broker_args():
    parser = argparse.ArgumentParser()

    # Learning settings
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument('--coordinator', type=str)
    parser.add_argument('--advertise-host', default=None,
                        help='Broker address reachable by coordinator and other brokers')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    return args
