"""Run a reproducible, live algorithm leaderboard with configurable time budgets."""
import argparse
import asyncio
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("config", nargs="?", help="Benchmark YAML/JSON configuration")
    parser.add_argument("--output", help="Fresh output directory")
    parser.add_argument("--budgets", type=int, nargs="+", help="Simulated seconds, e.g. 60 300 1800")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--variants", nargs="+", help="Subset of named runs")
    parser.add_argument("--evaluation-period", type=int)
    parser.add_argument("--plan", action="store_true", help="Resolve without training or downloads")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from dasklearn.benchmark import execute_benchmark, load_benchmark, run_worker
    if args.worker:
        asyncio.run(run_worker(args.worker))
        return 0
    if not args.config:
        parser.error("config is required")
    from args import create_parser
    try:
        benchmark = load_benchmark(args.config, create_parser(), output=args.output,
            budgets=args.budgets, seeds=args.seeds, variants=args.variants, period=args.evaluation_period)
        if args.plan:
            print(json.dumps(benchmark.manifest() | dict(output=str(benchmark.output)), indent=2, allow_nan=False))
            return 0
        if benchmark.tuning is not None:
            from dasklearn.lr_tuning import execute_tuned_benchmark
            return execute_tuned_benchmark(benchmark)
        return execute_benchmark(benchmark)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
