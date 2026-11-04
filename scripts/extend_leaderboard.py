"""Tune new algorithms and append their runs to existing paired leaderboards."""
import argparse
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from args import create_parser
    from dasklearn.benchmark import load_benchmark
    from dasklearn.leaderboard_extension import execute_extension, load_baselines, reusable_trials
    from dasklearn.lr_tuning import tuning_benchmark
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--baseline-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-tuning", type=Path, help="Previous tuning directory; reuse matching completed trials")
    parser.add_argument("--follow-baselines", action="store_true", help="Keep merging live baseline results without interrupting their workers")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    try:
        benchmark = load_benchmark(args.config, create_parser(), output=args.output)
        settings = load_baselines(benchmark, args.baseline_catalog, allow_pending=args.follow_baselines)
        if args.plan:
            print(json.dumps(dict(benchmark=benchmark.manifest(), output=str(benchmark.output),
                reused_trials=list(reusable_trials(tuning_benchmark(benchmark), args.reuse_tuning)) if args.reuse_tuning else [],
                settings=[dict(id=entry["id"], baseline=str(entry["output"]),
                               scenario=entry["scenario"], network=entry["suite"]["network"]) for entry in settings]), indent=2))
            return 0
        return execute_extension(benchmark, settings, baseline_catalog=args.baseline_catalog,
                                 reuse_tuning=args.reuse_tuning, follow_baselines=args.follow_baselines)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
