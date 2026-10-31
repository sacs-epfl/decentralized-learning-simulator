"""Run one experiment or a parameter sweep from a YAML/JSON file."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from args import create_parser
from dasklearn.experiments import EXECUTIONS, resolve_experiment
from dasklearn.experiment_runner import execute_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("config", help="Experiment .yaml/.yml/.json file")
    parser.add_argument("--output", help="Fresh output directory (relative to the current directory)")
    parser.add_argument("--execution", choices=EXECUTIONS)
    parser.add_argument("--plan", action="store_true", help="Print resolved runs without downloads or execution")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue the sweep after a failed run")
    args, overrides = parser.parse_known_args(argv)
    try:
        experiment = resolve_experiment(args.config, create_parser(), overrides,
                                        output=args.output, execution=args.execution)
        if args.plan:
            print(json.dumps(dict(name=experiment.name, output=str(experiment.output),
                                  runs=[asdict(run) for run in experiment.runs]), indent=2, allow_nan=False))
            return 0
        return execute_experiment(experiment, continue_on_error=args.continue_on_error)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
