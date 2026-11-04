"""Tune and train a pooled-data reference using optimizer steps, without simulation."""
import argparse
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from dasklearn.centralized_baseline import execute, load_config, train_worker
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, nargs='?')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'] * 4)
    parser.add_argument('--reuse-tuning', type=Path, help='Previous reference output; verify and reuse its completed LR search')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            train_worker(json.loads(args.worker.read_text()))
            return 0
        if args.config is None:
            parser.error('config is required')
        config = load_config(args.config)
        if args.plan:
            print(json.dumps(config, indent=2))
            return 0
        import re
        if not args.devices or any(not re.fullmatch(r'cpu|mps|cuda(?::[0-9]+)?', device) for device in args.devices):
            raise ValueError('Select concrete CPU/CUDA/MPS worker devices')
        return execute(config, devices=args.devices, reuse_tuning=args.reuse_tuning)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
