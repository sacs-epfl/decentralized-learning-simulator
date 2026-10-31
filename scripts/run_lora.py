"""Compatibility entry point for local text/vision LoRA runs."""
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from args import get_args
from dasklearn.algorithms import class_paths, import_class
from dasklearn.local_runner import run_local
from dasklearn.settings_factory import session_options
from main import get_torch_device


async def run():
    args = get_args()
    if not args.lora or args.dataset not in {"text", "vision"}:
        raise ValueError("Use --dataset text/vision --lora --base-model MODEL")
    if args.from_dir:
        raise ValueError("Use main.py --from-dir to replay saved workflows")
    if args.dry_run:
        raise ValueError("run_lora.py executes training; use main.py --dry-run or execution: simulate to generate a DAG")
    # Preserve the historical convenience defaults of this entry point.
    args.local, args.rounds = True, args.rounds or 5
    args.sample_size = min(args.sample_size, args.peers)
    device = get_torch_device() if args.device == "auto" else args.device
    settings_type, simulation_type = map(import_class, class_paths(args.algorithm))
    settings = settings_type(**session_options(args, device))
    await run_local(simulation_type(settings=settings))


if __name__ == "__main__":
    asyncio.run(run())
