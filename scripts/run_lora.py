"""Run a small LoRA simulation locally, with no broker setup."""
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from args import get_args
from dasklearn.algorithms import class_paths
from dasklearn.lora import lora_options
from dasklearn.datasets.huggingface import dataset_options
from dasklearn.session_settings import LearningSettings
from dasklearn.tasks.local import execute_local
from dasklearn.util import MICROSECONDS
from main import instantiate_class, get_torch_device


async def run():
    args = get_args()
    if not args.lora or args.dataset not in {"text", "vision"}:
        raise ValueError("Use --dataset text/vision --lora --base-model MODEL")
    if args.from_dir:
        raise ValueError("Use main.py --from-dir to replay saved workflows")
    settings_path, simulation_path = class_paths(args.algorithm)
    from dasklearn.simulation.dspodfl.settings import dspodfl_options
    from dasklearn.simulation.choco.settings import choco_options
    from dasklearn.simulation.qgm.settings import qgm_options
    from dasklearn.simulation.demo.settings import demo_options
    specific = {
        "fl": {"sample_size": min(args.sample_size, args.peers), "aggregation_weighting": args.fl_weighting},
        "dpsgd": {"topology": args.topology, "k": args.k, "el": args.el},
        "epidemic": {"topology": args.topology, "k": args.k, "el": args.el},
        "gossip": {"gl_period": args.gl_period * MICROSECONDS, "agg": args.agg},
        "super-gossip": {"gl_period": args.gl_period * MICROSECONDS, "agg": args.agg, "k": args.k},
        "adpsgd": {"agg": args.agg},
        "lubor": {"k": args.k, "no_weights": args.no_weights},
        "shatter": {"k": args.k, "r": args.r},
        "conflux": {"sample_size": min(args.sample_size, args.peers), "chunks_in_sample": args.chunks_in_sample,
                    "success_fraction": args.success_fraction},
        "moshpit": {"group_size": args.group_size, "averaging_steps": args.averaging_steps},
        "teleportation": {"sample_size": min(args.sample_size, args.peers), "topology": args.teleportation_topology},
        "choco": choco_options(args), "qgm": qgm_options(args),
        "demo": demo_options(args), "dspodfl": dspodfl_options(args),
    }.get(args.algorithm, {})
    settings = instantiate_class(settings_path, algorithm=args.algorithm, seed=args.seed,
        work_dir=args.work_dir, dataset=args.dataset, model=args.model, participants=args.peers,
        learning=LearningSettings(args.learning_rate, args.momentum, args.weight_decay, args.batch_size, args.local_steps),
        rounds=args.rounds or 5, duration=args.duration * MICROSECONDS, stop=args.stop,
        synchronous=args.synchronous, partitioner=args.partitioner, alpha=args.alpha,
        torch_device_name=get_torch_device() if args.device == "auto" else args.device, torch_threads=args.torch_threads,
        validation_set_fraction=args.validation_set_fraction,
        compute_validation_loss_global_model=args.compute_validation_loss_global_model,
        test_method=args.test_method, test_interval=args.test_interval, test_period=args.test_period * MICROSECONDS,
        dry_run=True, **lora_options(args), **dataset_options(args), **specific)
    simulation = instantiate_class(simulation_path, settings=settings)
    simulation.setup_directories()
    simulation.simulate()
    simulation.workflow_dag.build_task_indices()
    simulation.workflow_dag.check_validity()
    execute_local(simulation, retain_results=False)
    simulation.merge_accuracies_files()
    print(json.dumps({"output": simulation.data_dir, "adapter_payload_bytes": simulation.model_size}, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
