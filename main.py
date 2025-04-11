import asyncio
from asyncio import ensure_future
import json
import os
from multiprocessing import freeze_support
from typing import Dict, Type

import torch

from args import get_args
from dasklearn.session_settings import SessionSettings, LearningSettings
from dasklearn.util import MICROSECONDS

# Mapping of algorithms to their respective settings and simulation classes
ALGORITHM_MAPPING = {
    "fl": ("dasklearn.simulation.fl.settings.FLSettings", "dasklearn.simulation.fl.simulation.FLSimulation"),
    "dpsgd": ("dasklearn.simulation.dpsgd.settings.DPSGDSettings", "dasklearn.simulation.dpsgd.simulation.DPSGDSimulation"),
    "epidemic": ("dasklearn.simulation.dpsgd.settings.DPSGDSettings", "dasklearn.simulation.dpsgd.simulation.DPSGDSimulation"),
    "gossip": ("dasklearn.simulation.gossip.settings.GLSettings", "dasklearn.simulation.gossip.simulation.GossipSimulation"),
    "super-gossip": ("dasklearn.simulation.super_gossip.settings.SuperGossipSettings", "dasklearn.simulation.super_gossip.simulation.SuperGossipSimulation"),
    "adpsgd": ("dasklearn.simulation.adpsgd.settings.ADPSGDSettings", "dasklearn.simulation.adpsgd.simulation.ADPSGDSimulation"),
    "lubor": ("dasklearn.simulation.lubor.settings.LuborSettings", "dasklearn.simulation.lubor.simulation.LuborSimulation"),
    "conflux": ("dasklearn.simulation.conflux.settings.ConfluxSettings", "dasklearn.simulation.conflux.simulation.ConfluxSimulation"),
    "teleportation": ("dasklearn.simulation.teleportation.settings.TeleportationSettings", "dasklearn.simulation.teleportation.simulation.TeleportationSimulation"),
    "shatter": ("dasklearn.simulation.shatter.settings.ShatterSettings", "dasklearn.simulation.shatter.simulation.ShatterSimulation"),
    "pushsum": ("dasklearn.simulation.pushsum.settings.PushSumSettings", "dasklearn.simulation.pushsum.simulation.PushSumSimulation"),
}

def get_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_settings_from_dir(directory: str) -> Dict:
    """Loads settings from a directory and ensures LearningSettings is correctly instantiated."""
    settings_path = os.path.join(directory, "settings.json")
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Settings file {settings_path} does not exist.")
    
    with open(settings_path, "r") as settings_file:
        settings_dict = json.load(settings_file)

    # Ensure LearningSettings is correctly instantiated
    if "learning" in settings_dict:
        settings_dict["learning"] = LearningSettings(**settings_dict["learning"])

    # The device might be different now
    settings_dict["torch_device_name"] = get_torch_device()

    return settings_dict

def instantiate_class(class_path: str, **kwargs):
    """Dynamically imports and instantiates a class from a given module path."""
    module_name, class_name = class_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**kwargs)

def run():
    args = get_args()

    if args.from_dir:
        settings_dict = load_settings_from_dir(args.from_dir)
        settings_dict["from_dir"] = args.from_dir
        settings_dict["dry_run"] = False  # No dry runs
    else:
        learning_settings = LearningSettings(
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            local_steps=args.local_steps,
        )

        settings_dict = {
            "synchronous": args.synchronous,
            "algorithm": args.algorithm,
            "seed": args.seed,
            "dataset": args.dataset,
            "dataset_base_path": args.dataset_base_path,
            "validation_set_fraction": args.validation_set_fraction,
            "compute_validation_loss_global_model": args.compute_validation_loss_global_model,
            "torch_device_name": get_torch_device(),
            "work_dir": "",
            "learning": learning_settings,
            "participants": args.peers,
            "partitioner": args.partitioner,
            "alpha": args.alpha,
            "model": args.model,
            "test_interval": args.test_interval,
            "brokers": args.brokers,
            "traces": args.traces,
            "churn": args.churn,
            "min_bandwidth": args.min_bandwidth,
            "rounds": args.rounds,
            "port": args.port,
            "log_level": args.log_level,
            "torch_threads": args.torch_threads,
            "dry_run": args.dry_run,
            "duration": args.duration * MICROSECONDS,
            "test_period": args.test_period * MICROSECONDS,
            "test_method": args.test_method,
            "compute_graph_plot_size": args.compute_graph_plot_size,
            "stop": args.stop,
            "wait": args.wait,
            "stragglers_ratio": args.stragglers_ratio,
            "stragglers_proportion": args.stragglers_proportion,
            "profile": args.profile,
            "log_bandwidth_utilization": args.log_bandwidth_utilization,
            "dag_checkpoint_interval": args.dag_checkpoint_interval,
        }

    if args.algorithm not in ALGORITHM_MAPPING:
        raise RuntimeError(f"Unsupported algorithm {args.algorithm}")

    settings_class_path, simulation_class_path = ALGORITHM_MAPPING[args.algorithm]

    # Add algorithm-specific parameters
    algorithm_params = {
        "fl": {"sample_size": args.sample_size},
        "dpsgd": {"el": args.el, "topology": args.topology, "k": args.k},
        "epidemic": {"el": args.el, "topology": args.topology, "k": args.k},
        "gossip": {"gl_period": args.gl_period, "agg": args.agg},
        "super-gossip": {"gl_period": args.gl_period, "agg": args.agg, "k": args.k},
        "adpsgd": {"agg": args.agg},
        "lubor": {"k": args.k, "no_weights": args.no_weights},
        "conflux": {"sample_size": args.sample_size, "chunks_in_sample": args.chunks_in_sample, "success_fraction": args.success_fraction},
        "teleportation": {"sample_size": args.sample_size},
        "shatter": {"k": args.k, "r": args.r},
        "pushsum": {"sample_size": args.sample_size},
    }

    # Instantiate settings and simulation class
    settings_dict.update(algorithm_params.get(args.algorithm, {}))
    settings = instantiate_class(settings_class_path, **settings_dict)
    simulation = instantiate_class(simulation_class_path, settings=settings)

    ensure_future(simulation.run())

if __name__ == "__main__":
    freeze_support()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.call_later(0, run)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("Received exit signal, shutting down...")
    finally:
        loop.stop()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
