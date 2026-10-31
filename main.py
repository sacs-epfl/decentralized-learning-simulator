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

from dasklearn.algorithms import ALGORITHMS, class_paths

simulation_task = None

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


async def run_simulation(simulation, local=False):
    try:
        if local:
            from dasklearn.local_runner import run_local
            await run_local(simulation)
            asyncio.get_running_loop().stop()
        else:
            await simulation.run()
    except Exception:
        asyncio.get_running_loop().stop()
        raise

def run(args=None):
    global simulation_task
    args = args or get_args()

    if args.from_dir:
        settings_dict = load_settings_from_dir(args.from_dir)
        settings_dict["from_dir"] = args.from_dir
        settings_dict["dry_run"] = False  # No dry runs
        if args.device != "auto":
            settings_dict["torch_device_name"] = args.device
    else:
        from dasklearn.settings_factory import session_options
        settings_dict = session_options(args, get_torch_device() if args.device == "auto" else args.device)

    algorithm = settings_dict['algorithm']
    if algorithm not in ALGORITHMS:
        raise RuntimeError(f"Unsupported algorithm {algorithm}")

    settings_class_path, simulation_class_path = class_paths(algorithm)

    # Instantiate settings and simulation class
    settings = instantiate_class(settings_class_path, **settings_dict)
    simulation = instantiate_class(simulation_class_path, settings=settings)

    simulation_task = ensure_future(run_simulation(simulation, local=args.local))

if __name__ == "__main__":
    freeze_support()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        run()
        loop.run_forever()
        if simulation_task is not None and simulation_task.done():
            simulation_task.result()
    except KeyboardInterrupt:
        print("Received exit signal, shutting down...")
    finally:
        loop.stop()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
