import asyncio
from asyncio import ensure_future
from multiprocessing import freeze_support

import torch

from args import get_args
from dasklearn.session_settings import SessionSettings, LearningSettings


def run():
    args = get_args()

    learning_settings = LearningSettings(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        local_steps=args.local_steps,
    )

    # TODO add availability traces
    settings = SessionSettings(
        synchronous=args.synchronous,
        algorithm=args.algorithm,
        seed=args.seed,
        dataset=args.dataset,
        dataset_base_path=args.dataset_base_path,
        validation_set_fraction=args.validation_set_fraction,
        compute_validation_loss_global_model=args.compute_validation_loss_global_model,
        torch_device_name="cpu" if not torch.cuda.is_available() else "cuda:0",
        work_dir="",
        learning=learning_settings,
        participants=args.peers,
        partitioner="iid",
        model=args.model,
        test_interval=args.test_interval,
        brokers=args.brokers,
        capability_traces=args.capability_traces,
        rounds=args.rounds,
        port=args.port,
        log_level=args.log_level,
        torch_threads=args.torch_threads,
        dry_run=args.dry_run,
    )

    if settings.algorithm == "fl":
        from dasklearn.simulation.fl.simulation import FLSimulation
        simulation = FLSimulation(settings)
    elif settings.algorithm == "dpsgd":
        from dasklearn.simulation.dpsgd.simulation import DPSGDSimulation
        simulation = DPSGDSimulation(settings)
    elif settings.algorithm == "subset":
        from dasklearn.simulation.subset.simulation import SubsetDLSimulation
        simulation = SubsetDLSimulation(settings, args.sample_size)
    else:
        raise RuntimeError("Unsupported algorithm %s" % settings.algorithm)
    ensure_future(simulation.run())


if __name__ == "__main__":
    freeze_support()

    asyncio.get_event_loop().call_later(0, run)
    asyncio.get_event_loop().run_forever()
