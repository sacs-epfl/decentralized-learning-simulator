import asyncio
from asyncio import ensure_future
from multiprocessing import freeze_support
from typing import Dict

import torch

from args import get_args
from dasklearn.session_settings import SessionSettings, LearningSettings
from dasklearn.util import MICROSECONDS


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
        partitioner=args.partitioner,
        alpha=args.alpha,
        model=args.model,
        test_interval=args.test_interval,
        brokers=args.brokers,
        traces=args.traces,
        min_bandwidth=args.min_bandwidth,
        rounds=args.rounds,
        port=args.port,
        log_level=args.log_level,
        torch_threads=args.torch_threads,
        dry_run=args.dry_run,
        duration=args.duration * MICROSECONDS,
        test_period=args.test_period * MICROSECONDS,
        compute_graph_plot_size=args.compute_graph_plot_size,
        stop=args.stop,
        wait=args.wait,
        stragglers_ratio=args.stragglers_ratio,
        stragglers_proportion=args.stragglers_proportion,
    )

    if settings.algorithm == "fl":
        from dasklearn.simulation.fl.simulation import FLSimulation as SIM
        from dasklearn.simulation.fl.settings import FLSettings
        settings = FLSettings(**settings.__dict__, sample_size=args.sample_size)
    elif settings.algorithm == "dpsgd":
        from dasklearn.simulation.dpsgd.simulation import DPSGDSimulation as SIM
    elif settings.algorithm == "subset":
        from dasklearn.simulation.subset.simulation import SubsetDLSimulation as SIM
        from dasklearn.simulation.subset.settings import SubsetLearningSettings
        settings = SubsetLearningSettings(**settings.__dict__, sample_size=args.sample_size)
    elif settings.algorithm == "gossip":
        from dasklearn.simulation.gossip.simulation import GossipSimulation as SIM
        from dasklearn.simulation.gossip.settings import GLSettings
        settings = GLSettings(**settings.__dict__, gl_period=args.gl_period, agg=args.agg)
    elif settings.algorithm == "super-gossip":
        from dasklearn.simulation.super_gossip.simulation import SuperGossipSimulation as SIM
        from dasklearn.simulation.super_gossip.settings import SuperGossipSettings
        settings = SuperGossipSettings(**settings.__dict__, gl_period=args.gl_period, agg=args.agg, k=args.k)
    elif settings.algorithm == "adpsgd":
        from dasklearn.simulation.adpsgd.simulation import ADPSGDSimulation as SIM
        from dasklearn.simulation.adpsgd.settings import ADPSGDSettings
        settings = ADPSGDSettings(**settings.__dict__, agg=args.agg)
    elif settings.algorithm == "epidemic":
        from dasklearn.simulation.epidemic.simulation import EpidemicSimulation as SIM
        from dasklearn.simulation.epidemic.settings import ELSettings
        settings = ELSettings(**settings.__dict__, el=args.el)
    elif settings.algorithm == "lubor":
        from dasklearn.simulation.lubor.simulation import LuborSimulation as SIM
        from dasklearn.simulation.lubor.settings import LuborSettings
        settings = LuborSettings(**settings.__dict__, k=args.k, no_weights=args.no_weights)
    else:
        raise RuntimeError("Unsupported algorithm %s" % settings.algorithm)

    simulation = SIM(settings)
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
