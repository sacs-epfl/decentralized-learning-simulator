"""Run protocol event simulations and execute their training DAGs.

No broker, downloads, traces or GPUs are needed. Simulated seconds and actual
execution seconds are reported separately. This is a mechanism check on a small
synthetic classifier, not evidence of ImageNet performance.
"""
import argparse
import hashlib
import asyncio
import csv
import json
import logging
import platform
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.conflux.settings import ConfluxSettings
from dasklearn.simulation.conflux.simulation import ConfluxSimulation
from dasklearn.simulation.shatter.settings import ShatterSettings
from dasklearn.simulation.shatter.simulation import ShatterSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.training import add_training_arguments
from dasklearn.simulation.peer_baselines import BASELINE_VARIANTS, baseline_spec
from dasklearn.simulation.dspodfl.settings import add_dspodfl_arguments, dspodfl_options
from dasklearn.simulation.demo.settings import add_demo_arguments, demo_options
from dasklearn.tasks.task import Task
from dasklearn.simulation.choco.settings import add_choco_arguments, choco_options
from dasklearn.simulation.qgm.settings import add_qgm_arguments, qgm_options
from dasklearn.util import MICROSECONDS


async def run_one(args, variant, profile, partitioner, seed):
    Task.COUNTER.clear()
    if variant.startswith("shatter-r"):
        cls, config = ShatterSimulation, ShatterSettings
        specific = {"k": args.chunks, "r": int(variant.split("-r")[1])}
        algorithm = "shatter"
    elif variant in BASELINE_VARIANTS:
        cls, config, specific = baseline_spec(variant, args.peers, args.sample_size,
            getattr(args, "group_size", 8), getattr(args, "averaging_steps", 1))
        algorithm = variant
    elif variant == "conflux":
        cls, config = ConfluxSimulation, ConfluxSettings
        algorithm = variant
        specific = {"sample_size": args.sample_size, "chunks_in_sample": args.chunks, "success_fraction": args.success_fraction}
    else:
        raise ValueError("Unknown comparison variant: " + variant)
    if variant in {'qgm','dsgdm','dsgd'}:
        mode=specific['mode'];specific.update(qgm_options(args))
        if variant!='qgm':specific['mode']=mode
        algorithm='qgm'
    if variant=='choco':specific.update(choco_options(args))
    if variant=='demo':specific.update(demo_options(args))
    if variant=='dspodfl':specific.update(dspodfl_options(args))
    configuration = config(algorithm=algorithm, seed=seed, work_dir=str(args.output / "runs"),
        dataset=args.dataset, model=getattr(args, "model", "mlp"), participants=args.peers, partitioner=partitioner,
        dataset_base_path=getattr(args, "dataset_base_path", None),
        torch_device_name=getattr(args, "device", "cpu"),
        evaluation_split=getattr(args, "evaluation_split", "test"),
        evaluation_examples=getattr(args, "evaluation_examples", 0),
        gpu_residency=getattr(args,"gpu_residency",True), training_batching=getattr(args,"training_batching",True),
        training_batch_size=getattr(args,"training_batch_size",2), gpu_state_cache_mb=getattr(args,"gpu_state_cache_mb",256),
        alpha=args.alpha, learning=LearningSettings(args.learning_rate, getattr(args, "momentum", 0),
            getattr(args, "weight_decay", 0), getattr(args, "batch_size", 32), args.local_steps),
        rounds=0, duration=int(args.duration * MICROSECONDS), stop="duration",
        test_interval=0, test_period=int(args.test_period * MICROSECONDS), test_method="snapshot",
        network_latency_ms=args.latency_ms, torch_threads=getattr(args, "torch_threads", 1), unit_testing=True, dry_run=True, **specific)
    if getattr(args, "model_bytes", None) is not None:
        configuration.model_size_bytes = args.model_bytes
    random.seed(seed)
    sim = cls(configuration)
    # Avoid collisions for multiple degrees and profiles under the same seed.
    sim.data_dir = str(args.output / "runs" / f"{variant}-{profile}-{partitioner}-s{seed}")
    configuration.data_dir = sim.data_dir
    sim.setup_directories()

    def capacities():
        rng = random.Random(seed + 100)
        order = list(range(args.peers))
        rng.shuffle(order)
        fast = set(order[:max(1, args.peers // 4)])
        compute = [rng.choice([.5, 1, 2, 4]) for _ in range(args.peers)]
        compute_mean = np.mean(compute)
        bw_mean = (8 * len(fast) + args.peers - len(fast)) / args.peers
        for client in sim.clients:
            if getattr(args, "equal_capacity", False):
                client.bw_scheduler.bw_limit = args.bandwidth if profile == "uniform" else int(args.bandwidth * (8 if client.index in fast else 1) / bw_mean)
                from dasklearn import AUGMENTATION_FACTOR_SIM
                client.simulated_speed = args.step_ms / AUGMENTATION_FACTOR_SIM * (1 if profile == "uniform" else compute[client.index] / compute_mean)
                continue
            client.bw_scheduler.bw_limit = args.bandwidth if profile == "uniform" else (args.bandwidth * 4 if client.index in fast else args.bandwidth // 2)
            # Independent compute and bandwidth rankings, avoiding an implicit
            # assumption that fast networking implies fast local training.
            client.simulated_speed = 3 if profile == "uniform" else rng.choice([2, 5, 12, 30])

    sim.apply_traces = capacities
    network_rows = []
    snapshot = sim.test_snapshot
    def recorded_snapshot(event):
        # Include bytes already sent by unfinished transfers at the checkpoint.
        sent = sum(client.bw_scheduler.total_bytes_sent for client in sim.clients)
        # Project without calling update(): observing a checkpoint must not
        # change the transfer scheduler's integer rounding or future events.
        active = sum(min(t.transfer_size, t.transferred + int(
            (event.time-t.last_time_updated)/MICROSECONDS*t.allocated_bw))
            for client in sim.clients for t in client.bw_scheduler.outgoing_transfers.values())
        network_rows.append({"time": event.time/MICROSECONDS, "bytes_sent": sent+active})
        snapshot(event)
    sim.test_snapshot = recorded_snapshot
    started = time.perf_counter()
    sim.simulate()
    if getattr(args,"final_test",False):
        if configuration.evaluation_split=="test":raise ValueError("Use validation curves with a separate final test")
        for peer,(round_nr,model) in sorted(sim.latest_models.items()):
            sim.clients[peer].add_compute_task(Task(Task.generate_name("final_test"),"test",{
                "model":model,"round":round_nr,"peer":peer,"time":int(args.duration*MICROSECONDS),
                "evaluation_split":"test","evaluation_examples":0,"output_prefix":"heldout"}))
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    def progress(status):
        print(json.dumps({"run": Path(sim.data_dir).name, **status}), flush=True)
    execute_local(sim, retain_results=not getattr(args, "streaming", False),
                  progress=progress if getattr(args, "streaming", False) else None)
    (Path(sim.data_dir)/"training-executor.json").write_text(json.dumps(sim.execution_training_stats,indent=2)+"\n")
    sim.merge_accuracies_files()
    frame = pd.read_csv(Path(sim.data_dir) / "accuracies.csv")
    pd.DataFrame(network_rows).to_csv(Path(sim.data_dir)/"network-snapshots.csv", index=False)
    if variant == "demo":
        completions = [{"peer": c.index, "round": r, "time": s["completed_time"] / MICROSECONDS}
            for c in sim.clients for r, s in c.rounds.items() if s["done"]]
        pd.DataFrame(completions).to_csv(Path(sim.data_dir) / "demo-rounds.csv", index=False)
    curve = frame.groupby("time", as_index=False).agg(accuracy=("accuracy", "mean"), loss=("loss", "mean"), nodes=("peer", "nunique"))
    if not curve.nodes.eq(args.peers).all():
        raise RuntimeError("A snapshot omitted participating nodes")
    for name, value in [("variant", variant), ("profile", profile), ("partitioner", partitioner), ("seed", seed)]:
        curve[name] = value
    row = {"variant": variant, "profile": profile, "partitioner": partitioner, "seed": seed,
           "learning_rate": args.learning_rate,
           "final_accuracy": float(curve.accuracy.iloc[-1]), "final_loss": float(curve.loss.iloc[-1]),
           "model_bytes": sim.model_size, "bytes_sent": sum(c.bw_scheduler.total_bytes_sent for c in sim.clients),
           "training_tasks": sum(t.func in {"train", "demo_prepare", "qgm_train"} for t in sim.workflow_dag.tasks.values()),
           "simulated_compute_s": sum(c.compute_time for c in sim.clients) / MICROSECONDS,
           "actual_execution_s": time.perf_counter() - started}
    if getattr(args,"final_test",False):
        heldout=pd.concat([pd.read_csv(Path(sim.data_dir)/f"heldout_{peer}.csv") for peer in range(args.peers)],ignore_index=True)
        if len(heldout)!=args.peers or heldout.peer.nunique()!=args.peers:raise ValueError("Missing heldout node metrics")
        heldout.to_csv(Path(sim.data_dir)/"heldout.csv",index=False)
        row.update(final_test_accuracy=float(heldout.accuracy.mean()),final_test_loss=float(heldout.loss.mean()))
    for target in [40, 50, 60, 70, 80]:
        reached = curve.loc[curve.accuracy >= target, "time"]
        row[f"tta_{target}_s"] = float(reached.iloc[0]) if len(reached) else None
    print(json.dumps(row), flush=True)
    return row, curve


def plot(curves, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    combinations = list(curves.groupby(["profile", "partitioner"], sort=True))
    fig, axes = plt.subplots(1, len(combinations), figsize=(5 * len(combinations), 3.6), squeeze=False)
    for axis, ((profile, partitioner), data) in zip(axes[0], combinations):
        for variant, group in data.groupby("variant"):
            values = group.groupby("time").accuracy
            mean, std = values.mean(), values.std().fillna(0)
            axis.plot(mean.index, mean, label=variant)
            axis.fill_between(mean.index, mean - std, mean + std, alpha=0.15)
        axis.set(title=f"{profile}, {partitioner}", xlabel="Simulated time [s]", ylabel="Mean available-node accuracy [%]")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "accuracy.png", dpi=160)
    plt.close(fig)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/algorithm-validation"))
    parser.add_argument("--dataset", choices=["synthetic", "digits"], default="synthetic")
    parser.add_argument("--variants", nargs="+", default=["shatter-r2", "shatter-r4", "moshpit"])
    parser.add_argument("--profiles", nargs="+", choices=["uniform", "heterogeneous"], default=["uniform", "heterogeneous"])
    parser.add_argument("--partitioners", nargs="+", choices=["iid", "dirichlet"], default=["iid", "dirichlet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--peers", type=int, default=24)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--success-fraction", type=float, default=1)
    add_dspodfl_arguments(parser)
    add_demo_arguments(parser)
    add_choco_arguments(parser)
    add_qgm_arguments(parser)
    parser.add_argument("--duration", type=float, default=90)
    parser.add_argument("--test-period", type=float, default=5)
    parser.add_argument("--bandwidth", type=int, default=4000)
    parser.add_argument("--latency-ms", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3)
    add_training_arguments(parser)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    root = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for path in sorted(list((root / "dasklearn").rglob("*.py")) + [root / "args.py", root / "main.py", Path(__file__).resolve()]):
        source_hash.update(str(path.relative_to(root)).encode())
        source_hash.update(path.read_bytes())
    manifest = {**vars(args), "output": str(args.output), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "source_sha256": source_hash.hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows, curves = [], []
    for profile in args.profiles:
        for partitioner in args.partitioners:
            for seed in args.seeds:
                for variant in args.variants:
                    row, curve = await run_one(args, variant, profile, partitioner, seed)
                    rows.append(row)
                    curves.append(curve)
                    pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)
                    pd.concat(curves, ignore_index=True).to_csv(args.output / "curves.csv", index=False)
    plot(pd.concat(curves, ignore_index=True), args.output)


if __name__ == "__main__":
    asyncio.run(main())
