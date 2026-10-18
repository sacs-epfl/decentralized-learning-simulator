"""Run the real Shatter/Conflux event protocols and execute their training DAGs.

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
from dasklearn.simulation.conflux_v2.settings import ConfluxV2Settings
from dasklearn.simulation.conflux_v2.simulation import ConfluxV2Simulation
from dasklearn.simulation.shatter.settings import ShatterSettings
from dasklearn.simulation.shatter.simulation import ShatterSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS


async def run_one(args, variant, profile, partitioner, seed):
    Task.COUNTER.clear()
    if variant.startswith("shatter-r"):
        cls, config = ShatterSimulation, ShatterSettings
        specific = {"k": args.chunks, "r": int(variant.split("-r")[1])}
        algorithm = "shatter"
    else:
        cls, config = (ConfluxV2Simulation, ConfluxV2Settings) if variant == "conflux-v2" else (ConfluxSimulation, ConfluxSettings)
        algorithm = variant
        specific = {"sample_size": args.sample_size, "chunks_in_sample": args.chunks, "success_fraction": args.success_fraction}
    configuration = config(algorithm=algorithm, seed=seed, work_dir=str(args.output / "runs"),
        dataset=args.dataset, model="mlp", participants=args.peers, partitioner=partitioner,
        alpha=args.alpha, learning=LearningSettings(args.learning_rate, 0, 0, 32, args.local_steps),
        rounds=0, duration=int(args.duration * MICROSECONDS), stop="duration",
        test_interval=0, test_period=int(args.test_period * MICROSECONDS), test_method="snapshot",
        network_latency_ms=args.latency_ms, torch_threads=1, unit_testing=True, dry_run=True, **specific)
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
        for client in sim.clients:
            client.bw_scheduler.bw_limit = args.bandwidth if profile == "uniform" else (args.bandwidth * 4 if client.index in fast else args.bandwidth // 2)
            # Independent compute and bandwidth rankings, avoiding an implicit
            # assumption that fast networking implies fast local training.
            client.simulated_speed = 3 if profile == "uniform" else rng.choice([2, 5, 12, 30])

    sim.apply_traces = capacities
    started = time.perf_counter()
    sim.simulate()
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    execute_local(sim)
    sim.merge_accuracies_files()
    frame = pd.read_csv(Path(sim.data_dir) / "accuracies.csv")
    curve = frame.groupby("time", as_index=False).agg(accuracy=("accuracy", "mean"), loss=("loss", "mean"), nodes=("peer", "nunique"))
    if not curve.nodes.eq(args.peers).all():
        raise RuntimeError("A snapshot omitted participating nodes")
    for name, value in [("variant", variant), ("profile", profile), ("partitioner", partitioner), ("seed", seed)]:
        curve[name] = value
    row = {"variant": variant, "profile": profile, "partitioner": partitioner, "seed": seed,
           "final_accuracy": float(curve.accuracy.iloc[-1]), "final_loss": float(curve.loss.iloc[-1]),
           "model_bytes": sim.model_size, "bytes_sent": sum(c.bw_scheduler.total_bytes_sent for c in sim.clients),
           "training_tasks": sum(t.func == "train" for t in sim.workflow_dag.tasks.values()),
           "simulated_compute_s": sum(c.compute_time for c in sim.clients) / MICROSECONDS,
           "actual_execution_s": time.perf_counter() - started}
    for target in [60, 70, 80]:
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
    parser.add_argument("--output", type=Path, default=Path("output/conflux-validation"))
    parser.add_argument("--dataset", choices=["synthetic", "digits"], default="synthetic")
    parser.add_argument("--variants", nargs="+", default=["shatter-r2", "shatter-r4", "conflux", "conflux-v2"])
    parser.add_argument("--profiles", nargs="+", choices=["uniform", "heterogeneous"], default=["uniform", "heterogeneous"])
    parser.add_argument("--partitioners", nargs="+", choices=["iid", "dirichlet"], default=["iid", "dirichlet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--peers", type=int, default=24)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--success-fraction", type=float, default=1)
    parser.add_argument("--duration", type=float, default=90)
    parser.add_argument("--test-period", type=float, default=5)
    parser.add_argument("--bandwidth", type=int, default=4000)
    parser.add_argument("--latency-ms", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3)
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
