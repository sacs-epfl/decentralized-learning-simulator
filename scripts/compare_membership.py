#!/usr/bin/env python3
"""Compare loss-based MIA through local or received models at equal training budgets."""
import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import time
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from dasklearn.privacy import add_local_model_mia_task, IncomingModelObserver
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.dpsgd.settings import DPSGDSettings
from dasklearn.simulation.dpsgd.simulation import DPSGDSimulation
from dasklearn.simulation.shatter.settings import ShatterSettings
from dasklearn.simulation.shatter.simulation import ShatterSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS


def run_one(args, variant, partitioner, seed):
    Task.COUNTER.clear()
    torch.manual_seed(seed)
    if variant == "dpsgd":
        cls, config_type = DPSGDSimulation, DPSGDSettings
        algorithm, specific = "dpsgd", {"topology": "kreg", "k": args.degree}
    else:
        cls, config_type = ShatterSimulation, ShatterSettings
        algorithm, specific = "shatter", {"k": int(variant.removeprefix("shatter-k")), "r": args.degree}
    config = config_type(algorithm=algorithm, seed=seed, work_dir=str(args.output / "runs"),
        dataset=args.dataset, model=args.model, dataset_base_path=args.dataset_base_path,
        learning=LearningSettings(args.learning_rate, 0, 0, args.batch_size, args.local_steps),
        participants=args.peers, partitioner=partitioner, alpha=args.alpha, rounds=args.rounds,
        model_size_bytes=args.model_bytes,
        duration=int(args.max_duration * MICROSECONDS), stop="rounds", test_method="individual",
        test_interval=args.checkpoint_interval, test_period=0, torch_threads=1,
        network_latency_ms=args.latency_ms, dry_run=True, unit_testing=True, **specific)
    sim = cls(config)
    sim.data_dir = config.data_dir = str(args.output / "runs" / f"{variant}-{partitioner}-s{seed}")
    sim.setup_directories()
    observer = None
    if args.view == "incoming":
        observer = IncomingModelObserver(attackers=args.attackers,
            rounds=range(args.checkpoint_interval, args.rounds + 1, args.checkpoint_interval),
            chunk_count=specific["k"] if algorithm == "shatter" else None)
        sim.observers.append(observer)

    def capacities():
        # Equal compute/network capacity. Protocols retain their actual byte costs.
        for client in sim.clients:
            client.simulated_speed = 1
            client.bw_scheduler.bw_limit = args.bandwidth
    sim.apply_traces = capacities
    sim.simulate()
    training_tasks = [task for task in sim.workflow_dag.tasks.values() if task.func == "train"]
    if len(training_tasks) != args.peers * args.rounds:
        raise RuntimeError("Simulation did not complete the matched training budget; increase --max-duration")
    if any(round_nr != args.rounds for round_nr, _ in sim.latest_models.values()):
        raise RuntimeError("A peer did not finish its final aggregation")
    audits = []
    if observer is not None:
        audits = observer.attach_audits(sim, max_examples=args.examples, target_fpr=args.target_fpr,
                                       cohort_policy=args.cohort_policy)
        ledger = [asdict(value) for _, value in sorted(observer.observations.items())]
        (Path(sim.data_dir) / "incoming-observations.json").write_text(json.dumps(ledger, indent=2) + "\n")
        if not audits:
            raise RuntimeError("No completed incoming model deliveries at the selected rounds")
    else:
        # Local-model audits evaluate the complete states used for utility tests.
        for task in list(sim.workflow_dag.tasks.values()):
            if task.func == "test" and task.data["peer"] in args.attackers:
                audits.append(add_local_model_mia_task(sim, task.data["peer"], task.data["model"],
                    round_nr=task.data["round"], time_us=task.data["time"],
                    max_examples=args.examples, target_fpr=args.target_fpr, cohort_policy=args.cohort_policy))
        if len(audits) != len(args.attackers) * (args.rounds // args.checkpoint_interval):
            raise RuntimeError("Missing completed local-model checkpoints")
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    sim.workflow_dag.save_to_file(str(Path(sim.data_dir) / "workflow_dag"))
    started = time.perf_counter()
    execute_local(sim, retain_results=False)
    execution_s = time.perf_counter() - started
    sim.merge_accuracies_files()
    utility = pd.read_csv(Path(sim.data_dir) / "accuracies.csv")
    rows, audit_s = [], 0
    for task in audits:
        artifact = json.loads((Path(sim.data_dir) / f"mia-{task.name}.json").read_text())
        audit_s += artifact["execution_s"]
        for row in artifact["rows"]:
            match = utility[(utility.peer == row["attacker"]) & (utility["round"] == row["round"])]
            if len(match) != 1:
                raise RuntimeError("Missing or duplicate utility result for attacker checkpoint")
            row.update(variant=variant, partitioner=partitioner, seed=seed,
                       possible_pairs=len(args.attackers) * (args.peers - 1),
                       model_accuracy=float(match.accuracy.iloc[0]), model_loss=float(match.loss.iloc[0]))
            rows.append(row)
    run = {"variant": variant, "partitioner": partitioner, "seed": seed,
           "model_bytes": sim.model_size,
           "training_tasks": len(training_tasks), "local_sgd_steps": len(training_tasks) * args.local_steps,
           "simulated_time_s": sim.current_time / MICROSECONDS,
           "bytes_sent": sum(client.bw_scheduler.total_bytes_sent for client in sim.clients),
           "execution_including_audit_s": execution_s, "audit_execution_s": audit_s}
    print(json.dumps(run), flush=True)
    return rows, run


def summarize(frame):
    """Average pairs within seeds, then summarize independent training seeds."""
    metrics = ["balanced_accuracy", "roc_auc", "tpr", "fpr", "model_accuracy", "model_loss"]
    metrics += [key for key in ["visible_state_fraction", "received_chunks"] if key in frame]
    keys = ["variant", "partitioner", "seed", "round"]
    per_seed = frame.groupby(keys, as_index=False)[metrics].mean()
    if "possible_pairs" in frame:
        coverage = frame.groupby(keys, as_index=False).agg(
            observed_pairs=("attacker", "size"), possible_pairs=("possible_pairs", "first"))
        coverage["observed_pair_fraction"] = coverage.observed_pairs / coverage.possible_pairs
        per_seed = per_seed.merge(coverage, on=keys, validate="one_to_one")
        metrics += ["observed_pairs", "observed_pair_fraction"]
    rows = []
    for (variant, partitioner, round_nr), group in per_seed.groupby(["variant", "partitioner", "round"]):
        row = {"variant": variant, "partitioner": partitioner, "round": int(round_nr), "seeds": len(group)}
        for metric in metrics:
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return per_seed, pd.DataFrame(rows)


def plot(summary, output, view="local"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    partitioners = sorted(summary.partitioner.unique())
    fig, axes = plt.subplots(2, len(partitioners), figsize=(6 * len(partitioners), 6), squeeze=False)
    for col, partitioner in enumerate(partitioners):
        for variant, data in summary[summary.partitioner == partitioner].groupby("variant"):
            for axis, metric in [(axes[0, col], "balanced_accuracy"), (axes[1, col], "model_accuracy")]:
                scale = 100 if metric == "balanced_accuracy" else 1
                mean, std = data[metric + "_mean"] * scale, data[metric + "_std"] * scale
                axis.plot(data["round"], mean, marker="o", label=variant)
                axis.fill_between(data["round"], mean - std, mean + std, alpha=0.15)
        axes[0, col].axhline(50, color="grey", linestyle="--", label="MIA chance")
        axes[0, col].set(title=partitioner, ylabel="Balanced MIA accuracy [%]")
        axes[1, col].set(xlabel="Completed rounds (equal local SGD steps)", ylabel="Local model accuracy [%]")
        for axis in axes[:, col]:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    title = "received model / receiver-filled chunks" if view == "incoming" else "attacker's local model"
    fig.suptitle(f"Loss attack: {title}; shaded bands: seed SD")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/membership-comparison"))
    parser.add_argument("--view", choices=["local", "incoming"], default="local",
                        help="Incoming: delivered full models, or chunks filled with receiver-known state")
    parser.add_argument("--cohort-policy", choices=["class_matched", "global_nonmembers"], default="class_matched",
                        help="Match candidate labels, or retain the global nonmember label distribution")
    parser.add_argument("--variants", nargs="+", choices=["dpsgd", "shatter-k1", "shatter-k4", "shatter-k16"],
                        default=["dpsgd", "shatter-k4", "shatter-k16"])
    parser.add_argument("--dataset", choices=["digits", "synthetic", "cifar10"], default="digits")
    parser.add_argument("--dataset-base-path")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--partitioners", nargs="+", choices=["iid", "dirichlet"], default=["iid", "dirichlet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--peers", type=int, default=8)
    parser.add_argument("--attackers", nargs="+", type=int, help="Defaults to all peers, audited separately without collusion")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--examples", type=int, default=128, help="Maximum examples in each of the three audit sets per target")
    parser.add_argument("--target-fpr", type=float, default=0.1, help="Empirical FPR budget on independent calibration nonmembers")
    parser.add_argument("--bandwidth", type=int, default=1_000_000)
    parser.add_argument("--model-bytes", type=int, help="Fix simulated full-model wire size for paired experiments")
    parser.add_argument("--latency-ms", type=float, default=2)
    parser.add_argument("--max-duration", type=float, default=3600, help="Simulation safety limit in seconds")
    args = parser.parse_args()
    args.attackers = sorted(set(range(args.peers) if args.attackers is None else args.attackers))
    if args.peers < 2 or not args.attackers or any(peer < 0 or peer >= args.peers for peer in args.attackers):
        parser.error("provide at least two peers and valid attacker indices")
    if args.rounds < 1 or args.checkpoint_interval < 1 or args.rounds % args.checkpoint_interval:
        parser.error("rounds must be a positive multiple of checkpoint-interval")
    if min(args.local_steps, args.batch_size, args.examples, args.bandwidth) < 1 or not 0 < args.target_fpr < 1:
        parser.error("steps, batch size, examples and bandwidth must be positive; 0 < target-fpr < 1")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    if args.model_bytes is not None and args.model_bytes < 1:
        parser.error("model-bytes must be positive")
    if not 0 < args.degree < args.peers or args.degree * args.peers % 2:
        parser.error("degree must define a valid regular graph on real peers")
    return args


async def main():
    args = parse_args()
    args.output = args.output.resolve()
    # Preserve previous experiments, including unknown partial runs.
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.WARNING)
    source_hash = hashlib.sha256()
    for path in sorted([*(ROOT / "dasklearn").rglob("*.py"), Path(__file__).resolve()]):
        source_hash.update(str(path.relative_to(ROOT)).encode())
        source_hash.update(path.read_bytes())
    manifest = {**vars(args), "output": str(args.output), "python": platform.python_version(),
                "torch": torch.__version__, "numpy": np.__version__, "source_sha256": source_hash.hexdigest(),
                "attacker_view": ("actual received full models; received same-sender/same-round chunks "
                    "completed with receiver model frozen at first delivery; known sender IDs"
                    if args.view == "incoming" else "own local model; other peers' partitions are audit ground truth"),
                "membership": "assigned to target training partition (not a sampled-example trace)",
                "threshold": "loss < independent nonmember calibration quantile; cohort_policy defines label distribution",
                "aggregation": "equal-weight observed attacker/target pairs within seeds; mean and sample SD across seeds; unobserved pairs excluded",
                "heldout_use": "test split is divided into calibration and evaluation pools; no attack/utility tuning"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows, runs = [], []
    for partitioner in args.partitioners:
        for seed in args.seeds:
            for variant in args.variants:
                records, run = run_one(args, variant, partitioner, seed)
                rows.extend(records)
                runs.append(run)
                frame = pd.DataFrame(rows)
                frame.to_csv(args.output / "attacks.csv", index=False)
                pd.DataFrame(runs).to_csv(args.output / "runs.csv", index=False)
                per_seed, summary = summarize(frame)
                per_seed.to_csv(args.output / "per-seed.csv", index=False)
                summary.to_csv(args.output / "summary.csv", index=False)
    # Same target cohorts must be used for every model and every protocol.
    if frame.groupby(["partitioner", "seed", "target"]).cohort_sha256.nunique().max() != 1:
        raise RuntimeError("Comparison used different audit examples across protocols/checkpoints")
    plot(summary, args.output, args.view)
    print(summary[summary["round"] == args.rounds].to_string(index=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
