"""One resumable CIFAR sweep per GPU; all comparisons execute real protocol DAGs."""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
from validate_conflux import run_one, plot
from dasklearn.tasks.training import add_training_arguments


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-base-path", required=True)
    p.add_argument("--variants", nargs="+", default=["shatter-r2", "shatter-r4", "conflux-v2"])
    p.add_argument("--profiles", nargs="+", choices=["uniform", "heterogeneous"], default=["uniform", "heterogeneous"])
    p.add_argument("--partitioners", nargs="+", choices=["iid", "dirichlet"], default=["iid", "dirichlet"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--peers", type=int, default=128)
    p.add_argument("--sample-size", type=int, default=32)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--averaging-steps", type=int, default=1)
    p.add_argument("--model-bytes", type=int)
    p.add_argument("--duration", type=float, default=1800)
    p.add_argument("--test-period", type=float, default=120)
    p.add_argument("--bandwidth", type=int, default=100000)
    p.add_argument("--step-ms", type=float, default=20)
    p.add_argument("--latency-ms", type=float, default=20)
    p.add_argument("--learning-rate", type=float, default=.03)
    p.add_argument("--local-steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--alpha", type=float, default=.3)
    p.add_argument("--momentum", type=float, default=0)
    p.add_argument("--weight-decay", type=float, default=0)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--evaluation-split", choices=["test", "validation"], default="test")
    p.add_argument("--device", default="cuda:0")
    add_training_arguments(p)
    args = p.parse_args()
    args.dataset, args.model = "cifar10", "gnlenet"
    args.success_fraction, args.equal_capacity, args.streaming = 1, True, True
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(list((root / "dasklearn").rglob("*.py")) + list((root / "scripts").glob("*.py"))):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    manifest = {**vars(args), "output": str(args.output), "python": platform.python_version(),
                "torch": torch.__version__, "numpy": np.__version__, "source_sha256": digest.hexdigest(),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "split": "45000 training / 5000 validation / 10000 test; no augmentation",
                "capacity": "same mean bandwidth and compute step time across profiles"}
    provenance = Path(args.dataset_base_path) / "cifar10-hf-manifest.json"
    if provenance.exists():
        manifest["dataset_provenance"] = json.loads(provenance.read_text())
    manifest = json.loads(json.dumps(manifest, default=str))
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Refusing to resume with different source/configuration")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    summary, curve_path = args.output / "summary.csv", args.output / "curves.csv"
    rows = pd.read_csv(summary).to_dict("records") if summary.exists() else []
    curves = [pd.read_csv(curve_path)] if curve_path.exists() else []
    done = {(r["variant"], r["profile"], r["partitioner"], r["seed"]) for r in rows}
    for profile in args.profiles:
        for partitioner in args.partitioners:
            for seed in args.seeds:
                for variant in args.variants:
                    if (variant, profile, partitioner, seed) in done:
                        continue
                    print(json.dumps({"starting": [variant, profile, partitioner, seed]}), flush=True)
                    run_dir = args.output / "runs" / f"{variant}-{profile}-{partitioner}-s{seed}"
                    # Only remove incomplete accuracy outputs on resume; completed
                    # runs are skipped using the atomically written summary.
                    if run_dir.exists():
                        for file in run_dir.glob("accuracies*.csv"):
                            file.unlink()
                    row, curve = await run_one(args, variant, profile, partitioner, seed)
                    rows.append(row)
                    curves.append(curve)
                    pd.concat(curves, ignore_index=True).to_csv(curve_path.with_suffix(".tmp"), index=False)
                    curve_path.with_suffix(".tmp").replace(curve_path)
                    pd.DataFrame(rows).to_csv(summary.with_suffix(".tmp"), index=False)
                    summary.with_suffix(".tmp").replace(summary)
                    plot(pd.concat(curves, ignore_index=True), args.output)
    (args.output / "COMPLETE").write_text("All requested runs completed\n")


if __name__ == "__main__":
    asyncio.run(main())
