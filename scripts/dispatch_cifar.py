"""Select pilot learning rates using validation data and dispatch one job per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--peers", type=int, default=128)
p.add_argument("--duration", type=int, default=900)
p.add_argument("--test-period", type=int, default=90)
p.add_argument("--step-ms", type=float, default=20)
p.add_argument("--bandwidth", type=int, default=100000)
p.add_argument("--variants", nargs="+", choices=["shatter-r2", "shatter-r4", "moshpit", "conflux", "conflux-v2"],
               default=["shatter-r2", "shatter-r4", "moshpit"])
args = p.parse_args()
root = args.root.resolve()
output = root / "results" / f"main{args.peers}"
output.mkdir(parents=True, exist_ok=True)
variants = args.variants
jobs, selection = [], []
for profile in ["uniform", "heterogeneous"]:
    candidates = []
    for path in sorted((root / "results").glob(f"pilot64-{profile}-lr*")):
        if not (path / "COMPLETE").exists():
            raise RuntimeError(f"Incomplete pilot: {path}")
        manifest = json.loads((path / "manifest.json").read_text())
        curves = pd.read_csv(path / "curves.csv")
        times = sorted(curves.time.unique())[-2:]
        for variant, frame in curves[curves.time.isin(times)].groupby("variant"):
            candidates.append({"variant": variant, "profile": profile,
                               "learning_rate": manifest["learning_rate"],
                               "validation_score": float(frame.accuracy.mean()), "pilot": str(path)})
    for variant in variants:
        choices = [row for row in candidates if row["variant"] == variant]
        if len(choices) != 3:
            raise ValueError(f"Expected three learning rates for {variant}/{profile}")
        best = max(choices, key=lambda row: (row["validation_score"], -row["learning_rate"]))
        selection.append({"selected": best, "candidates": choices})
        job = {"variant": variant, "profile": profile, "learning_rate": best["learning_rate"],
               "output": str(output / f"{variant}-{profile}"), "status": "pending"}
        jobs.append(job)
plan = {"peers": args.peers, "duration": args.duration, "step_ms": args.step_ms,
        "test_period": args.test_period,
        "variants": variants,
        "bandwidth": args.bandwidth, "selection_rule": "highest mean validation accuracy over final two pilot snapshots; seed 100, Dirichlet alpha=.3",
        "selection": selection}
plan_path = output / "plan.json"
if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
    raise ValueError("Existing main plan differs")
plan_path.write_text(json.dumps(plan, indent=2) + "\n")


def run_gpu(gpu):
    assigned = jobs[gpu::2]
    for job in assigned:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        job["gpu"], job["status"], job["started"] = gpu, "running", time.time()
        status = output / f"status-gpu{gpu}.json"
        status.write_text(json.dumps(assigned, indent=2) + "\n")
        cmd = [sys.executable, str(Path(__file__).with_name("run_cifar.py")),
               "--dataset-base-path", str(root / "datasets"), "--output", job["output"],
               "--variants", job["variant"], "--profiles", job["profile"],
               "--peers", str(args.peers), "--sample-size", str(args.peers // 4),
               "--chunks", "16", "--duration", str(args.duration), "--test-period", str(args.test_period),
               "--learning-rate", str(job["learning_rate"]), "--step-ms", str(args.step_ms),
               "--bandwidth", str(args.bandwidth)]
        with (output / f"{job['variant']}-{job['profile']}.log").open("a") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        job["returncode"], job["finished"] = result.returncode, time.time()
        job["status"] = "complete" if result.returncode == 0 else "failed"
        status.write_text(json.dumps(assigned, indent=2) + "\n")
        print(json.dumps(job), flush=True)


with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run_gpu, [0, 1]))
if any(job["status"] != "complete" for job in jobs):
    raise RuntimeError("One or more GPU sweeps failed; inspect status files")
frames = [pd.read_csv(Path(job["output"]) / "summary.csv") for job in jobs]
curves = [pd.read_csv(Path(job["output"]) / "curves.csv") for job in jobs]
pd.concat(frames, ignore_index=True).to_csv(output / "summary.csv", index=False)
pd.concat(curves, ignore_index=True).to_csv(output / "curves.csv", index=False)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_conflux import plot
plot(pd.concat(curves, ignore_index=True), output)
(output / "COMPLETE").write_text(f"{len(jobs) * 6} paired runs completed\n")
