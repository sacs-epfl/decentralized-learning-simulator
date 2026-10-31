"""Isolated experiment runs, immutable output directories, and provenance."""
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance(root):
    root = Path(root)
    digest = hashlib.sha256()
    files = [root / "args.py", root / "main.py", *sorted((root / "dasklearn").rglob("*.py")),
             *sorted((root / "scripts").glob("*.py"))]
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    packages = {}
    for name in ("torch", "torchvision", "numpy", "datasets", "flwr-datasets", "transformers", "peft", "PyYAML"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")),
                source_sha256=digest.hexdigest(), python=platform.python_version(), packages=packages)


def metrics(directory):
    """Latest observation per evaluated peer; no extra model averaging/evaluation."""
    paths = list(directory.glob("data/*/accuracies.csv"))
    if not paths:
        return {}
    latest = {}
    with paths[0].open(newline="") as source:
        for row in csv.DictReader(source):
            peer = int(row["peer"])
            order = float(row["time"]), int(row["round"])
            if peer not in latest or order >= latest[peer][0]:
                latest[peer] = order, row
    if not latest:
        return {}
    rows = [row for _, row in latest.values()]
    if any(not math.isfinite(float(row[field])) for row in rows for field in ("accuracy", "loss", "time")):
        raise ValueError("Nonfinite evaluation metrics")
    return dict(evaluated_peers=len(rows), mean_accuracy=statistics.mean(float(row["accuracy"]) for row in rows),
                mean_loss=statistics.mean(float(row["loss"]) for row in rows),
                last_simulated_seconds=max(float(row["time"]) for row in rows))


def _stop(process):
    # Workers have their own process group so a Ctrl-C never leaves a broker
    # coordinator/local training child running behind the parent runner.
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def execute_experiment(experiment, *, continue_on_error=False):
    root = Path(__file__).resolve().parents[1]
    origin = provenance(root)
    try:
        experiment.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"Output directory already exists: {experiment.output}; choose a new --output") from exc
    (experiment.output / "runs").mkdir()
    (experiment.output / ("source" + experiment.path.suffix)).write_text(experiment.path.read_text())
    suite = dict(version=1, name=experiment.name, source_config=str(experiment.path), provenance=origin,
                 status="running", runs=[dict(name=run.name, status="planned") for run in experiment.runs])
    write_json(experiment.output / "manifest.json", suite)
    summaries = []
    exit_code = 0
    for index, run in enumerate(experiment.runs):
        directory = experiment.output / "runs" / run.name
        directory.mkdir()
        resolved = dict(version=1, name=experiment.name, execution=run.execution,
                        output=str(directory), settings=run.options)
        write_json(directory / "resolved.json", resolved)
        record = dict(version=1, name=run.name, execution=run.execution, status="running", options=run.options,
                      provenance=origin, started_at=datetime.now(timezone.utc).isoformat())
        suite["runs"][index]["status"] = "running"
        write_json(directory / "manifest.json", record)
        write_json(experiment.output / "manifest.json", suite)
        command = [sys.executable, str(root / "main.py"), "--config", str(directory / "resolved.json")]
        record["command"] = command
        started = time.monotonic()
        process = None
        interrupted = False
        print(f"[{index + 1}/{len(experiment.runs)}] {run.name}: {directory}", flush=True)
        try:
            with (directory / "run.log").open("w") as log:
                process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                           env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)
                record["returncode"] = process.wait()
            record["status"] = "completed" if record["returncode"] == 0 else "failed"
        except KeyboardInterrupt:
            interrupted = True
            if process is not None:
                _stop(process)
            record.update(status="interrupted", returncode=130)
        except OSError as exc:
            record.update(status="failed", returncode=1, error=str(exc))
        record["wall_seconds"] = time.monotonic() - started
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["metrics"] = {}
        try:
            settings = list(directory.glob("data/*/settings.json"))
            if settings:
                record["settings"] = json.loads(settings[0].read_text())
                record["artifacts"] = str(settings[0].parent)
            record["metrics"] = metrics(directory)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record["error"] = f"Could not summarize run artifacts: {exc}"
            if not interrupted:
                record["status"] = "failed"
        write_json(directory / "manifest.json", record)
        suite["runs"][index].update(status=record["status"], wall_seconds=record["wall_seconds"])
        row = dict(run=run.name, algorithm=run.options["algorithm"], dataset=run.options["dataset"],
                   partitioner=run.options["partitioner"], seed=run.options["seed"], status=record["status"],
                   wall_seconds=record["wall_seconds"], **record["metrics"])
        summaries.append(row)
        columns = ["run", "algorithm", "dataset", "partitioner", "seed", "status", "wall_seconds",
                   "evaluated_peers", "mean_accuracy", "mean_loss", "last_simulated_seconds"]
        with (experiment.output / "summary.csv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summaries)
        if record["status"] != "completed":
            exit_code = 130 if interrupted else 1
        if interrupted or exit_code and not continue_on_error:
            break
        write_json(experiment.output / "manifest.json", suite)
    suite["status"] = "interrupted" if exit_code == 130 else "failed" if exit_code else "completed"
    write_json(experiment.output / "manifest.json", suite)
    return exit_code
