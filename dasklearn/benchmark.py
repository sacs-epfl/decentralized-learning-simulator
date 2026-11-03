"""Paired, fixed simulated-time benchmarks and live, auditable leaderboards.

The JSON result is the interface to both the dashboard and offline reports.
Evaluation can use individual client models or an offline global parameter average.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import io
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time

from dasklearn.experiments import CONTROL_OPTIONS, _name, read_document, validate_options, validate_run
from dasklearn.experiment_runner import _stop, provenance, write_json


@dataclass
class BenchmarkRun:
    name: str
    variant: str
    seed: int
    options: dict


@dataclass
class Benchmark:
    path: Path
    name: str
    output: Path
    budgets: list
    seeds: list
    times: list
    targets: list
    network: dict
    runs: list
    tuning: dict | None = None
    selection: dict | None = None
    evaluation_model: str = "local"
    execution_devices: list | None = None

    def manifest(self):
        return dict(version=1, kind="algorithm-leaderboard", name=self.name,
                    budgets=self.budgets, seeds=self.seeds, times=self.times,
                    accuracy_targets=self.targets, network=self.network,
                    time_basis="simulated_seconds", runs=[asdict(run) for run in self.runs],
                    tuning=self.tuning, selection=self.selection,
                    evaluation_model=self.evaluation_model, execution_devices=self.execution_devices)


def _numbers(value, name, *, integer=False, zero=False):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    for item in value:
        if (type(item) not in ((int,) if integer else (int, float)) or not math.isfinite(item)
                or item < 0 or (not zero and item == 0)):
            raise ValueError(f"{name} requires {'nonnegative' if zero else 'positive'} finite {'integers' if integer else 'numbers'}")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicates")
    return sorted(value)


def load_benchmark(path, parser, *, output=None, budgets=None, seeds=None, variants=None, period=None):
    """Resolve without loading models, datasets, or creating output directories."""
    path, document = read_document(path, extra_fields={"benchmark", "tuning"})
    if "matrix" in document or document.get("execution", "local") != "local":
        raise ValueError("Benchmarks use named runs and local numerical execution")
    spec = document.get("benchmark")
    allowed = {"budgets", "seeds", "evaluation_period", "accuracy_targets", "network", "evaluation_model", "execution_devices"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError(f"benchmark must be a mapping with fields {sorted(allowed)}")
    budgets = _numbers(budgets if budgets is not None else spec.get("budgets"), "budgets", integer=True)
    evaluation_model = spec.get("evaluation_model", "local")
    if not isinstance(evaluation_model, str) or evaluation_model not in {"local", "global"}:
        raise ValueError("benchmark.evaluation_model must be local or global")
    execution_devices = spec.get("execution_devices")
    if execution_devices is not None and (not isinstance(execution_devices, list) or not execution_devices
            or any(not isinstance(device, str) or not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", device) for device in execution_devices)):
        raise ValueError("execution_devices must be a nonempty list of concrete CPU/CUDA/MPS devices; repeated devices share a GPU")
    seeds = _numbers(seeds if seeds is not None else spec.get("seeds"), "seeds", integer=True, zero=True)
    period = period if period is not None else spec.get("evaluation_period", 30)
    _numbers([period], "evaluation_period", integer=True)
    targets = _numbers(spec.get("accuracy_targets", [50, 70, 80]), "accuracy_targets", zero=True)
    if max(targets) > 100:
        raise ValueError("accuracy_targets are percentages in [0, 100]")
    targets = [int(target) if target == int(target) else target for target in targets]
    if max(budgets) // period + len(budgets) + 1 > 10000:
        raise ValueError("More than 10,000 evaluation checkpoints; increase evaluation_period")
    times = sorted({0, *range(period, max(budgets) + 1, period), *budgets})
    network = dict(profile="uniform", bandwidth_bytes_per_second=1_000_000, compute_step_ms=200)
    override = spec.get("network", {})
    if not isinstance(override, dict) or set(override) - set(network):
        raise ValueError(f"network fields must be {sorted(network)}")
    network.update(override)
    if network["profile"] not in {"uniform", "heterogeneous"}:
        raise ValueError("network.profile must be uniform or heterogeneous")
    for key in ("bandwidth_bytes_per_second", "compute_step_ms"):
        _numbers([network[key]], key)
    if network["bandwidth_bytes_per_second"] < 1 or network["compute_step_ms"] < .001:
        raise ValueError("Network bandwidth must be >= 1 byte/s and compute_step_ms >= .001")
    base = validate_options(parser, document.get("settings", {}), path.parent)
    controlled = {"seed", "duration", "rounds", "stop", "test_method", "test_interval", "test_period"}
    if controlled & base.keys():
        raise ValueError(f"Use benchmark fields for {sorted(controlled & base.keys())}")
    defaults = vars(parser.parse_args([]))
    defaults.update(momentum=0, local_steps=1, synchronous=False, torch_threads=1,
                    compute_graph_plot_size=0, log_level="WARNING")
    entries = document.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("A benchmark requires nonempty named runs")
    tuning = {"algorithm", "learning_rate", "momentum", "weight_decay", "batch_size", "local_steps",
              "synchronous", "topology", "k", "r", "el", "group_size", "averaging_steps", "sample_size",
              "chunks_in_sample", "success_fraction", "agg", "wait", "no_weights", "gl_period", "fl_weighting"}
    prefixes = ("qgm_", "choco_", "swift_", "gt_dsgd_", "detag_", "relaysgd_", "sgp_", "damsgrad_",
                "fspda_", "motef_", "demo_", "dspod_", "teleportation_")
    runs, names = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"name", "settings"}:
            raise ValueError("Each run requires name and optional settings")
        label = _name(entry.get("name"))
        if label in names:
            raise ValueError("Run names must be unique")
        names.add(label)
        specific = validate_options(parser, entry.get("settings", {}), path.parent)
        if any(key not in tuning and not key.startswith(prefixes) for key in specific):
            raise ValueError("Named runs may change algorithm/tuning/graph options only; put the fixed scenario in settings")
        options = defaults | base | specific
        if options["churn"] != "none" or options["traces"] != "none":
            raise ValueError("Version 1 benchmarks require stable membership and use benchmark.network capacities")
        if options["feedback_replay"] is not None:
            raise ValueError("A paired benchmark cannot reuse another run's feedback replay")
        for seed in seeds:
            effective = options | dict(seed=seed, rounds=0, stop="duration", duration=max(budgets),
                                       test_method="snapshot", test_interval=0, test_period=0)
            validate_run(effective)
            effective = {key: value for key, value in effective.items() if key not in CONTROL_OPTIONS}
            runs.append(BenchmarkRun(f"{label}-s{seed}", label, seed, effective))
    if variants is not None:
        if set(variants) - names or len(set(variants)) != len(variants) or not variants:
            raise ValueError(f"variants must be unique names from {sorted(names)}")
        runs = [run for run in runs if run.variant in variants]
    if len(runs) > 10000:
        raise ValueError("Benchmark exceeds 10,000 runs")
    target = Path(output).expanduser().resolve() if output else (path.parent / Path(document.get("output", "../output/leaderboard")).expanduser()).resolve()
    benchmark = Benchmark(path, document["name"], target, budgets, seeds, times, targets, network, runs,
                          evaluation_model=evaluation_model, execution_devices=execution_devices)
    if "tuning" in document:
        from dasklearn.lr_tuning import validate_tuning
        benchmark.tuning = validate_tuning(document["tuning"], benchmark, names)
    return benchmark


def configure_simulation(simulation, times, network, evaluation_model="local"):
    """Install passive observations, leaving protocol updates and messages intact."""
    from dasklearn import AUGMENTATION_FACTOR_SIM
    from dasklearn.simulation.events import Event, TEST_SNAPSHOT
    from dasklearn.tasks.references import TaskRef
    from dasklearn.tasks.task import Task
    from dasklearn.util import MICROSECONDS
    initialize = simulation.initialize_clients
    measurements = []

    def capacities():
        rng = random.Random(simulation.settings.seed + 100)
        count = simulation.settings.participants
        speeds, bandwidths = [1.] * count, [1.] * count
        if network["profile"] == "heterogeneous":
            # Independent rankings, with identical total capacity across algorithms.
            speeds = [rng.choice([.5, 1, 2, 4]) for _ in range(count)]
            bandwidths = [rng.choice([.5, 1, 2, 4]) for _ in range(count)]
        for client, speed, bandwidth in zip(simulation.clients, speeds, bandwidths):
            client.simulated_speed = network["compute_step_ms"] / AUGMENTATION_FACTOR_SIM * speed / statistics.mean(speeds)
            client.bw_scheduler.bw_limit = max(1, int(network["bandwidth_bytes_per_second"] * bandwidth / statistics.mean(bandwidths)))

    def initialize_and_schedule():
        initialize()
        initial = simulation.clients[0].create_compute_task("initialize", {"peer": 0, "round": 0, "time": 0})
        # Fallback for participants that have not completed their first update.
        simulation.latest_models = {peer: (0, TaskRef(initial.name, 0)) for peer in range(simulation.settings.participants)}
        for seconds in times:
            simulation.schedule(Event(int(seconds * MICROSECONDS), 0, TEST_SNAPSHOT, is_global=True))

    def snapshot(event):
        if evaluation_model == "global":
            checkpoints = [simulation.latest_models[peer] for peer in range(simulation.settings.participants)]
            round_nr = max(round_nr for round_nr, _ in checkpoints)
            client = simulation.clients[0]
            aggregate = client.create_compute_task("aggregate", dict(
                models=[model for _, model in checkpoints], weights=[1 / len(checkpoints)] * len(checkpoints),
                round=round_nr, time=event.time, peer=-1), name=Task.generate_name("benchmark_global_average"))
            client.create_compute_task("test", dict(model=TaskRef(aggregate.name, 0), round=round_nr,
                time=event.time, peer=-1), name=Task.generate_name("benchmark_global_test"))
        else:
            for peer in range(simulation.settings.participants):
                round_nr, model = simulation.latest_models[peer]
                simulation.clients[peer].create_compute_task("test", dict(
                    model=model, round=round_nr, time=event.time, peer=peer), name=Task.generate_name("benchmark_snapshot"))
        sent = sum(client.bw_scheduler.total_bytes_sent for client in simulation.clients)
        active = sum(min(transfer.transfer_size, transfer.transferred + int(
            (event.time - transfer.last_time_updated) / MICROSECONDS * transfer.allocated_bw))
            for client in simulation.clients for transfer in client.bw_scheduler.outgoing_transfers.values())
        measurements.append(dict(time=event.time / MICROSECONDS, bytes_sent=sent + active))

    simulation.apply_traces = capacities
    simulation.initialize_clients = initialize_and_schedule
    simulation.test_snapshot = snapshot
    return measurements


async def run_worker(resolved):
    """One process per seed/variant, including RNG, dataset, and executor state."""
    import argparse
    import json
    import torch
    from dasklearn.algorithms import class_paths, import_class
    from dasklearn.settings_factory import session_options
    from dasklearn.tasks.local import execute_local
    from dasklearn.tasks.task import Task
    document = json.loads(Path(resolved).read_text())
    args = argparse.Namespace(**document["options"], local=True, dry_run=False, work_dir=document["output"])
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    Task.COUNTER.clear()
    settings_path, simulation_path = class_paths(args.algorithm)
    settings = import_class(settings_path)(**session_options(args, device))
    # Process every event at the exact final budget even when another event has
    # the same timestamp. No evaluation includes work past the declared budget.
    settings.duration += 1
    simulation = import_class(simulation_path)(settings)
    evaluation_model = document.get("evaluation_model", "local")
    network = configure_simulation(simulation, document["times"], document["network"], evaluation_model)
    simulation.setup_directories()
    simulation.simulate()
    simulation.workflow_dag.build_task_indices()
    simulation.workflow_dag.check_validity()
    write_json(Path(document["output"]) / "network.json", network)
    write_json(Path(document["output"]) / "execution.json", dict(artifacts=simulation.data_dir,
               evaluation_model=evaluation_model, aggregation_weighting="uniform" if evaluation_model == "global" else None,
               settings=settings.to_dict()))
    execute_local(simulation, retain_results=False)
    # Keep live readers' per-peer files stable. The ordinary merge deletes them
    # while writing the combined file, which races a polling reader.
    merged = Path(simulation.data_dir) / "accuracies.csv"
    temporary = merged.with_suffix(".csv.tmp")
    with temporary.open("w") as target:
        for index, source in enumerate(sorted(Path(simulation.data_dir).glob("accuracies_*.csv"))):
            lines = source.read_text().splitlines(keepends=True)
            target.writelines(lines if index == 0 else lines[1:])
    temporary.replace(merged)
    write_json(Path(document["output"]) / "training-executor.json", simulation.execution_training_stats)


def read_curve(directory, peers, times, *, strict=False, evaluation_model="local"):
    """Read closed CSV rows; publish complete checkpoints for the selected readout."""
    import json
    directory = Path(directory)
    network_path = directory / "network.json"
    if not network_path.exists():
        if strict:
            raise ValueError("Missing network measurements")
        return []
    measurements = json.loads(network_path.read_text())
    network = {row["time"]: row["bytes_sent"] for row in measurements}
    if (len(network) != len(measurements) or sorted(network) != times
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in network.values())
            or any(network[b] < network[a] for a, b in zip(times, times[1:]))):
        raise ValueError("Invalid communication checkpoints")
    checkpoints = {}
    merged = sorted(directory.glob("data/*/accuracies.csv"))
    paths = merged or sorted(directory.glob("data/*/accuracies_*.csv"))
    for path in paths:
        source = path.read_text()
        # A writer may be between opening and closing its append operation.
        source = source[:source.rfind("\n") + 1]
        for row in csv.DictReader(io.StringIO(source)):
            stamp, peer = float(row["time"]), int(row["peer"])
            accuracy, loss = float(row["accuracy"]), float(row["loss"])
            valid_peer = peer == -1 if evaluation_model == "global" else 0 <= peer < peers
            if (stamp not in times or not valid_peer or not math.isfinite(loss)
                    or not math.isfinite(accuracy) or not 0 <= accuracy <= 100):
                raise ValueError("Invalid benchmark evaluation row")
            observations = checkpoints.setdefault(stamp, {})
            if peer in observations:
                raise ValueError("Duplicate peer at a checkpoint")
            observations[peer] = accuracy, loss
    curve = []
    for stamp in times:
        rows = checkpoints.get(stamp, {})
        if len(rows) != (1 if evaluation_model == "global" else peers):
            break
        if stamp not in network:
            raise ValueError("Missing communication checkpoint")
        values = sorted(accuracy for accuracy, _ in rows.values())
        curve.append(dict(time=stamp, accuracy=statistics.mean(values),
                          loss=statistics.mean(loss for _, loss in rows.values()),
                          worst_decile_accuracy=None if evaluation_model == "global" else statistics.mean(values[:max(1, math.ceil(peers / 10))]),
                          bytes_sent=network[stamp], peers=peers,
                          evaluated_models=1 if evaluation_model == "global" else peers))
    if strict and len(curve) != len(times):
        raise ValueError(f"Incomplete checkpoints: {len(curve)}/{len(times)}")
    return curve


def budget_metrics(curve, budget, targets):
    eligible = [point for point in curve if point["time"] <= budget]
    if not eligible or eligible[0]["time"] != 0 or eligible[-1]["time"] != budget:
        return None
    # Left-constant integration: later observations never improve earlier time.
    area = sum(a["accuracy"] * (b["time"] - a["time"]) for a, b in zip(eligible, eligible[1:])) / budget
    final = eligible[-1]
    return dict(accuracy=final["accuracy"], loss=final["loss"], average_accuracy=area,
                worst_decile_accuracy=final["worst_decile_accuracy"], bytes_sent=final["bytes_sent"],
                best_observed_accuracy=max(point["accuracy"] for point in eligible),
                time_to_accuracy={str(target): next((point["time"] for point in eligible if point["accuracy"] >= target), None)
                                  for target in targets})


def _estimate(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values) > 1 else None)


def leaderboard(suite):
    variants = list(dict.fromkeys(run["variant"] for run in suite["runs"]))
    tables = []
    for budget in suite["budgets"]:
        rows = []
        for variant in variants:
            runs = [run for run in suite["runs"] if run["variant"] == variant]
            observed = []
            for run in runs:
                if run["status"] not in {"running", "completed"}:
                    continue
                metrics = budget_metrics(run.get("curve", []), budget, suite["accuracy_targets"])
                if metrics is not None:
                    observed.append((run, metrics))
            row = dict(variant=variant, rank=None, seeds=[run["seed"] for run, _ in observed],
                       learning_rate=runs[0]["options"]["learning_rate"],
                       expected_seeds=len(suite["seeds"]), complete=len(observed) == len(suite["seeds"]),
                       failed_runs=sum(run["status"] in {"failed", "interrupted"} for run in runs))
            for field in ("accuracy", "loss", "average_accuracy", "worst_decile_accuracy", "bytes_sent", "best_observed_accuracy"):
                values = [metrics[field] for _, metrics in observed if metrics[field] is not None]
                row[field] = _estimate(values) if values else None
            row["execution_wall_seconds"] = _estimate([run["wall_seconds"] for run, _ in observed
                if run["status"] == "completed"]) if any(run["status"] == "completed" for run, _ in observed) else None
            row["time_to_accuracy"] = {}
            for target in suite["accuracy_targets"]:
                reached = [metrics["time_to_accuracy"][str(target)] for _, metrics in observed
                           if metrics["time_to_accuracy"][str(target)] is not None]
                row["time_to_accuracy"][str(target)] = dict(reached=len(reached), evaluated=len(observed),
                    mean_seconds=statistics.mean(reached) if reached else None, censor_seconds=budget)
            rows.append(row)
        rows.sort(key=lambda row: -(row["accuracy"]["mean"] if row["accuracy"] else -1))
        # Official ranks require the entire paired seed set for every variant.
        ranked = all(row["complete"] for row in rows)
        if ranked:
            for index, row in enumerate(rows):
                row["rank"] = (rows[index - 1]["rank"] if index and row["accuracy"]["mean"] == rows[index - 1]["accuracy"]["mean"] else index + 1)
        tables.append(dict(budget=budget, provisional=not ranked, rows=rows))
    return dict(version=1, name=suite["name"], status=suite["status"], time_basis="simulated_seconds",
                updated_at=datetime.now(timezone.utc).isoformat(), budgets=suite["budgets"],
                accuracy_targets=suite["accuracy_targets"], network=suite["network"],
                evaluation_split=suite["runs"][0]["options"]["evaluation_split"],
                scenario={key: suite["runs"][0]["options"][key] for key in ("dataset", "model", "peers", "partitioner", "alpha")},
                provenance=suite.get("provenance"), selection=suite.get("selection"),
                evaluation_model=suite.get("evaluation_model", "local"), execution_devices=suite.get("execution_devices"),
                tables=tables, runs=suite["runs"])


def save_results(output, suite):
    result = leaderboard(suite)
    write_json(output / "manifest.json", suite)
    write_json(output / "leaderboard.json", result)
    columns = ["budget", "variant", "rank", "provisional", "seeds", "accuracy_mean", "accuracy_sd",
               "average_accuracy", "worst_decile_accuracy", "bytes_sent", "best_observed_accuracy", "execution_wall_seconds", "learning_rate"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for table in result["tables"]:
        for row in table["rows"]:
            record = dict(budget=table["budget"], variant=row["variant"], rank=row["rank"],
                          provisional=table["provisional"], seeds=len(row["seeds"]),
                          accuracy_mean=row["accuracy"]["mean"] if row["accuracy"] else None,
                          accuracy_sd=row["accuracy"]["sd"] if row["accuracy"] else None,
                          learning_rate=row["learning_rate"])
            for key in columns[7:-1]:
                record[key] = row[key]["mean"] if row[key] else None
            writer.writerow(record)
    temporary = output / "leaderboard.csv.tmp"
    temporary.write_text(buffer.getvalue())
    temporary.replace(output / "leaderboard.csv")
    return result


def execute_benchmark(benchmark, *, poll_seconds=1, on_update=None):
    """Schedule isolated workers; one process slot per declared execution device."""
    root = Path(__file__).resolve().parents[1]
    try:
        benchmark.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"Output directory already exists: {benchmark.output}; choose a fresh --output") from exc
    (benchmark.output / "runs").mkdir()
    (benchmark.output / ("source" + benchmark.path.suffix)).write_text(benchmark.path.read_text())
    suite = benchmark.manifest() | dict(status="running", provenance=provenance(root))
    def publish():
        result = save_results(benchmark.output, suite)
        if on_update is not None:
            on_update(result)
    for run in suite["runs"]:
        run.update(status="planned", curve=[])
    publish()
    devices = benchmark.execution_devices or [None]
    active, next_run, interrupted = {}, 0, False

    def finish(slot, status, error=None):
        job = active.pop(slot)
        record = job["record"]
        if error is not None:
            record["error"] = error
        record.update(status=status, wall_seconds=time.monotonic() - job["started"],
                      finished_at=datetime.now(timezone.utc).isoformat())
        if job["log"] is not None:
            job["log"].close()
        write_json(job["directory"] / "manifest.json", record)

    try:
        while active or next_run < len(suite["runs"]):
            for slot, device in enumerate(devices):
                if slot in active or next_run == len(suite["runs"]):
                    continue
                record = suite["runs"][next_run]
                next_run += 1
                directory = benchmark.output / "runs" / record["name"]
                directory.mkdir()
                if device is not None:
                    record["options"]["device"] = device
                record.update(status="running", started_at=datetime.now(timezone.utc).isoformat(), executor_slot=slot)
                job = dict(record=record, directory=directory, process=None, log=None, started=time.monotonic())
                active[slot] = job
                resolved = dict(options=record["options"], times=benchmark.times, evaluation_model=benchmark.evaluation_model,
                                network=benchmark.network, output=str(directory))
                try:
                    write_json(directory / "resolved.json", resolved)
                    job["log"] = (directory / "run.log").open("w")
                    job["process"] = subprocess.Popen([sys.executable, str(root / "scripts/run_benchmark.py"),
                        "--worker", str(directory / "resolved.json")], cwd=root, stdout=job["log"],
                        stderr=subprocess.STDOUT, start_new_session=True)
                    print(f"[{next_run}/{len(suite['runs'])}] {record['name']} on {record['options']['device']} (slot {slot})", flush=True)
                except OSError as exc:
                    finish(slot, "failed", str(exc))
            for slot, job in list(active.items()):
                process, record = job["process"], job["record"]
                try:
                    code = process.poll()
                    if code is not None:
                        record["returncode"] = code
                        if code != 0:
                            raise ValueError(f"Worker exited {code}; see {job['directory'] / 'run.log'}")
                    record["curve"] = read_curve(job["directory"], record["options"]["peers"], benchmark.times,
                        strict=code == 0, evaluation_model=benchmark.evaluation_model)
                    record["wall_seconds"] = time.monotonic() - job["started"]
                    if code is not None:
                        finish(slot, "completed")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    _stop(process)
                    finish(slot, "failed", str(exc))
            publish()
            if active:
                try:
                    next(iter(active.values()))["process"].wait(timeout=poll_seconds)
                except subprocess.TimeoutExpired:
                    pass
    except KeyboardInterrupt:
        interrupted = True
        for slot, job in list(active.items()):
            if job["process"] is not None:
                _stop(job["process"])
            finish(slot, "interrupted", "Benchmark interrupted")
    finally:
        # Unexpected parent errors must not leave any worker process behind.
        for job in active.values():
            if job["process"] is not None:
                _stop(job["process"])
            if job["log"] is not None:
                job["log"].close()
    suite["status"] = "interrupted" if interrupted else ("failed" if any(run["status"] == "failed" for run in suite["runs"]) else "completed")
    publish()
    return 130 if interrupted else int(suite["status"] != "completed")
