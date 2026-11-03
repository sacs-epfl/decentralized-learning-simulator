import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from args import create_parser
from dasklearn.benchmark import configure_simulation, execute_benchmark, load_benchmark, read_curve

ROOT = Path(__file__).resolve().parents[1]


def configuration(tmp_path, *, devices=None, seeds=None):
    document = dict(version=1, name="global-test", output="results",
        benchmark=dict(budgets=[1, 3], seeds=seeds or [0, 1], evaluation_period=1,
                       evaluation_model="global", accuracy_targets=[0, 100]),
        settings=dict(dataset="synthetic", model="mlp", peers=4, device="cpu", evaluation_split="test"),
        runs=[dict(name="dpsgd", settings=dict(algorithm="dpsgd", topology="ring"))])
    if devices is not None:
        document["benchmark"]["execution_devices"] = devices
    path = tmp_path / "global.json"
    path.write_text(json.dumps(document))
    return path


def test_global900_plan_preserves_selected_lrs_and_uses_four_gpu_slots():
    benchmark = load_benchmark(ROOT / "experiments/leaderboard-cifar10-32-global900.yaml", create_parser())
    assert benchmark.evaluation_model == "global" and benchmark.tuning is None
    assert benchmark.budgets == [60, 300, 900] and len(benchmark.runs) == 15
    assert benchmark.execution_devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert {run.variant: run.options["learning_rate"] for run in benchmark.runs} == {
        "fedavg": .01, "dpsgd-ring": .01, "qgm": .01, "choco": .1, "demo": .001}
    assert {run.seed for run in benchmark.runs} == {10, 11, 12}


@pytest.mark.parametrize("field,value", [("evaluation_model", "unknown"), ("evaluation_model", []),
    ("execution_devices", []), ("execution_devices", ["cuda:no"]), ("execution_devices", [True])])
def test_invalid_readout_or_device_slots_are_rejected(tmp_path, field, value):
    path = configuration(tmp_path)
    document = json.loads(path.read_text())
    document["benchmark"][field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_benchmark(path, create_parser())


@pytest.mark.asyncio
async def test_global_snapshot_averages_parameters_then_evaluates_once_without_mutating_inputs(tmp_path, monkeypatch):
    from dasklearn import functions
    from dasklearn.datasets.synthetic import TensorPartition
    from dasklearn.model_evaluator import ModelEvaluator
    from dasklearn.session_settings import SessionSettings, LearningSettings
    from dasklearn.simulation.client import BaseClient
    from dasklearn.simulation.events import Event, TEST_SNAPSHOT
    from dasklearn.simulation.simulation import Simulation
    from dasklearn.tasks.local import execute_local
    from dasklearn.tasks.references import TaskRef

    settings = SessionSettings(algorithm="test", seed=0, work_dir=str(tmp_path), dataset="synthetic", model="mlp",
        participants=2, learning=LearningSettings(.1, 0, 0, 2, 1), evaluation_split="test", torch_threads=1)
    simulation = Simulation(settings)
    simulation.clients = [BaseClient(simulation, peer) for peer in range(2)]
    simulation.setup_directories()
    originals, evaluated = [], []

    def initialize(settings, params):
        model = torch.nn.Linear(16, 4)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
            model.bias[0] = 2 if params["peer"] == 0 else -1
        originals.append(model)
        return [model]

    partition = TensorPartition(torch.ones(2, 16), torch.zeros(2, dtype=torch.long))
    monkeypatch.setattr(functions, "initialize", initialize)
    monkeypatch.setattr(functions, "create_dataset", lambda settings: SimpleNamespace(load_split=lambda split: partition))
    evaluate = ModelEvaluator.evaluate_accuracy

    def observed(self, model, *args, **kwargs):
        evaluated.append(model.bias.detach().clone())
        return evaluate(self, model, *args, **kwargs)

    monkeypatch.setattr(ModelEvaluator, "evaluate_accuracy", observed)
    checkpoints = {}
    for peer, client in enumerate(simulation.clients):
        initial = client.create_compute_task("initialize", dict(peer=peer, round=0, time=0))
        checkpoints[peer] = (0, TaskRef(initial.name, 0))
    simulation.latest_models = dict(checkpoints)
    configure_simulation(simulation, [0], dict(profile="uniform", compute_step_ms=20,
                         bandwidth_bytes_per_second=1000000), "global")
    simulation.test_snapshot(Event(0, 0, TEST_SNAPSHOT, is_global=True))
    tasks = list(simulation.workflow_dag.tasks.values())
    assert sum(task.func == "test" for task in tasks) == 1
    average = next(task for task in tasks if task.func == "aggregate")
    assert average.data["weights"] == [.5, .5] and len(average.data["models"]) == 2
    simulation.workflow_dag.build_task_indices()
    simulation.workflow_dag.check_validity()
    execute_local(simulation, retain_results=False)
    assert len(evaluated) == 1 and evaluated[0][0].item() == .5
    rows = list(csv.DictReader((Path(simulation.data_dir) / "accuracies_-1.csv").open()))
    # One input model scores 100%, the other 0%; the averaged model scores 100%,
    # which distinguishes this readout from averaging individual accuracies (50%).
    assert len(rows) == 1 and float(rows[0]["accuracy"]) == 100
    assert [model.bias[0].item() for model in originals] == [2, -1]
    assert simulation.latest_models == checkpoints


def test_parallel_scheduler_bounds_slots_and_records_assignments(tmp_path, monkeypatch):
    import dasklearn.benchmark as module
    benchmark = load_benchmark(configuration(tmp_path, devices=["cpu", "cpu"], seeds=[0, 1, 2, 3, 4]), create_parser())
    active, peak, assignments = 0, 0, []

    class Process:
        def __init__(self, command, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            self.polls = 0
            self.done = False
            assignments.append(json.loads(Path(command[-1]).read_text())["options"]["device"])

        def poll(self):
            nonlocal active
            self.polls += 1
            if self.polls >= 2 and not self.done:
                active -= 1
                self.done = True
            return 0 if self.done else None

        def wait(self, **kwargs):
            return None

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module, "provenance", lambda root: {})
    monkeypatch.setattr(module, "read_curve", lambda *a, **kw: [dict(time=t, accuracy=50, loss=1,
        worst_decile_accuracy=None, bytes_sent=t*10, peers=4, evaluated_models=1) for t in benchmark.times])
    assert execute_benchmark(benchmark) == 0
    assert peak == 2 and active == 0 and assignments == ["cpu"] * 5
    result = json.loads((benchmark.output / "leaderboard.json").read_text())
    assert {record["executor_slot"] for record in result["runs"]} == {0, 1}


def test_interrupt_stops_every_active_worker_and_leaves_remaining_planned(tmp_path, monkeypatch):
    import dasklearn.benchmark as module
    benchmark = load_benchmark(configuration(tmp_path, devices=["cpu", "cpu"], seeds=[0, 1, 2]), create_parser())
    processes, stopped = [], []

    class Process:
        def __init__(self, *args, **kwargs):
            processes.append(self)

        def poll(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module, "_stop", lambda process: stopped.append(process))
    monkeypatch.setattr(module, "provenance", lambda root: {})
    assert execute_benchmark(benchmark) == 130
    assert stopped == processes and len(stopped) == 2
    result = json.loads((benchmark.output / "leaderboard.json").read_text())
    assert [run["status"] for run in result["runs"]] == ["interrupted", "interrupted", "planned"]


def test_real_global_parallel_run_has_one_score_per_checkpoint(tmp_path):
    path = configuration(tmp_path, devices=["cpu", "cpu"])
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_benchmark.py"), str(path)],
                            cwd=ROOT, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    root = tmp_path / "results"
    report = json.loads((root / "leaderboard.json").read_text())
    assert report["status"] == "completed" and report["evaluation_model"] == "global"
    assert report["tables"][-1]["rows"][0]["worst_decile_accuracy"] is None
    for run in report["runs"]:
        assert len(run["curve"]) == 4
        assert all(point["evaluated_models"] == 1 and point["peers"] == 4 for point in run["curve"])
        rows = list(csv.DictReader(next((root / "runs" / run["name"]).glob("data/*/accuracies.csv")).open()))
        assert len(rows) == 4 and all(int(row["peer"]) == -1 for row in rows)
        assert run["curve"][-1]["bytes_sent"] > 0
    assert report["tables"][-1]["provisional"] is False
