import csv
from dataclasses import replace
import json
from pathlib import Path

import pytest

from args import create_parser
from dasklearn.benchmark import NonfiniteEvaluation, execute_benchmark, load_benchmark, read_curve
from dasklearn.leaderboard_extension import reusable_trials
from dasklearn.lr_tuning import select_learning_rates, tuning_benchmark
from test_lr_tuning import configuration, synthetic_result


def global_plan(tmp_path):
    path = configuration(tmp_path)
    document = json.loads(path.read_text())
    document["benchmark"]["evaluation_model"] = "global"
    path.write_text(json.dumps(document))
    return load_benchmark(path, create_parser())


def write_trial(directory, trials, record, *, endpoint_loss="nan"):
    data = directory / "data" / "trial"
    data.mkdir(parents=True, exist_ok=True)
    (directory / "network.json").write_text(json.dumps([dict(time=t, bytes_sent=t * 100) for t in trials.times]))
    (directory / "resolved.json").write_text(json.dumps(dict(options=record["options"], times=trials.times,
        network=trials.network, evaluation_model=trials.evaluation_model)))
    with (data / "accuracies.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=["time", "peer", "accuracy", "loss"])
        writer.writeheader()
        writer.writerow(dict(time=0, peer=-1, accuracy=10, loss=2))
        writer.writerow(dict(time=trials.times[-1], peer=-1, accuracy=10, loss=endpoint_loss))


def test_nonfinite_checkpoint_reports_its_time_and_preserves_finite_prefix(tmp_path):
    benchmark = global_plan(tmp_path)
    trials = tuning_benchmark(benchmark)
    record = synthetic_result(benchmark)["runs"][0]
    directory = tmp_path / "trial"
    write_trial(directory, trials, record)
    with pytest.raises(NonfiniteEvaluation) as failure:
        read_curve(directory, 4, trials.times, strict=True, evaluation_model="global")
    assert failure.value.stamp == 2
    assert len(failure.value.curve) == 1 and failure.value.curve[0]["time"] == 0


@pytest.mark.parametrize("mode", ["allowed", "disabled", "test", "early", "missing_prefix", "local", "worker_error"])
def test_only_explicit_exact_budget_global_validation_divergence_is_an_outcome(tmp_path, monkeypatch, mode):
    import dasklearn.benchmark as module
    benchmark = global_plan(tmp_path)
    trials = tuning_benchmark(benchmark)
    if mode == "test":
        trials = replace(trials, runs=[replace(run, options=run.options | dict(evaluation_split="test")) for run in trials.runs])
    if mode == "local":
        trials = replace(trials, evaluation_model="local")

    class Process:
        def __init__(self, *args, **kwargs): pass
        def poll(self): return 1 if mode == "worker_error" else 0

    def nonfinite(*args, **kwargs):
        prefix = [] if mode == "missing_prefix" else [dict(time=0, accuracy=10, loss=2)]
        raise NonfiniteEvaluation(0 if mode == "early" else 2, prefix)

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module, "provenance", lambda root: {})
    monkeypatch.setattr(module, "_stop", lambda process: None)
    monkeypatch.setattr(module, "read_curve", nonfinite)
    code = execute_benchmark(trials, allow_divergent_trials=mode != "disabled")
    assert code == (0 if mode == "allowed" else 1)
    report = json.loads((trials.output / "leaderboard.json").read_text())
    assert all(run["status"] == ("diverged" if mode == "allowed" else "failed") for run in report["runs"])
    if mode == "allowed":
        assert all(run["divergence_time"] == 2 and run["tuning_budget_completed"] for run in report["runs"])


def test_selection_excludes_divergent_candidates_without_replacing_their_lr_budget(tmp_path):
    benchmark = global_plan(tmp_path)
    result = synthetic_result(benchmark) | dict(evaluation_model="global")
    record = result["runs"][1]
    record.update(status="diverged", divergence_time=2, tuning_budget_completed=True, curve=record["curve"][:1])
    selection = select_learning_rates(benchmark, result)
    assert selection["selected"]["a"] == .01
    assert selection["scores"]["a"][1]["mean_score"] is None
    result["runs"][0].update(status="diverged", divergence_time=2, tuning_budget_completed=True)
    with pytest.raises(ValueError, match="No finite LR candidate"):
        select_learning_rates(benchmark, result)


def test_failed_candidate_can_be_reused_only_with_raw_exact_budget_divergence(tmp_path):
    benchmark = global_plan(tmp_path)
    trials = tuning_benchmark(benchmark)
    record = synthetic_result(benchmark)["runs"][0]
    previous = tmp_path / "previous"
    directory = previous / "runs" / record["name"]
    write_trial(directory, trials, record)
    with pytest.raises(NonfiniteEvaluation) as failure:
        read_curve(directory, 4, trials.times, evaluation_model="global")
    record.update(status="failed", curve=failure.value.curve, wall_seconds=42)
    suite = trials.manifest() | dict(status="failed", runs=[record])
    path = previous / "manifest.json"
    path.write_text(json.dumps(suite))
    original = path.read_bytes()
    reused = reusable_trials(trials, previous)
    assert reused[record["name"]]["status"] == "diverged"
    assert reused[record["name"]]["wall_seconds"] == 42
    assert path.read_bytes() == original
    write_trial(directory, trials, record, endpoint_loss="1")
    assert reusable_trials(trials, previous) == {}


def test_seven_algorithm_batch_preserves_equal_budgets_and_local_steps():
    root = Path(__file__).resolve().parents[1]
    benchmark = load_benchmark(root / "experiments/leaderboard-cifar10-32-seven-additions.yaml", create_parser())
    trials = tuning_benchmark(benchmark)
    assert len(trials.runs) == 35 and len(benchmark.runs) == 21
    assert benchmark.tuning["trials_per_algorithm"] == 5
    assert benchmark.tuning["simulated_seconds_per_algorithm"] == 1500
    assert benchmark.evaluation_model == "global" and benchmark.budgets == [60, 300, 900]
    assert benchmark.execution_devices == ["cuda:0", "cuda:1"] * 4
    assert all(run.options["local_steps"] == 1 and run.options["evaluation_split"] == "validation"
               and run.options["evaluation_examples"] == 1000 for run in trials.runs)
    assert {run.options["algorithm"] for run in trials.runs} == {
        "detag", "teleportation", "moshpit", "adpsgd", "epidemic", "gossip", "shatter"}
    assert all(run.seed in {10, 11, 12} and run.options["evaluation_split"] == "test" for run in benchmark.runs)
