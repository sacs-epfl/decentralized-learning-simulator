import json
from pathlib import Path
import subprocess
import sys

import pytest

from args import create_parser
from dasklearn.benchmark import load_benchmark
from dasklearn.lr_tuning import execute_tuned_benchmark, select_learning_rates, tuning_benchmark

ROOT = Path(__file__).resolve().parents[1]


def configuration(tmp_path, **changes):
    document = dict(version=1, name="tuned-test", output="results",
        benchmark=dict(budgets=[1, 2], seeds=[10], evaluation_period=1, accuracy_targets=[0, 100]),
        settings=dict(dataset="synthetic", model="mlp", peers=4, device="cpu", evaluation_split="test"),
        runs=[dict(name="a", settings=dict(algorithm="dpsgd", topology="ring")),
              dict(name="b", settings=dict(algorithm="qgm"))],
        tuning=dict(budget=2, seeds=[0], evaluation_examples=100,
                    learning_rates=dict(a=[.01, .1], b=[.001, .01])))
    document.update(changes)
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps(document))
    return path


def synthetic_result(benchmark):
    trials = tuning_benchmark(benchmark)
    result = dict(status="completed", runs=[])
    for run in trials.runs:
        # The second candidate wins, except algorithm b is tied to test the rule.
        accuracy = 70 if "lr02" in run.variant or run.variant.startswith("b-") else 60
        curve = [dict(time=0, accuracy=10, loss=2, worst_decile_accuracy=10, bytes_sent=0),
                 dict(time=2, accuracy=accuracy, loss=1, worst_decile_accuracy=accuracy, bytes_sent=100)]
        result["runs"].append(dict(name=run.name, variant=run.variant, seed=run.seed,
                                   options=run.options, status="completed", curve=curve))
    return result


def test_equal_budget_plan_has_five_representatives_and_no_data_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Planning loaded model/data")
    monkeypatch.setattr("dasklearn.datasets.create_dataset", forbidden)
    monkeypatch.setattr("dasklearn.models.create_model", forbidden)
    benchmark = load_benchmark(ROOT / "experiments/leaderboard-cifar10-32.yaml", create_parser())
    trials = tuning_benchmark(benchmark)
    assert len(trials.runs) == 25 and len(benchmark.runs) == 15
    assert benchmark.tuning["trials_per_algorithm"] == 5
    assert benchmark.tuning["simulated_seconds_per_algorithm"] == 1500
    assert {run.options["algorithm"] for run in trials.runs} == {"fl", "dpsgd", "qgm", "choco", "demo"}
    assert all(run.options["evaluation_split"] == "validation" and run.seed == 0 for run in trials.runs)
    assert all(run.options["evaluation_examples"] == 1000 and run.options["duration"] == 300 for run in trials.runs)
    assert all(run.options["dataset"] == "cifar10" and run.options["model"] == "gnlenet"
               and run.options["peers"] == 32 for run in trials.runs)
    assert all(run.options["evaluation_split"] == "test" and run.seed in {10, 11, 12} for run in benchmark.runs)
    assert trials.times == [0, 300]


@pytest.mark.parametrize("patch,match", [
    (dict(learning_rates=dict(a=[.01], b=[.001, .01])), "same number"),
    (dict(learning_rates=dict(a=[.01], missing=[.01])), "exactly"),
    (dict(learning_rates=dict(a=[.01, .01], b=[.01, .1])), "duplicates"),
    (dict(learning_rates=dict(a=[0], b=[.1])), "positive"),
    (dict(seeds=[10]), "disjoint"),
    (dict(seeds=[]), "nonempty"),
    (dict(budget=0), "positive"),
    (dict(evaluation_period=0), "positive"),
    (dict(evaluation_examples=-1), "nonnegative"),
    (dict(metric="test_accuracy"), "metric"),
    (dict(metric="average_accuracy", evaluation_period=2), "intermediate"),
    (dict(typo=True), "mapping"),
])
def test_invalid_tuning_is_rejected_before_any_output(tmp_path, patch, match):
    path = configuration(tmp_path)
    document = json.loads(path.read_text())
    document["tuning"].update(patch)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=match):
        load_benchmark(path, create_parser())
    assert not (tmp_path / "results").exists()


def test_selection_uses_complete_validation_grid_and_lower_lr_breaks_ties(tmp_path):
    benchmark = load_benchmark(configuration(tmp_path), create_parser())
    result = synthetic_result(benchmark)
    selected = select_learning_rates(benchmark, result)
    assert selected["selected"] == dict(a=.1, b=.001)
    assert selected["selection_split"] == "validation"
    assert selected["evaluation_seeds"] == [10]
    result["runs"][0]["options"]["evaluation_split"] = "test"
    with pytest.raises(ValueError, match="validation-only"):
        select_learning_rates(benchmark, result)


def test_selection_averages_all_tuning_seeds_and_ignores_early_peaks(tmp_path):
    path = configuration(tmp_path)
    document = json.loads(path.read_text())
    document["tuning"]["seeds"] = [0, 1]
    path.write_text(json.dumps(document))
    benchmark = load_benchmark(path, create_parser())
    result = synthetic_result(benchmark)
    for record in result["runs"]:
        if record["variant"] == "a-lr01":
            record["curve"][0]["accuracy"] = 100
            record["curve"][-1]["accuracy"] = 99 if record["seed"] == 0 else 1
        elif record["variant"] == "a-lr02":
            record["curve"][-1]["accuracy"] = 60
    selected = select_learning_rates(benchmark, result)
    assert selected["selected"]["a"] == .1  # Mean 60 beats mean 50, despite seed 0's 99.
    assert [row["mean_score"] for row in selected["scores"]["a"]] == [50, 60]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed", "wrong_seed", "wrong_budget", "missing_checkpoint"])
def test_incomplete_or_mismatched_trials_cannot_win(tmp_path, mutation):
    benchmark = load_benchmark(configuration(tmp_path), create_parser())
    result = synthetic_result(benchmark)
    if mutation == "missing":
        result["runs"].pop()
    elif mutation == "duplicate":
        result["runs"][-1] = result["runs"][0]
    elif mutation == "failed":
        result["runs"][0]["status"] = "failed"
    elif mutation == "wrong_seed":
        result["runs"][0]["options"]["seed"] = 5
    elif mutation == "wrong_budget":
        result["runs"][0]["options"]["duration"] = 1
    else:
        result["runs"][0]["curve"].pop()
    with pytest.raises(ValueError):
        select_learning_rates(benchmark, result)


def test_variant_subset_and_final_seeds_override_keep_pairing(tmp_path):
    benchmark = load_benchmark(configuration(tmp_path), create_parser(), variants=["b"], seeds=[11, 12])
    assert benchmark.tuning["learning_rates"] == dict(b=[.001, .01])
    assert len(tuning_benchmark(benchmark).runs) == 2
    assert [run.seed for run in benchmark.runs] == [11, 12]


def test_failed_search_cannot_start_final_evaluation(tmp_path, monkeypatch):
    benchmark = load_benchmark(configuration(tmp_path), create_parser())
    calls = []
    def fail(stage, **kwargs):
        calls.append(stage)
        return 1
    monkeypatch.setattr("dasklearn.lr_tuning.execute_benchmark", fail)
    assert execute_tuned_benchmark(benchmark) == 1
    assert len(calls) == 1
    state = json.loads((benchmark.output / "pipeline.json").read_text())
    assert state["phase"] == "tuning" and state["status"] == "failed"
    assert not (benchmark.output / "selection.json").exists()
    assert not (benchmark.output / "evaluation").exists()
    with pytest.raises(ValueError, match="already exists"):
        execute_tuned_benchmark(benchmark)


def test_real_pipeline_freezes_lr_then_evaluates_disjoint_seeds_on_test(tmp_path):
    path = configuration(tmp_path)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_benchmark.py"), str(path)],
                            cwd=ROOT, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    root = tmp_path / "results"
    state = json.loads((root / "pipeline.json").read_text())
    report = json.loads((root / "leaderboard.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    benchmark = load_benchmark(path, create_parser())
    trials = json.loads((root / "tuning" / "leaderboard.json").read_text())
    assert selection == select_learning_rates(benchmark, trials)
    assert state["status"] == "completed" and state["phase"] == "evaluation"
    assert len(trials["runs"]) == 4 and len(report["runs"]) == 2
    assert all(not table["provisional"] for table in report["tables"])
    for run in report["runs"]:
        assert run["seed"] == 10 and run["options"]["evaluation_split"] == "test"
        assert run["options"]["learning_rate"] == selection["selected"][run["variant"]]
    assert report["pipeline"]["selected"] == selection["selected"]
    assert (root / "leaderboard.csv").read_bytes() == (root / "evaluation" / "leaderboard.csv").read_bytes()
