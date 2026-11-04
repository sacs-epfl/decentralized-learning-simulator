from copy import deepcopy
from dataclasses import asdict
import csv
import json
from pathlib import Path

import pytest

from args import create_parser
from dasklearn.benchmark import load_benchmark, save_results
from dasklearn.leaderboard_extension import execute_extension, load_baselines, reusable_trials
from dasklearn.lr_tuning import tuning_benchmark


def completed(benchmark):
    suite = benchmark.manifest() | dict(status="completed")
    for run in suite["runs"]:
        run.update(status="completed", wall_seconds=1, curve=[dict(time=time,
            accuracy=20 + run["options"]["learning_rate"] * 100, loss=1,
            worst_decile_accuracy=None, bytes_sent=time * 100, peers=2,
            evaluated_models=1) for time in benchmark.times])
    return suite


@pytest.fixture
def plan(tmp_path):
    document = dict(version=1, name="additions", output="extended",
        benchmark=dict(budgets=[1, 2], seeds=[10, 11], evaluation_period=1,
                       evaluation_model="global", accuracy_targets=[0, 100]),
        settings=dict(dataset="synthetic", model="mlp", peers=2, device="cpu", evaluation_split="test"),
        runs=[dict(name="swift", settings=dict(algorithm="swift")),
              dict(name="gt-dsgd", settings=dict(algorithm="gt-dsgd"))],
        tuning=dict(budget=2, seeds=[0], evaluation_examples=32,
                    learning_rates={"swift": [.01, .02], "gt-dsgd": [.01, .02]}))
    path = tmp_path / "additions.json"
    path.write_text(json.dumps(document))
    benchmark = load_benchmark(path, create_parser())
    settings = []
    for identifier, partitioner, profile in [("iid-uniform", "iid", "uniform"),
            ("iid-heterogeneous", "iid", "heterogeneous"),
            ("dirichlet-uniform", "dirichlet", "uniform"),
            ("dirichlet-heterogeneous", "dirichlet", "heterogeneous")]:
        original = deepcopy(document)
        original.pop("tuning")
        original["runs"] = [dict(name="old", settings=dict(algorithm="dpsgd"))]
        original["settings"].update(partitioner=partitioner, alpha=.1)
        original["benchmark"]["network"] = dict(profile=profile)
        original["output"] = identifier
        source = tmp_path / (identifier + ".json")
        source.write_text(json.dumps(original))
        baseline = load_benchmark(source, create_parser())
        baseline.output.mkdir()
        save_results(baseline.output, completed(baseline))
        settings.append(dict(id=identifier, label=identifier, output=identifier))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(dict(version=1, default=settings[0]["id"], settings=settings)))
    return benchmark, catalog


def fake_executor(benchmark, *, on_update, **kwargs):
    benchmark.output.mkdir(parents=True)
    result = save_results(benchmark.output, completed(benchmark))
    on_update(result)
    return 0


def test_extension_freezes_validation_winners_and_preserves_original_runs(plan, monkeypatch):
    benchmark, catalog = plan
    settings = load_baselines(benchmark, catalog)
    before = {entry["id"]: (entry["output"] / "manifest.json").read_bytes() for entry in settings}
    calls = []

    def execute(current, *, on_update, **kwargs):
        calls.append(current)
        if current.tuning is None and current.runs[0].options["evaluation_split"] == "validation":
            for entry in settings:
                initial = json.loads((benchmark.output / "settings" / entry["id"] / "leaderboard.json").read_text())
                new = [run for run in initial["runs"] if run["variant"] != "old"]
                assert all(run["status"] == "planned" and run["options"]["learning_rate"] is None for run in new)
                assert all(table["provisional"] for table in initial["tables"])
        return fake_executor(current, on_update=on_update)

    monkeypatch.setattr("dasklearn.leaderboard_extension.execute_benchmark", execute)
    assert execute_extension(benchmark, settings, baseline_catalog=catalog) == 0
    assert len(calls) == 5
    assert len(calls[0].runs) == 4 and calls[0].seeds == [0]
    assert all(run.options["evaluation_split"] == "validation" for run in calls[0].runs)
    selection = json.loads((benchmark.output / "selection.json").read_text())
    assert selection["selected"] == {"swift": .02, "gt-dsgd": .02}
    for entry, final in zip(settings, calls[1:]):
        assert final.seeds == [10, 11]
        assert final.network == entry["suite"]["network"]
        assert all(run.options["partitioner"] == entry["scenario"]["partitioner"]
                   and run.options["learning_rate"] == .02 and run.options["evaluation_split"] == "test"
                   for run in final.runs)
        assert (entry["output"] / "manifest.json").read_bytes() == before[entry["id"]]
        result = json.loads((benchmark.output / "settings" / entry["id"] / "leaderboard.json").read_text())
        assert result["status"] == "completed" and len(result["runs"]) == 6
        assert all(not table["provisional"] for table in result["tables"])
        assert result["runs"][:2] == [run | dict(source_output=str(entry["output"])) for run in entry["suite"]["runs"]]
        assert result["extension"]["selected"] == selection["selected"]


@pytest.mark.parametrize("code,status", [(1, "failed"), (130, "interrupted")])
def test_failed_or_interrupted_tuning_never_starts_final_evaluation(plan, monkeypatch, code, status):
    benchmark, catalog = plan
    settings = load_baselines(benchmark, catalog)
    calls = []
    def fail(current, *, on_update, **kwargs):
        calls.append(current)
        return code
    monkeypatch.setattr("dasklearn.leaderboard_extension.execute_benchmark", fail)
    assert execute_extension(benchmark, settings, baseline_catalog=catalog) == code
    assert len(calls) == 1 and not (benchmark.output / "selection.json").exists()
    pipeline = json.loads((benchmark.output / "pipeline.json").read_text())
    assert pipeline["status"] == status
    for entry in settings:
        result = json.loads((benchmark.output / "settings" / entry["id"] / "leaderboard.json").read_text())
        assert result["status"] == status
        assert all(run["status"] == "interrupted" for run in result["runs"] if run["variant"] != "old")
        assert all(table["provisional"] for table in result["tables"])


@pytest.mark.parametrize("mutation", ["seeds", "readout", "incomplete", "dataset", "duplicate", "capacity"])
def test_incompatible_baselines_are_rejected_before_output(plan, mutation):
    benchmark, catalog = plan
    path = catalog.parent / "iid-uniform" / "manifest.json"
    suite = json.loads(path.read_text())
    if mutation == "seeds": suite["seeds"] = [99]
    elif mutation == "readout": suite["evaluation_model"] = "local"
    elif mutation == "incomplete": suite["runs"][0]["curve"].pop()
    elif mutation == "dataset": suite["runs"][0]["options"]["dataset"] = "digits"
    elif mutation == "duplicate": suite["runs"][0]["variant"] = "swift"
    else: suite["network"]["compute_step_ms"] *= 2
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError):
        load_baselines(benchmark, catalog)
    assert not benchmark.output.exists()


def test_real_additions_have_equal_tuning_budget_and_disjoint_final_seeds():
    root = Path(__file__).resolve().parents[1]
    benchmark = load_benchmark(root / "experiments/leaderboard-cifar10-32-additions.yaml", create_parser())
    trials = tuning_benchmark(benchmark)
    assert len(trials.runs) == 25 and len(benchmark.runs) == 15
    assert benchmark.tuning["simulated_seconds_per_algorithm"] == 1500
    assert benchmark.evaluation_model == "global" and benchmark.budgets == [60, 300, 900]
    assert trials.seeds == [0] and benchmark.seeds == [10, 11, 12]
    assert {run.options["algorithm"] for run in trials.runs} == {"swift", "gt-dsgd", "motef", "relaysgd", "damsgrad"}
    assert all(run.options["evaluation_examples"] == 1000 for run in trials.runs)
    assert benchmark.execution_devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]


def previous_trials(trials, path):
    path.mkdir()
    suite = completed(trials)
    suite["status"] = "interrupted"
    for record in suite["runs"]:
        directory = path / "runs" / record["name"]
        data = directory / "data" / "trial"
        data.mkdir(parents=True)
        (directory / "resolved.json").write_text(json.dumps(dict(options=record["options"],
            times=trials.times, network=trials.network, evaluation_model=trials.evaluation_model)))
        (directory / "network.json").write_text(json.dumps([
            {key: point[key] for key in ["time", "bytes_sent"]} for point in record["curve"]]))
        with (data / "accuracies.csv").open("w") as output:
            writer = csv.DictWriter(output, fieldnames=["time", "peer", "accuracy", "loss"])
            writer.writeheader()
            for point in record["curve"]:
                writer.writerow({key: point[key] for key in ["time", "accuracy", "loss"]} | dict(peer=-1))
    (path / "manifest.json").write_text(json.dumps(suite))
    return suite


def test_reuse_preserves_completed_trials_and_only_schedules_remaining_grid(plan, monkeypatch):
    benchmark, catalog = plan
    previous = catalog.parent / "previous-tuning"
    suite = previous_trials(tuning_benchmark(benchmark), previous)
    for record in suite["runs"][2:]:
        record["status"] = "interrupted"
    (previous / "manifest.json").write_text(json.dumps(suite))
    before = (previous / "manifest.json").read_bytes()
    calls = []
    def execute(current, *, on_update, **kwargs):
        calls.append(current)
        return fake_executor(current, on_update=on_update)
    monkeypatch.setattr("dasklearn.leaderboard_extension.execute_benchmark", execute)
    assert execute_extension(benchmark, load_baselines(benchmark, catalog), baseline_catalog=catalog,
                             reuse_tuning=previous) == 0
    assert len(calls) == 5 and len(calls[0].runs) == 2
    assert all(run.name not in {record["name"] for record in suite["runs"][:2]} for run in calls[0].runs)
    full = json.loads((benchmark.output / "tuning" / "manifest.json").read_text())
    assert len(full["runs"]) == 4 and full["status"] == "completed"
    assert sum("reused_from" in record for record in full["runs"]) == 2
    assert (previous / "manifest.json").read_bytes() == before
    selected = json.loads((benchmark.output / "selection.json").read_text())
    assert selected["selected"] == {"swift": .02, "gt-dsgd": .02}


@pytest.mark.parametrize("mutation", ["rate", "network", "raw_settings", "raw_loss", "missing_checkpoint"])
def test_reused_trials_require_matching_protocol_data_and_raw_observations(plan, mutation):
    benchmark, catalog = plan
    trials = tuning_benchmark(benchmark)
    previous = catalog.parent / "previous-tuning"
    suite = previous_trials(trials, previous)
    record = suite["runs"][0]
    directory = previous / "runs" / record["name"]
    if mutation == "rate": record["options"]["learning_rate"] *= 2
    elif mutation == "network": suite["network"]["compute_step_ms"] *= 2
    elif mutation == "raw_settings":
        path = directory / "resolved.json"
        resolved = json.loads(path.read_text())
        resolved["options"]["evaluation_examples"] += 1
        path.write_text(json.dumps(resolved))
    elif mutation == "raw_loss": record["curve"][-1]["loss"] = 999
    else:
        path = directory / "data" / "trial" / "accuracies.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    (previous / "manifest.json").write_text(json.dumps(suite))
    with pytest.raises(ValueError):
        reusable_trials(trials, previous)


def test_sgp_replacement_keeps_other_algorithms_candidates_and_budget():
    root = Path(__file__).resolve().parents[1]
    original = load_benchmark(root / "experiments/leaderboard-cifar10-32-additions.yaml", create_parser())
    current = load_benchmark(root / "experiments/leaderboard-cifar10-32-additions-sgp.yaml", create_parser())
    assert {run.options["algorithm"] for run in current.runs} == {"sgp", "gt-dsgd", "motef", "relaysgd", "damsgrad"}
    assert len(tuning_benchmark(current).runs) == 25 and len(current.runs) == 15
    assert current.tuning["simulated_seconds_per_algorithm"] == 1500
    for variant in ["gt-dsgd", "motef", "relaysgd", "damsgrad"]:
        assert current.tuning["learning_rates"][variant] == original.tuning["learning_rates"][variant]
        assert [run for run in current.runs if run.variant == variant] == [run for run in original.runs if run.variant == variant]


def test_following_live_baselines_refreshes_results_and_waits_for_completion(plan, monkeypatch):
    benchmark, catalog = plan
    output = catalog.parent / "iid-uniform"
    finished = json.loads((output / "manifest.json").read_text())
    active = deepcopy(finished)
    active["status"] = "running"
    active["runs"][0]["status"] = "running"
    active["runs"][0]["curve"].pop()
    save_results(output, active)
    with pytest.raises(ValueError, match="fully completed"):
        load_baselines(benchmark, catalog)
    settings = load_baselines(benchmark, catalog, allow_pending=True)
    monkeypatch.setattr("dasklearn.leaderboard_extension.execute_benchmark", fake_executor)
    waits = []

    def independent_baseline_finishes(seconds):
        waits.append(seconds)
        report = json.loads((benchmark.output / "settings" / "iid-uniform" / "leaderboard.json").read_text())
        assert report["status"] == "running"
        assert report["extension"]["phase"] == "waiting_for_baselines"
        assert all(run["status"] == "completed" for run in report["runs"] if run["variant"] != "old")
        assert any(table["provisional"] for table in report["tables"])
        save_results(output, finished)

    monkeypatch.setattr("dasklearn.leaderboard_extension.time.sleep", independent_baseline_finishes)
    assert execute_extension(benchmark, settings, baseline_catalog=catalog, follow_baselines=True) == 0
    assert waits == [2]
    report = json.loads((benchmark.output / "settings" / "iid-uniform" / "leaderboard.json").read_text())
    assert report["status"] == "completed" and report["runs"][0]["curve"] == finished["runs"][0]["curve"]
    assert all(not table["provisional"] for table in report["tables"])


def test_following_baselines_rejects_failed_and_incomplete_completed_sources(plan):
    benchmark, catalog = plan
    output = catalog.parent / "iid-uniform"
    suite = json.loads((output / "manifest.json").read_text())
    suite["status"] = "failed"
    save_results(output, suite)
    with pytest.raises(ValueError, match="failed or interrupted"):
        load_baselines(benchmark, catalog, allow_pending=True)
    suite["status"] = "completed"
    suite["runs"][0]["curve"].pop()
    save_results(output, suite)
    with pytest.raises(ValueError, match="complete paired checkpoints"):
        load_baselines(benchmark, catalog, allow_pending=True)


def test_extra_three_have_equal_lr_budgets_and_one_local_step():
    root = Path(__file__).resolve().parents[1]
    benchmark = load_benchmark(root / "experiments/leaderboard-cifar10-32-detag-teleportation-moshpit.yaml", create_parser())
    trials = tuning_benchmark(benchmark)
    assert len(trials.runs) == 15 and len(benchmark.runs) == 9
    assert benchmark.tuning["trials_per_algorithm"] == 5
    assert benchmark.tuning["simulated_seconds_per_algorithm"] == 1500
    assert benchmark.evaluation_model == "global" and benchmark.budgets == [60, 300, 900]
    assert benchmark.execution_devices == ["cuda:0", "cuda:1"] * 4
    assert trials.seeds == [0] and benchmark.seeds == [10, 11, 12]
    assert {run.variant for run in trials.runs} == {"detag-lr01", "detag-lr02", "detag-lr03", "detag-lr04", "detag-lr05",
        "teleportation-lr01", "teleportation-lr02", "teleportation-lr03", "teleportation-lr04", "teleportation-lr05",
        "moshpit-lr01", "moshpit-lr02", "moshpit-lr03", "moshpit-lr04", "moshpit-lr05"}
    assert all(run.options["local_steps"] == 1 and run.options["evaluation_examples"] == 1000 for run in trials.runs)
