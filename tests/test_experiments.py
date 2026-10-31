import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from args import create_parser, get_args
from dasklearn.experiments import read_document, resolve_experiment
from dasklearn.settings_factory import session_options
from dasklearn.algorithms import ALGORITHMS, class_paths, import_class


ROOT = Path(__file__).resolve().parents[1]


def configuration(tmp_path, **changes):
    document = dict(version=1, name="test", output="results", execution="local", settings=dict(
        algorithm="dpsgd", dataset="synthetic", model="mlp", device="cpu", peers=4, topology="ring",
        rounds=2, local_steps=1, batch_size=16, learning_rate=.1, momentum=0, test_interval=1, torch_threads=1))
    document.update(changes)
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document))
    return path


def command(path, *options):
    return [sys.executable, str(ROOT / "scripts/run_experiment.py"), str(path), *options]


def invoke(path, *options):
    # Child processes inherit the test interpreter/environment, but place any
    # optional library caches in the temporary test directory.
    env = dict(os.environ, HF_HOME=str(path.parent / "hf-cache"), MPLCONFIGDIR=str(path.parent / "mpl-cache"))
    return subprocess.run(command(path, *options), cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)


def test_matrix_expansion_cli_precedence_and_zero_seed(tmp_path):
    path = configuration(tmp_path, matrix={"partitioner": ["iid", "dirichlet"], "seed": [0, 42]})
    experiment = resolve_experiment(path, create_parser())
    assert [(run.options["partitioner"], run.options["seed"]) for run in experiment.runs] == [
        ("iid", 0), ("iid", 42), ("dirichlet", 0), ("dirichlet", 42)]
    assert len({run.name for run in experiment.runs}) == 4
    overridden = resolve_experiment(path, create_parser(), ["--seed", "7", "--no-synchronous", "--no-gpu-residency"])
    assert len(overridden.runs) == 2  # A CLI value replaces its matrix axis.
    assert all(run.options["seed"] == 7 and run.options["synchronous"] is False
               and run.options["gpu_residency"] is False for run in overridden.runs)


def test_named_runs_and_relative_file_paths(tmp_path):
    settings = dict(algorithm="dpsgd", dataset="text", rounds=1, dataset_name="json",
                    dataset_data_files={"train": ["data/train.jsonl"], "test": "https://example.com/test.jsonl"},
                    dataset_cache_dir="cache", base_model="org/checkpoint")
    path = configuration(tmp_path, settings=settings, runs=[
        dict(name="iid", settings=dict(partitioner="iid")),
        dict(name="non-iid", settings=dict(partitioner="dirichlet", alpha=.1))])
    experiment = resolve_experiment(path, create_parser())
    assert experiment.output == tmp_path / "results"
    options = experiment.runs[0].options
    assert options["dataset_data_files"]["train"] == [str(tmp_path / "data/train.jsonl")]
    assert options["dataset_data_files"]["test"] == "https://example.com/test.jsonl"
    assert options["dataset_cache_dir"] == str(tmp_path / "cache")
    assert options["base_model"] == "org/checkpoint"
    assert len(experiment.runs) == 2 and experiment.runs[1].options["alpha"] == .1


def test_yaml_types_and_config_cli_overrides(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("""version: 1
execution: local
output: results
settings:
  dataset: synthetic
  rounds: 2
  learning-rate: 0.03
  synchronous: true
  gpu_residency: false
""")
    args = get_args(["--config", str(path), "--learning-rate", ".2", "--no-synchronous"])
    assert args.local and args.learning_rate == .2 and not args.synchronous and not args.gpu_residency
    assert args.work_dir == str(tmp_path / "results")
    simulated = get_args(["--config", str(path), "--dry-run"])
    assert simulated.dry_run and not simulated.local
    assert not get_args(["--config", str(path), "--no-local"]).local


@pytest.mark.parametrize("changes,match", [
    ({"version": 2}, "version"), ({"version": True}, "version"),
    ({"typo": 1}, "Unknown experiment"), ({"settings": {"sead": 1}}, "Unknown setting"),
    ({"settings": {"peers": True}}, "integer"), ({"settings": {"synchronous": "false"}}, "true or false"),
    ({"settings": {"alpha": float("nan")}}, "finite"), ({"settings": {"algorithm": "unknown"}}, "Invalid algorithm"),
    ({"settings": {"rounds": 0}}, "positive rounds"),
    ({"settings": {"work_dir": "elsewhere"}}, "top-level"),
    ({"matrix": {"seed": []}}, "nonempty"), ({"matrix": {"seed": [1, 1]}}, "Duplicate values"),
    ({"matrix": {"seed": [1]}, "runs": []}, "either matrix or runs"),
    ({"runs": [{"name": "../escape"}]}, "names"),
    ({"runs": [{"name": "same"}, {"name": "same"}]}, "unique names"),
])
def test_invalid_configs_are_rejected_before_output_creation(tmp_path, changes, match):
    path = configuration(tmp_path, **changes)
    with pytest.raises(ValueError, match=match):
        resolve_experiment(path, create_parser())
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("name,source", [
    ("duplicate.json", '{"version":1,"version":1}'),
    ("duplicate.yaml", 'version: 1\nversion: 1\n'),
    ("unsafe.yaml", '!!python/object/apply:os.system ["touch SHOULD_NOT_EXIST"]'),
])
def test_duplicate_keys_and_yaml_python_objects_are_rejected(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    with pytest.raises(ValueError):
        read_document(path)


def test_single_run_config_cannot_silently_ignore_matrix(tmp_path):
    path = configuration(tmp_path, matrix={"seed": [0, 1]})
    with pytest.raises(SystemExit) as failure:
        get_args(["--config", str(path)])
    assert failure.value.code == 2


def test_plan_does_not_access_models_datasets_or_output(tmp_path, monkeypatch, capsys):
    from scripts.run_experiment import main
    def forbidden(*args, **kwargs):
        pytest.fail("Planning must not load models or datasets")
    monkeypatch.setattr("dasklearn.datasets.create_dataset", forbidden)
    monkeypatch.setattr("dasklearn.models.create_model", forbidden)
    output = tmp_path / "not-created"
    assert main([str(ROOT / "experiments/food101.yaml"), "--plan", "--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["runs"]) == 2 and plan["runs"][1]["options"]["partitioner"] == "dirichlet"
    assert not output.exists()


def test_femnist_defaults_never_override_explicit_learning_settings(tmp_path):
    path = configuration(tmp_path, settings=dict(dataset="femnist", rounds=1, learning_rate=.03, momentum=.5))
    options = resolve_experiment(path, create_parser()).runs[0].options
    assert (options["learning_rate"], options["momentum"]) == (.03, .5)
    args = get_args(["--dataset", "femnist", "--learning-rate", ".03", "--momentum", ".5"])
    assert (args.learning_rate, args.momentum) == (.03, .5)
    defaults = get_args(["--dataset", "femnist"])
    assert (defaults.learning_rate, defaults.momentum) == (.004, 0)


def test_shared_settings_preserve_algorithm_options_and_seconds():
    qgm = session_options(get_args(["--algorithm", "qgm", "--qgm-beta", ".7", "--qgm-nesterov"]), "cpu")
    assert qgm["beta"] == .7 and qgm["nesterov"]
    gossip = session_options(get_args(["--algorithm", "gossip", "--gl-period", "2"]), "cpu")
    assert gossip["gl_period"] == 2_000_000
    evaluation = session_options(get_args(["--evaluation-examples", "100", "--evaluation-split", "validation",
                                          "--model-size-bytes", "4096", "--release-cuda-cache"]), "cpu")
    assert evaluation["evaluation_examples"] == 100 and evaluation["evaluation_split"] == "validation"
    assert evaluation["model_size_bytes"] == 4096 and evaluation["release_cuda_cache"]


@pytest.mark.parametrize("algorithm", list(ALGORITHMS))
def test_shared_factory_supports_every_registered_algorithm(algorithm):
    args = get_args(["--algorithm", algorithm, "--dataset", "synthetic", "--rounds", "2"])
    settings_type = import_class(class_paths(algorithm)[0])
    settings = settings_type(**session_options(args, "cpu"))
    assert settings.algorithm == algorithm and settings.rounds == 2
    assert settings.learning.batch_size == args.batch_size


def test_single_file_reaches_the_existing_distributed_coordinator(tmp_path, monkeypatch):
    import main
    path = configuration(tmp_path, execution="distributed", settings=dict(
        algorithm="qgm", dataset="synthetic", rounds=2, qgm_beta=.7, qgm_nesterov=True, device="cpu"))
    args = get_args(["--config", str(path), "--qgm-beta", ".8"])
    observed = []
    original = main.instantiate_class
    async def finished():
        pass
    def instantiate(path, **kwargs):
        if ".settings." in path:
            return original(path, **kwargs)
        from types import SimpleNamespace
        observed.append(kwargs["settings"])
        return SimpleNamespace(run=finished)
    monkeypatch.setattr(main, "instantiate_class", instantiate)
    monkeypatch.setattr(main, "ensure_future", lambda coroutine: coroutine.close())
    main.run(args)
    assert observed[0].beta == .8 and observed[0].nesterov
    assert observed[0].torch_device_name == "cpu" and not observed[0].dry_run


def test_interrupted_worker_is_stopped_and_recorded(tmp_path, monkeypatch):
    import dasklearn.experiment_runner as runner
    experiment = resolve_experiment(configuration(tmp_path), create_parser())
    class Process:
        def wait(self):
            raise KeyboardInterrupt
    process = Process()
    stopped = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, "provenance", lambda root: {"source_sha256": "test"})
    monkeypatch.setattr(runner, "_stop", lambda child: stopped.append(child))
    assert runner.execute_experiment(experiment) == 130
    assert stopped == [process]
    suite = json.loads((experiment.output / "manifest.json").read_text())
    assert suite["status"] == "interrupted" and suite["runs"][0]["status"] == "interrupted"


def test_real_sweep_runs_dags_and_saves_reproducible_artifacts(tmp_path):
    path = configuration(tmp_path, matrix={"partitioner": ["iid", "dirichlet"], "seed": [0, 1]})
    result = invoke(path)
    assert result.returncode == 0, result.stderr + result.stdout
    root = tmp_path / "results"
    suite = json.loads((root / "manifest.json").read_text())
    assert suite["status"] == "completed" and len(suite["runs"]) == 4
    assert len(suite["provenance"]["source_sha256"]) == 64
    for directory in sorted((root / "runs").iterdir()):
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["metrics"]["evaluated_peers"] == 4
        assert manifest["settings"]["learning"]["learning_rate"] == .1
        assert (Path(manifest["artifacts"]) / "workflow_dag").is_file()
        assert (directory / "resolved.json").is_file() and (directory / "run.log").is_file()
    with (root / "summary.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4 and {int(row["seed"]) for row in rows} == {0, 1}
    before = (root / "manifest.json").read_bytes()
    repeated = invoke(path)
    assert repeated.returncode != 0 and "already exists" in repeated.stderr
    assert (root / "manifest.json").read_bytes() == before


def test_simulate_mode_saves_dag_without_training(tmp_path):
    path = configuration(tmp_path, execution="simulate")
    result = invoke(path)
    assert result.returncode == 0, result.stderr + result.stdout
    manifest = json.loads(next((tmp_path / "results/runs").glob("*/manifest.json")).read_text())
    artifacts = Path(manifest["artifacts"])
    assert (artifacts / "workflow_dag").is_file()
    assert not (artifacts / "accuracies.csv").exists()
    assert manifest["metrics"] == {}


def test_runtime_failure_is_recorded_and_can_continue(tmp_path):
    path = configuration(tmp_path, runs=[dict(name="bad", settings=dict(peers=1)), dict(name="good")])
    result = invoke(path, "--continue-on-error")
    assert result.returncode == 1
    suite = json.loads((tmp_path / "results/manifest.json").read_text())
    assert suite["status"] == "failed"
    assert [run["status"] for run in suite["runs"]] == ["failed", "completed"]
    failed = next((tmp_path / "results/runs").glob("*bad*/run.log"))
    assert "at least two participants" in failed.read_text()
