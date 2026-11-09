import csv
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from args import create_parser
from dasklearn.algorithms import ALGORITHMS
from dasklearn.benchmark import budget_metrics, checkpoint_budget_view, leaderboard, load_benchmark, read_curve
from dasklearn.leaderboard_server import create_server

ROOT = Path(__file__).resolve().parents[1]


def config(tmp_path, **changes):
    document = dict(version=1, name="paired-test", output="results", benchmark=dict(
        budgets=[3, 5], seeds=[0, 1], evaluation_period=2, accuracy_targets=[40, 80]),
        settings=dict(dataset="synthetic", model="mlp", peers=4, device="cpu", evaluation_split="validation"),
        runs=[dict(name="dpsgd", settings=dict(algorithm="dpsgd", topology="ring"))])
    document.update(changes)
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(document))
    return path


def test_plan_has_exact_irregular_budgets_and_fixed_paired_scenario(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Planning accessed models/datasets")
    monkeypatch.setattr("dasklearn.models.create_model", forbidden)
    monkeypatch.setattr("dasklearn.datasets.create_dataset", forbidden)
    benchmark = load_benchmark(config(tmp_path), create_parser())
    assert benchmark.times == [0, 2, 3, 4, 5]
    assert [run.seed for run in benchmark.runs] == [0, 1]
    assert all(run.options["duration"] == 5 and run.options["rounds"] == 0 for run in benchmark.runs)
    assert not benchmark.output.exists()
    example = load_benchmark(ROOT / "experiments/leaderboard.yaml", create_parser())
    # MoTEF-VR awaits screening; Conflux V2 is scoped to its CIFAR-10 campaign.
    assert {run.options["algorithm"] for run in example.runs} == set(ALGORITHMS) - {"motef-vr", "conflux-v2"}
    assert len(example.runs) == 23 * 3


@pytest.mark.parametrize("changes,match", [
    ({"benchmark": dict(budgets=[0], seeds=[0])}, "positive"),
    ({"benchmark": dict(budgets=[5], seeds=[True])}, "integers"),
    ({"benchmark": dict(budgets=[5, 5], seeds=[0])}, "duplicates"),
    ({"benchmark": dict(budgets=[5], seeds=[0], evaluation_period=0)}, "positive"),
    ({"benchmark": dict(budgets=[5], seeds=[0], accuracy_targets=[101])}, "percentages"),
    ({"benchmark": dict(budgets=[5], seeds=[0], typo=3)}, "mapping"),
    ({"benchmark": dict(budgets=[10**12], seeds=[0], evaluation_period=1)}, "10,000"),
    ({"settings": dict(seed=1)}, "benchmark fields"),
    ({"settings": dict(churn="synthetic")}, "stable membership"),
    ({"runs": [dict(name="x", settings=dict(algorithm="dpsgd", dataset="digits"))]}, "fixed scenario"),
    ({"runs": [dict(name="x"), dict(name="x")]}, "unique"),
    ({"matrix": dict(seed=[0])}, "either matrix or runs"),
])
def test_invalid_benchmarks_fail_before_creation(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        load_benchmark(config(tmp_path, **changes), create_parser())
    assert not (tmp_path / "results").exists()


def test_subset_and_overrides_preserve_requested_seed_zero(tmp_path):
    path = config(tmp_path, runs=[dict(name="a"), dict(name="b")])
    result = load_benchmark(path, create_parser(), budgets=[7], seeds=[0], variants=["b"], period=4)
    assert result.times == [0, 4, 7]
    assert [run.name for run in result.runs] == ["b-s0"]
    with pytest.raises(ValueError, match="variants"):
        load_benchmark(path, create_parser(), variants=["missing"])


def write_observations(tmp_path, rows):
    data = tmp_path / "data" / "run"
    data.mkdir(parents=True, exist_ok=True)
    (tmp_path / "network.json").write_text(json.dumps([dict(time=t, bytes_sent=t*100) for t in [0, 3, 5]]))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["time", "peer", "accuracy", "loss"])
    writer.writeheader()
    writer.writerows(rows)
    (data / "accuracies_0.csv").write_text(buffer.getvalue())
    return data


def test_partial_checkpoints_never_publish_or_fill_from_the_future(tmp_path):
    rows = [dict(time=0, peer=peer, accuracy=20, loss=2) for peer in range(2)]
    rows += [dict(time=3, peer=0, accuracy=40, loss=1)]
    rows += [dict(time=5, peer=peer, accuracy=60, loss=.5) for peer in range(2)]
    data = write_observations(tmp_path, rows)
    assert [p["time"] for p in read_curve(tmp_path, 2, [0, 3, 5])] == [0]
    with pytest.raises(ValueError, match="Incomplete"):
        read_curve(tmp_path, 2, [0, 3, 5], strict=True)
    with (data / "accuracies_0.csv").open("a") as stream:
        stream.write("3,1,40,1\n5,0,")  # Ignore only the uncommitted tail.
    curve = read_curve(tmp_path, 2, [0, 3, 5], strict=True)
    metrics = budget_metrics(curve, 5, [40, 80])
    assert metrics["average_accuracy"] == 28
    assert metrics["accuracy"] == 60 and metrics["bytes_sent"] == 500
    assert metrics["time_to_accuracy"] == {"40": 3, "80": None}
    assert budget_metrics(curve, 4, []) is None


def test_checkpoint_view_uses_exact_600_second_scores_without_changing_saved_results(tmp_path):
    benchmark = load_benchmark(config(tmp_path, runs=[dict(name='a'), dict(name='b')],
        benchmark=dict(budgets=[60, 300, 900], seeds=[0, 1], evaluation_period=60)), create_parser())
    suite = benchmark.manifest() | dict(status='completed')
    for run in suite['runs']:
        run.update(status='completed', wall_seconds=1, curve=[])
        for stamp in benchmark.times:
            accuracy = (95 if run['variant']=='a' else 40) if stamp==900 else (80 if run['variant']=='a' else 90)
            run['curve'].append(dict(time=stamp, accuracy=accuracy+run['seed'], loss=1.,
                                    worst_decile_accuracy=accuracy, bytes_sent=stamp*100))
    document = leaderboard(suite)
    before = deepcopy(document)
    view = checkpoint_budget_view(document, 600, times=benchmark.times, seeds=benchmark.seeds)
    assert document==before and view['budgets']==[60, 300, 600, 900]
    rows = next(table['rows'] for table in view['tables'] if table['budget']==600)
    assert rows[0]['variant']=='b' and rows[0]['rank']==1 and rows[0]['accuracy']['mean']==90.5
    assert rows[1]['variant']=='a' and rows[1]['rank']==2 and rows[1]['accuracy']['mean']==80.5
    assert view['tables'][-1]==document['tables'][-1]
    assert view['tables'][-1]['rows'][0]['variant']=='a'
    assert checkpoint_budget_view(view, 600, times=benchmark.times, seeds=benchmark.seeds) is view
    assert checkpoint_budget_view(document, 600, times=[0, 900], seeds=benchmark.seeds) is document


@pytest.mark.parametrize("bad", [dict(accuracy=float("nan")), dict(accuracy=101), dict(peer=3), dict(time=4)])
def test_corrupt_metrics_are_rejected(tmp_path, bad):
    write_observations(tmp_path, [dict(time=0, peer=0, accuracy=20, loss=1) | bad])
    with pytest.raises(ValueError, match="Invalid"):
        read_curve(tmp_path, 2, [0, 3, 5])


def test_duplicate_peers_rejected_and_worst_decile_uses_clients(tmp_path):
    rows = [dict(time=t, peer=p, accuracy=10+p, loss=1) for t in [0, 3, 5] for p in range(11)]
    write_observations(tmp_path, rows)
    assert read_curve(tmp_path, 11, [0, 3, 5])[-1]["worst_decile_accuracy"] == 10.5
    write_observations(tmp_path, rows + [rows[-1]])
    with pytest.raises(ValueError, match="Duplicate"):
        read_curve(tmp_path, 11, [0, 3, 5])


def test_invalid_network_cannot_produce_a_score(tmp_path):
    write_observations(tmp_path, [])
    (tmp_path / "network.json").write_text(json.dumps([
        dict(time=0, bytes_sent=0), dict(time=3, bytes_sent=300), dict(time=5, bytes_sent=200)]))
    with pytest.raises(ValueError, match="communication"):
        read_curve(tmp_path, 2, [0, 3, 5])


def test_official_ranks_require_all_paired_seeds_and_failed_runs_are_excluded(tmp_path):
    benchmark = load_benchmark(config(tmp_path, runs=[dict(name="a"), dict(name="b")]), create_parser())
    suite = benchmark.manifest() | dict(status="running")
    for index, run in enumerate(suite["runs"]):
        run.update(status="completed", wall_seconds=10, curve=[dict(time=t, accuracy=20+index, loss=1,
            bytes_sent=t*100, worst_decile_accuracy=10) for t in benchmark.times])
    result = leaderboard(suite)
    assert not result["tables"][0]["provisional"]
    assert result["tables"][0]["rows"][0]["variant"] == "b"
    assert result["tables"][0]["rows"][0]["accuracy"]["sd"] == pytest.approx(2**-.5)
    suite["runs"][-1]["status"] = "failed"
    result = leaderboard(suite)
    assert result["tables"][0]["provisional"]
    assert all(row["rank"] is None for row in result["tables"][0]["rows"])
    b = next(row for row in result["tables"][0]["rows"] if row["variant"] == "b")
    assert b["seeds"] == [0] and b["failed_runs"] == 1
    assert b["time_to_accuracy"]["80"] == dict(reached=0, evaluated=1, mean_seconds=None, censor_seconds=3)


def test_interrupted_benchmark_stops_worker_and_preserves_status(tmp_path, monkeypatch):
    import dasklearn.benchmark as module
    benchmark = load_benchmark(config(tmp_path), create_parser())
    class Process:
        def poll(self):
            raise KeyboardInterrupt
    process = Process()
    stopped = []
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(module, "_stop", lambda child: stopped.append(child))
    monkeypatch.setattr(module, "provenance", lambda root: {})
    assert module.execute_benchmark(benchmark) == 130
    assert stopped == [process]
    result = json.loads((benchmark.output / "leaderboard.json").read_text())
    assert result["status"] == "interrupted"
    assert result["runs"][0]["status"] == "interrupted"
    assert result["runs"][1]["status"] == "planned"
    with pytest.raises(ValueError, match="already exists"):
        module.execute_benchmark(benchmark)


def test_dashboard_only_serves_results_and_assets(tmp_path):
    (tmp_path / "leaderboard.json").write_text('{"status":"running"}')
    (tmp_path / "secret.txt").write_text("private")
    with create_server(tmp_path, port=0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base + "/") as response:
                assert response.status == 200 and b"Algorithm leaderboard" in response.read()
            with urlopen(base + "/leaderboard.json") as response:
                assert json.load(response)["status"] == "running"
                assert response.headers["Cache-Control"] == "no-store"
            for route in ["/secret.txt", "/../secret.txt", "/runs/x/run.log"]:
                with pytest.raises(HTTPError) as error:
                    urlopen(base + route)
                assert error.value.code == 404
        finally:
            server.shutdown()
            thread.join()


def test_real_benchmark_exports_complete_metrics_and_fedavg_trains_past_one_round(tmp_path):
    path = config(tmp_path, benchmark=dict(budgets=[1, 3], seeds=[0], evaluation_period=2, accuracy_targets=[0, 100]),
        runs=[dict(name="fedavg", settings=dict(algorithm="fl", sample_size=4)),
              dict(name="gossip", settings=dict(algorithm="gossip", gl_period=1))])
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_benchmark.py"), str(path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "results"
    report = json.loads((output / "leaderboard.json").read_text())
    assert report["status"] == "completed"
    assert all(not table["provisional"] for table in report["tables"])
    for run in report["runs"]:
        assert [p["time"] for p in run["curve"]] == [0, 1, 2, 3]
        assert all(p["peers"] == 4 for p in run["curve"])
        rows = list(csv.DictReader(next((output / "runs" / run["name"]).glob("data/*/accuracies.csv")).open()))
        assert max(int(row["round"]) for row in rows) > 1
        assert run["curve"][-1]["bytes_sent"] > 0
    assert len(list(csv.DictReader((output / "leaderboard.csv").open()))) == 4
