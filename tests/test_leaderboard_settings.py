import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from args import create_parser
from dasklearn.benchmark import load_benchmark
from dasklearn.leaderboard_server import create_server, load_settings

ROOT = Path(__file__).resolve().parents[1]


def catalog(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(dict(version=1, default="extreme", settings=[
        dict(id="iid", label="IID / uniform", output="iid-results"),
        dict(id="extreme", label="Dirichlet alpha 0.1 / heterogeneous", output="extreme-results")])))
    return path


@pytest.fixture
def served(tmp_path):
    path = catalog(tmp_path)
    for name in ["iid", "extreme"]:
        output = tmp_path / f"{name}-results"
        output.mkdir()
        (output / "leaderboard.json").write_text(json.dumps(dict(name=name, status="running")))
        (output / "leaderboard.csv").write_text(f"setting\n{name}\n")
    with create_server(catalog=path, port=0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def test_catalog_routes_each_setting_without_mixing_results_or_csv(served):
    with urlopen(served + "/settings.json") as response:
        data = json.load(response)
    assert data["default"] == "extreme"
    assert [entry["id"] for entry in data["settings"]] == ["iid", "extreme"]
    assert all(set(entry) == {"id", "label"} for entry in data["settings"])
    for query, expected in [("", "extreme"), ("?setting=iid", "iid"), ("?setting=extreme", "extreme")]:
        with urlopen(served + "/leaderboard.json" + query) as response:
            assert json.load(response)["name"] == expected
        with urlopen(served + "/leaderboard.csv" + query) as response:
            assert response.read().decode() == f"setting\n{expected}\n"


@pytest.mark.parametrize("query", ["?setting=unknown", "?setting=../../etc", "?setting=iid&setting=extreme"])
def test_unknown_or_ambiguous_settings_cannot_select_arbitrary_files(served, query):
    with pytest.raises(HTTPError) as error:
        urlopen(served + "/leaderboard.json" + query)
    assert error.value.code == 404


def test_single_directory_server_exposes_compatible_catalog(tmp_path):
    default, entries = load_settings(output=tmp_path)
    assert default == "default" and entries[0]["output"] == tmp_path.resolve()
    with pytest.raises(ValueError, match="either"):
        load_settings()
    with pytest.raises(ValueError, match="either"):
        load_settings(output=tmp_path, catalog=catalog(tmp_path))


def test_pending_setting_is_allowed_before_results_exist(tmp_path):
    default, entries = load_settings(catalog=catalog(tmp_path))
    assert default == "extreme" and len(entries) == 2
    assert not entries[1]["output"].exists()


@pytest.mark.parametrize("mutation", ["duplicate", "invalid_id", "missing_default", "empty", "version"])
def test_invalid_catalog_rejected_before_serving(tmp_path, mutation):
    path = catalog(tmp_path)
    data = json.loads(path.read_text())
    if mutation == "duplicate":
        data["settings"][1]["id"] = "iid"
    elif mutation == "invalid_id":
        data["settings"][0]["id"] = "../escape"
    elif mutation == "missing_default":
        data["default"] = "missing"
    elif mutation == "empty":
        data["settings"] = []
    else:
        data["version"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_settings(catalog=path)


def test_extreme_experiment_changes_only_partition_and_hardware_profile():
    baseline = load_benchmark(ROOT / "experiments/leaderboard-cifar10-32-global900.yaml", create_parser())
    extreme = load_benchmark(ROOT / "experiments/leaderboard-cifar10-32-dirichlet01-heterogeneous.yaml", create_parser())
    assert extreme.network["profile"] == "heterogeneous"
    assert extreme.network | {"profile": "uniform"} == baseline.network
    assert extreme.budgets == baseline.budgets == [60, 300, 900]
    assert extreme.execution_devices == baseline.execution_devices
    assert extreme.evaluation_model == baseline.evaluation_model == "global"
    assert extreme.tuning is None and len(extreme.runs) == 15
    for new, old in zip(extreme.runs, baseline.runs):
        assert new.name == old.name and new.seed == old.seed
        assert new.options["partitioner"] == "dirichlet" and new.options["alpha"] == .1
        assert new.options | {"partitioner": "iid"} == old.options
