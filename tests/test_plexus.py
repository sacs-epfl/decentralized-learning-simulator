"""Independent numerical, causal and wire-accounting checks for paper Plexus."""
import ast
import copy
import math
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from dasklearn.simulation.events import Event
from dasklearn.simulation.plexus.sampling import ordered_peers, sample
from dasklearn.simulation.plexus.settings import PlexusSettings
from dasklearn.simulation.plexus.simulation import PlexusSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.references import TaskRef
from dasklearn.util import MICROSECONDS
from helpers import settings, simulate


def reference_class(filename, name):
    """Execute pinned, unmodified authors' code without installing IPv8."""
    path = Path(__file__).parent / "reference" / "plexus" / filename
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module.startswith("dlsim"))]
    namespace = {"PeerManager": object, "GradientAggregation": object}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("seed", [0, 3, 110])
def test_hash_sampler_matches_executed_author_code(seed):
    author = reference_class("sample_manager.py", "SampleManager")(None, 4, 1)
    identities = tuple(f"plexus:{seed}:{peer}".encode() for peer in range(16))
    for round_nr in range(1, 30):
        expected = author.get_ordered_sample_list(round_nr, list(reversed(identities)))
        assert ordered_peers(round_nr, identities) == expected
        assert sample(round_nr, identities, 4) == tuple(identities.index(peer) for peer in expected[:4])
    assert sample(1, identities, 4) != sample(2, identities, 4)


@pytest.mark.asyncio
@pytest.mark.parametrize("fraction", [1, .5, .8])
async def test_exact_uniform_average_and_next_round_inputs(tmp_path, fraction):
    sim = simulate(PlexusSimulation, settings(PlexusSettings, tmp_path,
        sample_size=3, success_fraction=fraction, rounds=3))
    results = execute_local(sim)
    author = reference_class("fedavg.py", "FedAvg")
    trained = {(task.data["round"], task.data["peer"]): task
               for task in sim.workflow_dag.tasks.values() if task.func == "train"}
    assert len(sim.round_records) == 3
    for record in sim.round_records:
        round_nr = record["round"]
        assert record["aggregator"] in record["sample"]
        assert len(record["contributors"]) == max(1, math.floor(3 * fraction))
        models = [results[trained[round_nr, peer].name][0] for peer in record["contributors"]]
        before = [copy.deepcopy(model.state_dict()) for model in models]
        expected = author.aggregate(models, [])
        actual = results[record["aggregate_task"]][0]
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value, rtol=1e-6, atol=1e-7)
        for model, old in zip(models, before):
            for key, value in old.items():
                assert torch.equal(model.state_dict()[key], value)
        assert len([d for d in sim.delivery_records if d[0] == round_nr]) == 3
        if round_nr < 3:
            for peer in record["next_sample"]:
                task = trained[round_nr + 1, peer]
                assert task.data["model"] == TaskRef(record["aggregate_task"], 0)
                delivery = next(time for r, p, time in sim.delivery_records if r == round_nr and p == peer)
                assert task.data["time"] >= delivery >= record["time_us"]
    assert all(task.data["round"] <= 3 for task in trained.values())


@pytest.mark.asyncio
async def test_highest_bandwidth_current_sample_not_global_or_next(tmp_path):
    configuration = settings(PlexusSettings, tmp_path, sample_size=3, test_period=0)
    sim = PlexusSimulation(configuration)
    sim.setup_directories()
    def capacities():
        for peer in sim.clients:
            peer.bw_scheduler.bw_limit = (peer.index + 1) * 100_000
    sim.apply_traces = capacities
    sim.simulate()
    client = sim.clients[0]
    differing_samples = []
    for round_nr in range(1, 20):
        current = client.sample(round_nr)
        assert client.aggregator(round_nr) == max(current)
        assert all(c.sample(round_nr) == current for c in sim.clients)
        if max(current) != max(client.sample(round_nr + 1)):
            differing_samples.append(round_nr)
    assert differing_samples
    assert any(client.aggregator(r) != 5 for r in range(1, 20))


@pytest.mark.asyncio
async def test_one_reducer_exact_bytes_latency_and_bandwidth_bottleneck(tmp_path):
    elapsed = []
    for bandwidth in [100_000, 200_000]:
        sim = PlexusSimulation(settings(PlexusSettings, tmp_path / str(bandwidth),
            participants=3, sample_size=3, rounds=1, model_size_bytes=100_000,
            test_period=0, network_latency_ms=10))
        sim.setup_directories()
        def capacities():
            for client in sim.clients:
                client.bw_scheduler.bw_limit = bandwidth
                client.simulated_speed = 1
        sim.apply_traces = capacities
        sim.simulate()
        record = sim.round_records[0]
        aggregator = sim.clients[record["aggregator"]]
        upload = aggregator.data_transfer_size(100_000, {"phase": "upload", "round": 1})
        download = aggregator.data_transfer_size(100_000, {"phase": "aggregate", "round": 1})
        assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == 2 * (upload + download)
        assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == 2 * (upload + download)
        assert aggregator.bw_scheduler.total_bytes_received == 2 * upload
        assert aggregator.bw_scheduler.total_bytes_sent == 2 * download
        assert aggregator.bw_scheduler.bw_limit == bandwidth
        last_delivery = max(time for _, _, time in sim.delivery_records)
        expected = aggregator.get_train_time() + 2 * (upload + download) / bandwidth * MICROSECONDS + 20_000
        assert abs(last_delivery - expected) <= 4
        elapsed.append(last_delivery)
    assert elapsed[0] > elapsed[1]


@pytest.mark.asyncio
async def test_upload_arrival_threshold_duplicates_and_late_models(tmp_path):
    sim = simulate(PlexusSimulation, settings(PlexusSettings, tmp_path,
        sample_size=3, success_fraction=.8, test_period=0))
    client = sim.clients[0]
    round_nr = 50
    sample_peers = client.sample(round_nr)
    aggregator = sim.clients[client.aggregator(round_nr)]
    model = sim.initial_model
    def upload(sender):
        return Event(sim.current_time, aggregator.index, "received_upload", {
            "metadata": {"round": round_nr, "phase": "upload"},
            "from": sender, "payload": model})
    outsider = next(peer for peer in range(6) if peer not in sample_peers)
    aggregator.received_upload(upload(outsider))
    assert round_nr not in aggregator.uploads
    aggregator.received_upload(upload(sample_peers[0]))
    aggregator.received_upload(upload(sample_peers[0]))
    assert len(aggregator.uploads[round_nr]) == 1
    assert round_nr not in aggregator.completed_aggregations
    aggregator.received_upload(upload(sample_peers[1]))
    assert round_nr in aggregator.completed_aggregations
    record = sim.round_records[-1]
    assert record["contributors"] == tuple(sorted(sample_peers[:2]))
    count = len(sim.workflow_dag.tasks)
    aggregator.received_upload(upload(sample_peers[2]))
    assert len(sim.workflow_dag.tasks) == count
    assert round_nr not in aggregator.uploads


@pytest.mark.asyncio
async def test_aggregate_receipt_required_and_duplicate_safe(tmp_path):
    sim = simulate(PlexusSimulation, settings(PlexusSettings, tmp_path,
        sample_size=3, rounds=1, test_period=0))
    record = sim.round_records[0]
    peer = sim.clients[record["next_sample"][0]]
    event = Event(sim.current_time, peer.index, "received_aggregate", {
        "from": record["aggregator"], "metadata": {"round": 1},
        "payload": TaskRef(record["aggregate_task"], 0)})
    before = len(sim.delivery_records)
    peer.received_aggregate(event)
    assert len(sim.delivery_records) == before
    assert 2 not in peer.started_rounds
    # A correctly shaped message from a different sender cannot activate a round.
    event.data["metadata"]["round"] = 20
    event.data["from"] = (peer.aggregator(20) + 1) % 6
    peer.received_aggregate(event)
    assert 21 not in peer.started_rounds
    peer.latest_received_round = 100
    event.data["from"] = peer.aggregator(20)
    peer.received_aggregate(event)
    assert len(sim.delivery_records) == before


@pytest.mark.asyncio
async def test_no_overlapping_training_even_with_partial_rounds(tmp_path):
    sim = PlexusSimulation(settings(PlexusSettings, tmp_path,
        sample_size=4, success_fraction=.5, rounds=0, duration=4 * MICROSECONDS, test_period=0))
    sim.setup_directories()
    def capacities():
        for peer in sim.clients:
            peer.simulated_speed = 100 if peer.index == 0 else 1
    sim.apply_traces = capacities
    sim.simulate()
    starts = defaultdict(list)
    for task in sim.workflow_dag.tasks.values():
        if task.func == "train":
            starts[task.data["peer"]].append(task.data["time"])
    assert len(sim.round_records) > 3
    for peer, times in starts.items():
        duration = sim.clients[peer].get_train_time()
        assert all(b - a >= duration for a, b in zip(times, times[1:]))


@pytest.mark.asyncio
async def test_single_peer_has_zero_wire_bytes_and_no_snapshot_dependency(tmp_path):
    sim = simulate(PlexusSimulation, settings(PlexusSettings, tmp_path,
        participants=1, sample_size=1, rounds=3, test_period=0))
    assert len(sim.round_records) == len(sim.delivery_records) == 3
    assert sim.clients[0].bw_scheduler.total_bytes_sent == 0
    assert len(execute_local(sim)) == len(sim.workflow_dag.tasks)


@pytest.mark.asyncio
async def test_saved_measurements_and_streaming_execution(tmp_path):
    import pandas as pd
    frames = []
    for retain in [True, False]:
        sim = simulate(PlexusSimulation, settings(PlexusSettings, tmp_path / str(retain), sample_size=3))
        result = execute_local(sim, retain_results=retain)
        assert bool(result) == retain
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(Path(sim.data_dir) / "accuracies.csv").sort_values(["time", "peer"]).reset_index(drop=True))
        sim.save_measurements()
        rounds = pd.read_csv(Path(sim.data_dir) / "plexus-rounds.csv")
        assert list(rounds["round"]) == [1, 2, 3]
        deliveries = pd.read_csv(Path(sim.data_dir) / "plexus-deliveries.csv")
        assert len(deliveries) == 9
    pd.testing.assert_frame_equal(*frames)


@pytest.mark.asyncio
async def test_global_benchmark_snapshots_use_received_models_and_idle_initialization(tmp_path):
    from dasklearn.benchmark import configure_simulation
    sim = PlexusSimulation(settings(PlexusSettings, tmp_path, sample_size=2,
        rounds=0, duration=2 * MICROSECONDS, test_period=0, model_size_bytes=1000))
    sim.setup_directories()
    measurements = configure_simulation(sim, [0, 1, 2], {
        "profile": "uniform", "bandwidth_bytes_per_second": 100_000, "compute_step_ms": 5,
    }, evaluation_model="global")
    sim.simulate()
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    assert len(sim.round_records) > 1
    assert [point["time"] for point in measurements] == [0, 1, 2]
    assert measurements[-1]["bytes_sent"] > 0
    initial_tasks = [task for task in sim.workflow_dag.tasks.values() if task.func == "initialize"]
    assert len(initial_tasks) == 1
    assert sim.initial_model == TaskRef(initial_tasks[0].name, 0)
    tests = [task for task in sim.workflow_dag.tasks.values() if task.name.startswith("benchmark_global_test")]
    assert len(tests) == 3
    assert all(task.data["peer"] == -1 for task in tests)
    assert len(execute_local(sim)) == len(sim.workflow_dag.tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("extra,match", [
    ({"sample_size": 0}, "sample_size"), ({"sample_size": 7}, "sample_size"),
    ({"success_fraction": 0}, "success_fraction"), ({"success_fraction": float("nan")}, "success_fraction"),
    ({"success_fraction": 1.1}, "success_fraction"), ({"churn": "synthetic"}, "stable membership"),
    ({"synchronous": True}, "synchronous"), ({"test_method": "global"}, "snapshot"),
])
async def test_unsupported_settings_are_rejected(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        PlexusSimulation(settings(PlexusSettings, tmp_path, **({"sample_size": 3} | extra)))


def test_cli_settings_and_experiment_runner_integration(tmp_path):
    from args import get_args
    from dasklearn.settings_factory import session_options
    from dasklearn.experiments import validate_run
    from dasklearn.simulation.peer_baselines import baseline_spec
    args = get_args(["--algorithm", "plexus", "--peers", "6", "--sample-size", "3",
                     "--success-fraction", ".8", "--momentum", "0", "--rounds", "2"])
    for name in ["synchronous", "compute_validation_loss_global_model", "dry_run", "profile", "log_bandwidth_utilization"]:
        setattr(args, name, False)
    config = PlexusSettings(**session_options(args, "cpu"))
    assert config.sample_size == 3 and config.success_fraction == .8
    assert PlexusSettings.from_dict(config.to_dict()) == config
    validate_run(vars(args))
    cls, kind, specific = baseline_spec("plexus", 6, 3)
    assert (cls, kind, specific) == (PlexusSimulation, PlexusSettings, {"sample_size": 3})
