import copy
from types import SimpleNamespace

import pytest
import torch

from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.bandwidth_scheduler import BWScheduler
from dasklearn.simulation.slot_bandwidth_scheduler import SlotBWScheduler
from dasklearn.simulation.conflux.settings import ConfluxSettings
from dasklearn.simulation.conflux.simulation import ConfluxSimulation
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.simulation.conflux_v2.settings import ConfluxV2Settings
from dasklearn.simulation.conflux_v2.simulation import ConfluxV2Simulation
from dasklearn.simulation.shatter.settings import ShatterSettings
from dasklearn.simulation.shatter.simulation import ShatterSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.util import MICROSECONDS


def settings(cls, tmp_path, **kwargs):
    values = dict(algorithm="test", seed=3, work_dir=str(tmp_path), dataset="synthetic", model="mlp",
                  learning=LearningSettings(0.1, 0, 0, 16, 2), participants=6, rounds=3,
                  duration=30 * MICROSECONDS, test_interval=0, test_period=2 * MICROSECONDS,
                  test_method="snapshot", dry_run=True, unit_testing=True, torch_threads=1)
    values.update(kwargs)
    return cls(**values)


def simulate(cls, configuration):
    sim = cls(configuration)
    sim.setup_directories()
    sim.simulate()
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    return sim


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,config,kwargs", [
    (ConfluxSimulation, ConfluxSettings, dict(sample_size=3, chunks_in_sample=3)),
    (ConfluxV2Simulation, ConfluxV2Settings, dict(sample_size=3, chunks_in_sample=3)),
    (ShatterSimulation, ShatterSettings, dict(k=3, r=2)),
])
async def test_real_protocol_and_training_dag(tmp_path, cls, config, kwargs):
    sim = simulate(cls, settings(config, tmp_path, **kwargs))
    assert all(type(client.bw_scheduler) is BWScheduler for client in sim.clients)
    trained = [task for task in sim.workflow_dag.tasks.values() if task.func == "train"]
    assert trained and max(task.data["round"] for task in trained) == 3
    assert sum(client.compute_time for client in sim.clients) > 0
    assert sum(client.bw_scheduler.total_bytes_sent for client in sim.clients) == sum(client.bw_scheduler.total_bytes_received for client in sim.clients)
    # One shared fixed-cadence evaluation, including initially idle sampled nodes.
    snapshots = [task for task in sim.workflow_dag.tasks.values() if task.name.startswith("test_snapshot")]
    assert len([task for task in snapshots if task.data["time"] == 2 * MICROSECONDS]) == 6
    results = execute_local(sim)
    assert len(results) == len(sim.workflow_dag.tasks)
    sim.merge_accuracies_files()
    import pandas as pd
    frame = pd.read_csv(sim.data_dir + "/accuracies.csv")
    assert frame["learning_rate"].eq(0.1).all()
    assert frame["peer"].between(0, 5).all()
    assert frame["time"].between(2, 30).all()
    assert frame["accuracy"].between(0, 100).all()
    assert len(frame) == len(snapshots)


@pytest.mark.asyncio
@pytest.mark.parametrize("fraction", [0.5, 1.0])
async def test_v2_reconstructs_exact_common_average(tmp_path, fraction):
    sim = simulate(ConfluxV2Simulation, settings(ConfluxV2Settings, tmp_path,
        sample_size=4, chunks_in_sample=3, success_fraction=fraction, rounds=2))
    results = execute_local(sim)
    for round_nr in [1, 2]:
        contributors = sim.contributor_sets[round_nr]
        models = [results[t.name][0] for t in sim.workflow_dag.tasks.values()
                  if t.func == "train" and t.data["round"] == round_nr and t.data["peer"] in contributors]
        expected = torch.stack([ChunkManager.get_flat_params(model) for model in models]).mean(0)
        reconstructed = [results[t.name][0] for t in sim.workflow_dag.tasks.values()
                         if t.func == "reconstruct_from_chunks" and t.data["round"] == round_nr]
        assert len(reconstructed) == sim.settings.sample_size
        for model in reconstructed:
            torch.testing.assert_close(ChunkManager.get_flat_params(model), expected)


def test_chunks_preserve_remainder_and_do_not_mutate_inputs():
    model = torch.nn.Linear(3, 2)
    chunks = ChunkManager.chunk_model(model, 3)
    inputs = [[chunk] for chunk in chunks]
    restored = ChunkManager.reconstruct_model(inputs, torch.nn.Linear(3, 2))
    torch.testing.assert_close(ChunkManager.get_flat_params(model), ChunkManager.get_flat_params(restored))
    assert all(isinstance(chunk, list) for chunk in inputs)
    with pytest.raises(ValueError):
        ChunkManager.chunk_model(model, 100)


def test_asymmetric_slot_accounting():
    sim = SimpleNamespace(current_time=0, events=[])
    sim.schedule = lambda event: sim.events.append(event)
    sender = SimpleNamespace(index=0, simulator=sim)
    receiver = SimpleNamespace(index=1, simulator=sim, on_incoming_model=lambda event: None)
    a, b = SlotBWScheduler(sender, total_bw=10_000_000), SlotBWScheduler(receiver, total_bw=1_000_000)
    transfer = a.add_transfer(b, 1_000_000, None, {})
    sim.current_time = sim.events[0].time
    a.on_outgoing_transfer_complete(transfer)
    assert a.total_bytes_sent == b.total_bytes_received == 1_000_000


def test_cancelled_bytes_and_work_conservation():
    sim = SimpleNamespace(current_time=0, events=[])
    sim.schedule = lambda event: sim.events.append((event.time, event.index, event))
    a = BWScheduler(SimpleNamespace(index=0, simulator=sim))
    b = BWScheduler(SimpleNamespace(index=1, simulator=sim))
    a.bw_limit = b.bw_limit = 1_000_000
    transfer = a.add_transfer(b, 1_000_000, None, {})
    assert sim.events[0][0] == MICROSECONDS
    sim.current_time = MICROSECONDS // 4
    b.kill_transfer(transfer)
    b.kill_transfer(transfer)
    assert a.total_bytes_sent == b.total_bytes_received == 250_000
    assert a.allocated_outgoing == b.allocated_incoming == 0


@pytest.mark.asyncio
async def test_payloads_are_snapshots_and_training_does_not_mutate_predecessor(tmp_path):
    from dasklearn import functions
    sim = simulate(ConfluxV2Simulation, settings(ConfluxV2Settings, tmp_path,
                                               sample_size=2, chunks_in_sample=2))
    message = {"peers": [1]}
    sim.clients[0].send_message_to_client(1, "ready", message)
    message["peers"].append(2)
    assert sim.events[-1][2].data["message"] == {"peers": [1]}
    functions.dataset = None
    functions.model_trainers = None
    model = functions.initialize(sim.settings, {})[0]
    before = copy.deepcopy(model.state_dict())
    functions.train(sim.settings, {"model": model, "peer": 0, "round": 1, "time": 0})
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)


@pytest.mark.asyncio
async def test_v2_rejects_unsupported_churn(tmp_path):
    with pytest.raises(ValueError, match="stable membership"):
        ConfluxV2Simulation(settings(ConfluxV2Settings, tmp_path, sample_size=2, churn="synthetic"))
