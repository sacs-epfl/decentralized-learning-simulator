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
from dasklearn.simulation.shatter.settings import ShatterSettings
from dasklearn.simulation.shatter.simulation import ShatterSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.util import MICROSECONDS
from helpers import settings, simulate






@pytest.mark.asyncio
@pytest.mark.parametrize("cls,config,kwargs", [
    (ConfluxSimulation, ConfluxSettings, dict(sample_size=3, chunks_in_sample=3)),
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
    sim = simulate(ShatterSimulation, settings(ShatterSettings, tmp_path, k=2, r=2))
    message = {"peers": [1]}
    sim.clients[0].send_control(1, "ready", message)
    message["peers"].append(2)
    assert sim.events[-1][2].data["payload"]["message"] == {"peers": [1]}
    functions.dataset = None
    functions.model_trainers = None
    model = functions.initialize(sim.settings, {})[0]
    before = copy.deepcopy(model.state_dict())
    functions.train(sim.settings, {"model": model, "peer": 0, "round": 1, "time": 0})
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)
