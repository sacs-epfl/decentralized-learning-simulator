import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dasklearn.privacy.incoming import IncomingModelObserver, complete_partial_model
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.simulation.events import Event, INCOMING_PAYLOAD
from dasklearn.tasks.dag import WorkflowDAG
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.references import TaskRef
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings


def test_partial_completion_uses_only_received_chunks_and_handles_remainder():
    baseline, victim = nn.Linear(3, 2), nn.Linear(3, 2)
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.zero_()
        for parameter in victim.parameters():
            parameter.fill_(7)
    chunks = ChunkManager.chunk_model(victim, 3)
    assert [chunk.numel() for chunk in chunks] == [2, 2, 4]
    completed = complete_partial_model(baseline, {2: chunks[2]}, 3)
    expected = torch.tensor([0., 0., 0., 0., 7., 7., 7., 7.])
    assert torch.equal(ChunkManager.get_flat_params(completed), expected)
    assert ChunkManager.get_flat_params(baseline).eq(0).all()
    assert ChunkManager.get_flat_params(victim).eq(7).all()
    full = complete_partial_model(baseline, dict(enumerate(chunks)), 3)
    assert torch.equal(ChunkManager.get_flat_params(full), ChunkManager.get_flat_params(victim))
    with pytest.raises(ValueError, match="shape"):
        complete_partial_model(baseline, {0: torch.zeros(3)}, 3)
    with pytest.raises(ValueError, match="indices"):
        complete_partial_model(baseline, {3: torch.zeros(2)}, 3)


def test_observer_freezes_background_and_separates_rounds_senders_and_receivers():
    sim = SimpleNamespace(latest_models={})
    receiver = SimpleNamespace(index=0, simulator=sim)
    observer = IncomingModelObserver(attackers=[0], rounds=[2, 3], chunk_count=4)
    old, new = TaskRef("old", 0), TaskRef("new", 0)
    observer.on_model_published(receiver, 1, old)

    def deliver(sender, round_nr, chunk, time):
        source = f"chunks-{sender}-{round_nr}"
        observer.on_payload(receiver, Event(time, 0, INCOMING_PAYLOAD, {
            "from": sender, "payload": TaskRef(source, chunk),
            "metadata": {"round": round_nr, "chunk": chunk}}))
    deliver(1, 2, 0, 10)
    observer.on_model_published(receiver, 2, new)
    deliver(1, 2, 2, 20)
    deliver(1, 2, 2, 25)  # A duplicate physical delivery is not an extra slice.
    deliver(1, 3, 1, 30)
    deliver(2, 2, 1, 35)
    deliver(0, 2, 3, 40)  # Local self-delivery is excluded.
    deliver(1, 1, 3, 45)  # Unselected round is excluded.
    assert len(observer.observations) == 3
    case = observer.observations[0, 1, 2]
    assert case.baseline == old
    assert case.first_delivery_us == 10 and case.last_delivery_us == 25
    assert case.deliveries == 3 and set(case.chunks) == {0, 2}
    assert observer.observations[0, 1, 3].baseline == new
    assert set(observer.observations[0, 1, 3].chunks) == {1}
    other_receiver = SimpleNamespace(index=3, simulator=sim)
    observer.on_payload(other_receiver, Event(50, 3, INCOMING_PAYLOAD, {
        "from": 1, "payload": TaskRef("other", 0), "metadata": {"round": 2, "chunk": 0}}))
    assert len(observer.observations) == 3


def _simulate(config_type, cls, tmp_path, observer, *, algorithm, latency=0, **extra):
    Task.COUNTER.clear()
    config = settings(config_type, tmp_path, algorithm=algorithm, network_latency_ms=latency,
                      test_period=0, test_method="individual", **extra)
    sim = cls(config)
    sim.setup_directories()
    if observer:
        sim.observers.append(observer)
    sim.simulate()
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    return sim


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_count", [None, 1, 4, 16])
@pytest.mark.parametrize("latency", [0, 5])
async def test_completed_delivery_audits_preserve_protocol_and_training_and_replay(tmp_path, chunk_count, latency):
    from dasklearn.algorithms import class_paths, import_class
    algorithm = "dpsgd" if chunk_count is None else "shatter"
    config_type, cls = map(import_class, class_paths(algorithm))
    extra = {"topology": "ring"} if chunk_count is None else {"k": chunk_count, "r": 2}
    baseline = _simulate(config_type, cls, tmp_path / "baseline", None,
                         algorithm=algorithm, latency=latency, **extra)
    expected = execute_local(baseline)
    observer = IncomingModelObserver(attackers=[0], rounds=[3], chunk_count=chunk_count)
    # This is an observer-invariance comparison: hold the modeled wire size
    # fixed. PyTorch's pickle storage IDs can change serialized lengths with
    # allocator/test order, even for the same model structure.
    sim = _simulate(config_type, cls, tmp_path / "observed", observer,
                    algorithm=algorithm, latency=latency, model_size_bytes=baseline.model_size, **extra)
    assert sim.event_trace_hash.hexdigest() == baseline.event_trace_hash.hexdigest()
    assert sim.workflow_dag.serialize() == baseline.workflow_dag.serialize()
    assert [client.bw_scheduler.total_bytes_sent for client in sim.clients] == [
        client.bw_scheduler.total_bytes_sent for client in baseline.clients]
    assert observer.observations
    audits = observer.attach_audits(sim, max_examples=20)
    for case in observer.observations.values():
        assert case.attacker == 0 and case.sender != 0 and case.round_nr == 3
        assert case.last_delivery_us <= sim.current_time
        if case.baseline:
            own_source = sim.workflow_dag.tasks[case.baseline.task]
            assert own_source.data["peer"] == 0
            assert own_source.available_at <= case.first_delivery_us
    if chunk_count is not None:
        completions = [task for task in sim.workflow_dag.tasks.values() if task.func == "complete_incoming"]
        assert len(completions) == len(audits)
        for task, case in zip(completions, [value for _, value in sorted(observer.observations.items())]):
            assert task.data["chunks"] == case.chunks
            # The completion task has only a receiver background and delivered
            # slice outputs as direct dependencies; no private victim model.
            direct = {TaskRef(source.name, output) for source, output in task.inputs}
            assert direct == {task.data["baseline"], *case.chunks.values()}
    sim.workflow_dag.build_task_indices()
    actual = execute_local(sim)
    for name, values in expected.items():
        for a, b in zip(actual[name], values):
            if isinstance(b, torch.Tensor):
                assert torch.equal(a, b)
            else:
                for key, value in b.state_dict().items():
                    assert torch.equal(a.state_dict()[key], value)
    rows = [actual[task.name][0] for task in audits]
    path = tmp_path / "incoming-dag"
    sim.workflow_dag.save_to_file(str(path))
    sim.workflow_dag = WorkflowDAG.load_from_file(str(path))
    sim.workflow_dag.build_task_indices()
    assert execute_local(sim, retain_results=False) == {}
    for task, expected_rows in zip(audits, rows):
        artifact = json.loads((Path(sim.data_dir) / f"mia-{task.name}.json").read_text())
        assert artifact["rows"] == expected_rows
        row = expected_rows[0]
        assert 0 < row["visible_state_fraction"] <= 1
        assert row["target"] != row["attacker"]
        assert row["attacker_view"] == ("incoming_full" if chunk_count is None else "incoming_partial")
        if chunk_count in (None, 1):
            assert row["visible_state_fraction"] == 1


@pytest.mark.asyncio
async def test_transferred_bytes_are_not_observations_until_latency_expires(tmp_path):
    from dasklearn.simulation.dpsgd.settings import DPSGDSettings
    from dasklearn.simulation.dpsgd.simulation import DPSGDSimulation
    observer = IncomingModelObserver(attackers=range(6))
    sim = DPSGDSimulation(settings(DPSGDSettings, tmp_path, algorithm="dpsgd", topology="ring",
        rounds=1, duration=2 * MICROSECONDS, network_latency_ms=2000, test_period=0))
    sim.observers.append(observer)
    sim.setup_directories()
    sim.simulate()
    assert sum(client.bw_scheduler.total_bytes_sent for client in sim.clients) > 0
    assert not observer.observations


@pytest.mark.asyncio
async def test_early_chunks_use_public_initialization_if_receiver_has_not_finished_training(tmp_path):
    from dasklearn.simulation.shatter.settings import ShatterSettings
    from dasklearn.simulation.shatter.simulation import ShatterSimulation
    observer = IncomingModelObserver(attackers=[0], rounds=[1], chunk_count=4)
    sim = ShatterSimulation(settings(ShatterSettings, tmp_path, algorithm="shatter", rounds=1,
        k=4, r=2, test_period=0, test_method="individual"))
    sim.observers.append(observer)
    sim.setup_directories()
    def speeds():
        for client in sim.clients:
            client.simulated_speed = 100 if client.index == 0 else 1
    sim.apply_traces = speeds
    sim.simulate()
    assert observer.observations
    assert any(case.baseline is None for case in observer.observations.values())
    audits = observer.attach_audits(sim, max_examples=10)
    initial = [task for task in sim.workflow_dag.tasks.values() if task.name.startswith("mia_public_initialization")]
    assert len(initial) == 1 and initial[0].available_at == 0
    sim.workflow_dag.build_task_indices()
    result = execute_local(sim)
    assert len(audits) > 0 and all(result[task.name][0] for task in audits)


def test_incoming_summary_reports_observed_pairs_and_excludes_unseen_pairs():
    import pandas as pd
    from scripts.compare_membership import summarize
    row = {"variant": "dpsgd", "partitioner": "iid", "seed": 0, "round": 1, "attacker": 0,
           "possible_pairs": 6, "visible_state_fraction": 1, "received_chunks": 1,
           **{metric: .6 for metric in ["balanced_accuracy", "roc_auc", "tpr", "fpr", "model_accuracy", "model_loss"]}}
    per_seed, summary = summarize(pd.DataFrame([row, dict(row, attacker=1)]))
    assert per_seed.observed_pairs.iloc[0] == 2
    assert per_seed.observed_pair_fraction.iloc[0] == pytest.approx(1 / 3)
    assert summary.balanced_accuracy_mean.iloc[0] == .6
