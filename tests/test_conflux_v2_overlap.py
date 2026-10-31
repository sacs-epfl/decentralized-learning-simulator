"""Protocol equivalence and overlap checks for eager full-sample uploads."""
import argparse

import pytest
import torch

from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.simulation.conflux_v2.settings import ConfluxV2Settings, add_conflux_v2_arguments
from dasklearn.simulation.conflux_v2.simulation import ConfluxV2Simulation
from dasklearn.tasks.local import execute_local
from helpers import settings


def run_protocol(tmp_path, eager, chunks, fraction=1, skew=True, sample_size=4):
    config = settings(ConfluxV2Settings, tmp_path, participants=8, sample_size=sample_size,
                      chunks_in_sample=chunks, relay_count=chunks, immediate_uploads=eager,
                      success_fraction=fraction, rounds=3, model_size_bytes=12001,
                      test_period=0, network_latency_ms=10)
    sim = ConfluxV2Simulation(config)
    sim.setup_directories()
    uploads = []

    def profile():
        for client in sim.clients:
            client.bw_scheduler.bw_limit = 100_000
            # Every sample has a slow participant, independently of relays.
            client.simulated_speed = 20 * (client.index + 1) if skew else 20
            original = client.send_chunk

            def recorded(recipient, model, round_nr, chunk_idx, phase, original=original, peer=client.index):
                if phase == "upload":
                    uploads.append((round_nr, peer, sim.current_time))
                original(recipient, model, round_nr, chunk_idx, phase)

            client.send_chunk = recorded

    sim.apply_traces = profile
    sim.simulate()
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    results = execute_local(sim)
    models = {(task.data["round"], task.data["peer"]):
              ChunkManager.get_flat_params(results[task.name][0]).clone()
              for task in sim.workflow_dag.tasks.values() if task.func == "reconstruct_from_chunks"}
    assert len(models) == 3 * sample_size
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == sum(
        c.bw_scheduler.total_bytes_received for c in sim.clients)
    return sim, models, uploads


@pytest.mark.asyncio
@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("chunks", [2, 4, 8])
async def test_immediate_uploads_preserve_every_round_and_overlap_training(tmp_path, skew, chunks):
    reference, expected, _ = run_protocol(tmp_path / "reference", False, chunks, skew=skew)
    eager, actual, uploads = run_protocol(tmp_path / "eager", True, chunks, skew=skew)
    assert eager.contributor_sets == reference.contributor_sets
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key])
    # Every reducer waits for precisely the common set, even if it is not a trainer.
    for client in eager.clients:
        for r, state in client.rounds.items():
            for idx in state["reduced"]:
                assert sorted(state["uploads"][idx]) == eager.contributor_sets[r]
    trains = [t for t in eager.workflow_dag.tasks.values() if t.func == "train" and t.data["round"] == 1]
    last_finish = max(t.data["time"] + eager.clients[t.data["peer"]].get_train_time() for t in trains)
    if skew:
        assert min(time for r, _, time in uploads if r == 1) < last_finish
    assert max(t for _, _, t in eager.completed_rounds) < max(t for _, _, t in reference.completed_rounds)


@pytest.mark.asyncio
@pytest.mark.parametrize("fraction", [.5, .99])
async def test_partial_rounds_retain_identical_ready_set_timing_and_bytes(tmp_path, fraction):
    reference, expected, _ = run_protocol(tmp_path / "reference", False, 4, fraction)
    eager, actual, _ = run_protocol(tmp_path / "eager", True, 4, fraction)
    assert eager.contributor_sets == reference.contributor_sets
    assert eager.completed_rounds == reference.completed_rounds
    assert sum(c.bw_scheduler.total_bytes_sent for c in eager.clients) == sum(
        c.bw_scheduler.total_bytes_sent for c in reference.clients)
    for key in expected:
        assert torch.equal(actual[key], expected[key])


@pytest.mark.asyncio
async def test_wider_placement_preserves_common_average_and_uses_more_reducers(tmp_path):
    narrow, expected, _ = run_protocol(tmp_path / "narrow", True, 2)
    wide, actual, _ = run_protocol(tmp_path / "wide", True, 4)
    assert len(set(narrow.clients[0].layout(1)[1])) == 2
    assert len(set(wide.clients[0].layout(1)[1])) == 4
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-7)


@pytest.mark.asyncio
async def test_single_contributor_rounds_complete(tmp_path):
    sim, _, _ = run_protocol(tmp_path, True, 4, sample_size=1)
    # A one-contributor average still reconstructs exactly and advances all rounds.
    assert all(len(peers) == 1 for peers in sim.contributor_sets.values())


def test_immediate_uploads_is_explicit_and_reversible():
    parser = argparse.ArgumentParser()
    add_conflux_v2_arguments(parser)
    assert not parser.parse_args([]).immediate_uploads
    assert parser.parse_args(["--immediate-uploads"]).immediate_uploads
    assert not parser.parse_args(["--immediate-uploads", "--no-immediate-uploads"]).immediate_uploads
