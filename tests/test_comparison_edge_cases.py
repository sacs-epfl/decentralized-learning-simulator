import csv
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from dasklearn import functions
from dasklearn.session_settings import LearningSettings, SessionSettings
from dasklearn.simulation.teleportation.settings import TeleportationSettings
from dasklearn.simulation.teleportation.simulation import TeleportationSimulation
from dasklearn.simulation.events import INCOMING_PAYLOAD
from dasklearn.simulation.bandwidth_scheduler import BWScheduler
from dasklearn.simulation.shatter.client import ShatterClient
from dasklearn.simulation.shatter.round import Round
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS


@pytest.mark.asyncio
async def test_sampled_clients_cannot_train_two_rounds_simultaneously(tmp_path):
    settings = TeleportationSettings(algorithm="teleportation", seed=1, work_dir=str(tmp_path),
        dataset="synthetic", learning=LearningSettings(.1, 0, 0, 16, 2), participants=6,
        sample_size=4, duration=5 * MICROSECONDS,
        rounds=0, unit_testing=True, test_interval=0, test_period=0)
    sim = TeleportationSimulation(settings)
    sim.setup_directories()
    def profile():
        for peer in sim.clients:
            peer.simulated_speed = 200 if peer.index == 0 else 1
    sim.apply_traces = profile
    sim.simulate()
    starts = defaultdict(list)
    for task in sim.workflow_dag.tasks.values():
        if task.func == "train":
            starts[task.data["peer"]].append(task.data["time"])
    for peer, times in starts.items():
        duration = sim.clients[peer].get_train_time()
        ordered = sorted(times)
        assert all(right - left >= duration for left, right in zip(ordered, ordered[1:]))
    assert len(starts) > 1


@pytest.mark.asyncio
async def test_shatter_same_host_chunks_are_local(tmp_path):
    from dasklearn.simulation.shatter.settings import ShatterSettings
    from dasklearn.simulation.simulation import Simulation
    from helpers import settings
    sim = Simulation(settings(ShatterSettings, tmp_path, participants=1, k=1))
    sim.model_size = 100
    client = ShatterClient(sim, 0)
    sim.clients = [client]
    client.create_compute_task('chunk', {'model': None, 'n': 1}, name='chunked_model')
    round_info = Round(1)
    client.send_chunk(round_info, "chunked_model", 0, 0)
    assert sim.events[0][2].action == INCOMING_PAYLOAD
    assert client.bw_scheduler.total_bytes_sent == 0


@pytest.mark.asyncio
async def test_local_steps_continue_across_epoch_boundary(tmp_path):
    settings = TeleportationSettings(algorithm="teleportation", seed=1, work_dir=str(tmp_path),
        dataset="synthetic", learning=LearningSettings(.1, 0, 0, 128, 5), participants=2,
        sample_size=2)
    from dasklearn.datasets import create_dataset
    from dasklearn.model_trainer import ModelTrainer
    trainer = ModelTrainer(create_dataset(settings), settings, 0)
    model = functions.initialize(settings, {})[0]
    result = trainer.train(model, 5, seed=1)
    assert result["samples"] == 5 * 128


def test_sampler_is_reproducible_seeded_and_shared_between_variants():
    from dasklearn.simulation.conflux.sample_manager import SampleManager
    peers = list(range(100))
    first = SampleManager.get_sample(2, peers, 10, seed=1)
    assert first == SampleManager.get_sample(2, list(reversed(peers)), 10, seed=1)
    assert first != SampleManager.get_sample(2, peers, 10, seed=2)


def test_legacy_coalescing_does_not_deduplicate_overlapping_averages():
    from dasklearn.simulation.conflux.client import ConfluxClient
    # Possessing mean(A,B) and mean(B,C) does not provide mean(A,B,C).
    chunks = [(0, frozenset({"A", "B"})), (0, frozenset({"B", "C"}))]
    assert ConfluxClient.coalesce_disjoint_chunks(chunks) == {"A", "B"}
    # A separate C contribution makes the exact union constructible.
    chunks.append((0, frozenset({"C"})))
    assert ConfluxClient.coalesce_disjoint_chunks(chunks) == {"A", "B", "C"}


@pytest.mark.parametrize("partitioner", ["iid", "dirichlet"])
def test_real_digits_partitions_are_disjoint(partitioner):
    from dasklearn.datasets.digits import DigitsDataset
    settings = SessionSettings(algorithm="test", work_dir="", dataset="digits", seed=1, participants=24,
                               alpha=.3, partitioner=partitioner, learning=LearningSettings(.1, 0, 0, 8, 1))
    dataset = DigitsDataset(settings)
    assert all(len(partition) >= 8 for partition in dataset.partitions)
    assert sum(len(partition) for partition in dataset.partitions) + len(dataset.test) == 1797
    fingerprints = []
    for partition in [*dataset.partitions, dataset.test]:
        fingerprints.extend((row.numpy().tobytes(), int(label)) for row, label in zip(partition.features, partition.labels))
    assert len(set(fingerprints)) == 1797
