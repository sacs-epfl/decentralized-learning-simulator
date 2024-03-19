from collections import Counter

import pytest

from dasklearn.session_settings import SessionSettings, LearningSettings
from dasklearn.simulation.gossip.simulation import GossipSimulation


@pytest.fixture
def session_settings(tmpdir):
    return SessionSettings(
        algorithm="gossip",
        seed=3,
        work_dir=tmpdir,
        dataset="cifar10",
        learning=LearningSettings(batch_size=0, learning_rate=0, momentum=0, weight_decay=0, local_steps=0),
        participants=10,
        duration=100,
        period=10,
        test_period=60,
        dry_run=True,
        brokers=1,
        unit_testing=True,
        capability_traces="data/client_device_capacity",
    )


def sanity_check(workflow_dag):
    task_counter = Counter()
    for task in workflow_dag.tasks.values():
        if task.func == "aggregate":
            assert len(task.inputs) == 2
        elif task.func == "train":
            assert len(task.inputs) == 1 or task.data["time"] == 0
        elif task.func == "test":
            assert len(task.inputs) == 1
        task_counter[task.func] += 1
    assert sum(task_counter.values()) == len(workflow_dag.tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("participants", [10, 50, 100, 200])
async def test_gossip(seed, participants, session_settings):
    session_settings.seed = seed
    session_settings.participants = participants
    sim = GossipSimulation(session_settings)
    await sim.run()
    sanity_check(sim.workflow_dag)