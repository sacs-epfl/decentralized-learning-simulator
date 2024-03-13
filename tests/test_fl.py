import pytest

from dasklearn.session_settings import SessionSettings, LearningSettings
from dasklearn.simulation.fl.simulation import FLSimulation


@pytest.fixture
def session_settings(tmpdir):
    return SessionSettings(
        algorithm="fl",
        seed=3,
        work_dir=tmpdir,
        dataset="cifar10",
        learning=LearningSettings(batch_size=0, learning_rate=0, momentum=0, weight_decay=0, local_steps=0),
        participants=10,
        rounds=5,
        dry_run=True,
        brokers=1,
        unit_testing=True,
        capability_traces="data/client_device_capacity",
    )


def sanity_check(participants: int, workflow_dag):
    assert len(workflow_dag.tasks) == participants * 5 + 5
    for task_name, task in workflow_dag.tasks.items():
        if task.name.startswith("agg_"):
            assert len(task.inputs) == participants
        elif task.name.startswith("train_"):
            assert len(task.outputs) == 1
            assert task.outputs[0].name.startswith("agg_")


@pytest.mark.asyncio
@pytest.mark.parametrize("participants", [10, 20, 50, 100])
async def test_fl(participants, session_settings):
    session_settings.participants = participants
    sim = FLSimulation(session_settings)
    await sim.run()
    sanity_check(participants, sim.workflow_dag)
