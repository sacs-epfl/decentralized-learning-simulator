import pytest

from dasklearn.session_settings import SessionSettings, LearningSettings
from dasklearn.simulation.subset.simulation import SubsetDLSimulation


@pytest.fixture
def session_settings(tmpdir):
    return SessionSettings(
        algorithm="subset",
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


def sanity_check(sample_size: int, workflow_dag):
    assert len(workflow_dag.tasks) == sample_size * 5 * 2
    for task_name, task in workflow_dag.tasks.items():
        if task.name.startswith("agg_"):
            assert len(task.inputs) == 2
        elif task.name.startswith("train_"):
            assert len(task.outputs) == 2
            for out_task in task.outputs:
                assert out_task.name.startswith("agg_")


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("sample_size", [5, 10])
@pytest.mark.parametrize("participants", [20, 50, 100])
async def test_subset_learning(sync, seed, sample_size, participants, session_settings):
    session_settings.synchronous = sync
    session_settings.seed = seed
    session_settings.participants = participants
    sim = SubsetDLSimulation(session_settings, sample_size)
    await sim.run()
    sanity_check(sample_size, sim.workflow_dag)
