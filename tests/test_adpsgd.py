import pytest

from dasklearn.session_settings import SessionSettings, LearningSettings
from dasklearn.simulation.adpsgd.simulation import ADPSGDSimulation


@pytest.fixture
def session_settings(tmpdir):
    return SessionSettings(
        algorithm="adpsgd",
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
    compute_gradient_init = 0
    gradient_update_init = 0
    for task_name, task in workflow_dag.tasks.items():
        if task.func == "compute_gradient":
            assert len(task.inputs) <= 1
            if len(task.inputs) == 0:
                compute_gradient_init += 1
            else:
                assert task.inputs[0].func in ["aggregate", "gradient_update"]
        elif task.func == "gradient_update":
            assert len(task.inputs) == 2 or len(task.inputs) == 1
            if len(task.inputs) == 1:
                gradient_update_init += 1
        elif task.func == "aggregate":
            assert len(task.inputs) == 2
            assert "gradient_update" in list(map(lambda x: x.func, task.inputs))
        elif task.func == "test":
            assert len(task.inputs) == 1
            assert len(task.outputs) == 0
        else:
            assert False
    assert compute_gradient_init == participants
    assert gradient_update_init == participants


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("participants", [10, 50, 100, 200])
@pytest.mark.parametrize("duration", [20, 100, 500, 1000])
async def test_adpsgd(seed, participants, duration, session_settings):
    session_settings.duration = duration
    session_settings.seed = seed
    session_settings.participants = participants
    sim = ADPSGDSimulation(session_settings)
    await sim.run()
    sanity_check(session_settings.participants, sim.workflow_dag)
