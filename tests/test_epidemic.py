import math

import pytest

from dasklearn.session_settings import SessionSettings, LearningSettings
from dasklearn.simulation.epidemic.simulation import EpidemicSimulation


@pytest.fixture
def session_settings(tmpdir):
    return SessionSettings(
        algorithm="epidemic",
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
        compute_graph_plot_size=0,
    )


def sanity_check(participants: int, workflow_dag, el: str):
    k: int = math.floor(math.log2(participants))
    assert len(workflow_dag.tasks) == 2 * 5 * participants
    for task_name, task in workflow_dag.tasks.items():
        if task.name.startswith("agg_") and el == "oracle":
            assert len(task.inputs) == k + 1
        elif task.name.startswith("train_"):
            assert len(task.outputs) == k + 1
            for out_task in task.outputs:
                assert out_task.name.startswith("agg_")


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("participants", [10, 50, 100, 200])
@pytest.mark.parametrize("el", ["local", "oracle"])
async def test_epidemic(sync, seed, participants, session_settings, el):
    session_settings.synchronous = sync
    session_settings.seed = seed
    session_settings.participants = participants
    session_settings.el = el
    sim = EpidemicSimulation(session_settings)
    await sim.run()
    sanity_check(participants, sim.workflow_dag, el)
