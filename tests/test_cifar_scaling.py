from dasklearn.simulation.moshpit.simulation import MoshpitSimulation
from dasklearn.simulation.moshpit.settings import MoshpitSettings
import numpy as np
import pandas as pd
import pytest

from dasklearn.datasets.cifar10 import partition_indices
from dasklearn.tasks.local import execute_local
from helpers import settings, simulate


@pytest.mark.parametrize("partitioner", ["iid", "dirichlet"])
def test_cifar_partition_is_complete_disjoint_and_reproducible(partitioner):
    labels = np.repeat(np.arange(10), 500)
    parts = partition_indices(labels, 128, partitioner, .3, 42)
    assert parts == partition_indices(labels, 128, partitioner, .3, 42)
    assert min(map(len, parts)) >= 8
    assert sorted(i for part in parts for i in part) == list(range(len(labels)))


@pytest.mark.asyncio
async def test_streaming_executor_preserves_every_snapshot(tmp_path):
    frames = []
    for retain in [True, False]:
        sim = simulate(MoshpitSimulation, settings(MoshpitSettings, tmp_path / str(retain),
            group_size=3))
        status = []
        result = execute_local(sim, retain_results=retain, progress=status.append)
        if not retain:
            assert result == {}
        assert status[-1]["completed"] == len(sim.workflow_dag.tasks)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + "/accuracies.csv").sort_values(["time", "peer"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(*frames)
