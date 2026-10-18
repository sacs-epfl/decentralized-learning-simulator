import numpy as np
import pandas as pd
import pytest

from dasklearn.datasets.cifar10 import partition_indices
from dasklearn.tasks.local import execute_local
from dasklearn.simulation.conflux_v2.settings import ConfluxV2Settings
from dasklearn.simulation.conflux_v2.simulation import ConfluxV2Simulation
from test_conflux_comparison import settings, simulate


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
        sim = simulate(ConfluxV2Simulation, settings(ConfluxV2Settings, tmp_path / str(retain),
            sample_size=3, chunks_in_sample=3))
        status = []
        result = execute_local(sim, retain_results=retain, progress=status.append)
        if not retain:
            assert result == {}
        assert status[-1]["completed"] == len(sim.workflow_dag.tasks)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + "/accuracies.csv").sort_values(["time", "peer"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(*frames)
