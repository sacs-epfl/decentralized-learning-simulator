from dasklearn.session_settings import LearningSettings
from dasklearn.util import MICROSECONDS


def settings(cls, tmp_path, **kwargs):
    values = dict(algorithm="test", seed=3, work_dir=str(tmp_path), dataset="synthetic", model="mlp",
                  learning=LearningSettings(0.1, 0, 0, 16, 2), participants=6, rounds=3,
                  duration=30 * MICROSECONDS, test_interval=0, test_period=2 * MICROSECONDS,
                  test_method="snapshot", dry_run=True, unit_testing=True, torch_threads=1)
    values.update(kwargs)
    return cls(**values)


def simulate(cls, configuration):
    sim = cls(configuration)
    sim.setup_directories()
    sim.simulate()
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    return sim
