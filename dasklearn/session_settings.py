from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json

from dasklearn.gradient_aggregation import GradientAggregationMethod


@dataclass
class LearningSettings:
    """
    Settings related to the learning process.
    """
    learning_rate: float
    momentum: float
    weight_decay: float
    batch_size: int
    local_steps: int


@dataclass_json
@dataclass
class SessionSettings:
    """
    All settings related to a training session.
    """
    algorithm: str
    seed: int
    work_dir: str
    dataset: str
    learning: LearningSettings
    participants: int
    synchronous: bool = False
    model: Optional[str] = None
    alpha: float = 1
    dataset_base_path: Optional[str] = None
    validation_set_fraction: float = 0
    compute_validation_loss_global_model: bool = False
    partitioner: str = "iid"  # iid, shards or dirichlet
    gradient_aggregation: GradientAggregationMethod = GradientAggregationMethod.FEDAVG
    torch_device_name: str = "cpu"
    test_interval: int = 0
    scheduler: Optional[str] = None
    brokers: Optional[int] = None
    traces: str = "none"
    min_bandwidth: int = 0
    rounds: int = 10
    data_dir: str = ""
    port: int = 5555
    log_level: str = "INFO"
    torch_threads: int = 4
    dry_run: bool = False
    unit_testing: bool = False
    duration: int = 100
    test_period: int = 100
    compute_graph_plot_size: int = 0
    stop: str = "rounds"  # rounds, duration
    wait: bool = False
    stragglers_proportion: float = 0.0  # value between 0=none and 1=all
    stragglers_ratio: float = 0.1  # value between 0=no_action and 1=no_struggle
    profile: bool = False
