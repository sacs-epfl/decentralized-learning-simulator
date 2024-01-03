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
    model: Optional[str] = None
    alpha: float = 1
    dataset_base_path: Optional[str] = None
    partitioner: str = "iid"  # iid, shards or dirichlet
    gradient_aggregation: GradientAggregationMethod = GradientAggregationMethod.FEDAVG
    torch_device_name: str = "cpu"
    test_interval: int = 0
    scheduler: Optional[str] = None
    brokers: Optional[int] = None
    capability_traces: Optional[str] = None
    rounds: int = 10
    data_dir: str = ""
    port: int = 5555
