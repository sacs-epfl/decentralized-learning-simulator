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
    lora: bool = False
    base_model: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer: Optional[str] = None
    lora_rank: int = 8
    lora_alpha: float = 16
    lora_dropout: float = 0.0
    lora_target_modules: Optional[str] = None
    lora_task: str = "sequence-classification"
    num_labels: int = 2
    max_sequence_length: int = 128
    text_dataset: str = "toy-text"
    text_dataset_config: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    alpha: float = 1
    dataset_base_path: Optional[str] = None
    validation_set_fraction: float = 0
    compute_validation_loss_global_model: bool = False
    partitioner: str = "iid"  # iid, shards or dirichlet
    gradient_aggregation: GradientAggregationMethod = GradientAggregationMethod.FEDAVG
    torch_device_name: str = "cpu"
    test_interval: int = 0
    scheduler: Optional[str] = None
    brokers: int = 1
    traces: str = "none"
    churn: str = "none"
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
    test_method: str = "individual"
    compute_graph_plot_size: int = 0
    stop: str = "rounds"  # rounds, duration
    wait: bool = False
    stragglers_proportion: float = 0.0  # value between 0=none and 1=all
    stragglers_ratio: float = 0.1  # value between 0=no_action and 1=no_struggle
    profile: bool = False
    log_bandwidth_utilization: bool = False
    from_dir: Optional[str] = None
    dag_checkpoint_interval: int = 0
    control_message_bytes: int = 64  # Header, in addition to the serialized control payload.
    network_latency_ms: float = 0
    evaluation_split: str = "test"
    evaluation_examples: int = 0
    release_cuda_cache: bool = False  # Opt in when sharing a GPU with other applications.
    model_size_bytes: Optional[int] = None  # Explicit payload size for paired/replayed experiments.
    gpu_residency: bool = True
    training_batching: bool = True
    training_batch_size: int = 2
    gpu_state_cache_mb: int = 256
    feedback_execution: str = "local"  # Local or persistent broker RPCs for adaptive runs.
    feedback_replay: Optional[str] = None  # Trusted feedback.pkl from the same source/settings.
