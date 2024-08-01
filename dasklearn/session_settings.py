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
    capability_traces: Optional[str] = None
    rounds: int = 10
    data_dir: str = ""
    port: int = 5555
    log_level: str = "INFO"
    torch_threads: int = 4
    dry_run: bool = False
    unit_testing: bool = False
    duration: int = 100
    gl_period: int = 10
    test_period: int = 100
    compute_graph_plot_size: int = 0
    agg: str = "default"  # default, average or age
    stop: str = "rounds"  # rounds, duration
    wait: bool = False
    el: str = "oracle"  # oracle, local
    k: int = 0
    no_weights: bool = False
    stragglers_proportion: float = 0.0  # value between 0=none and 1=all
    stragglers_ratio: float = 0.1  # value between 0=no_action and 1=no_struggle
    sample_size: int = 0

    def save_to_file(self, path: str) -> None:
        with open(path, "w") as file:
            file.write("setting,value\n")
            file.write("learning_rate,%f\n" % self.learning.learning_rate)
            file.write("momentum,%f\n" % self.learning.momentum)
            file.write("weight_decay,%f\n" % self.learning.weight_decay)
            file.write("batch_size,%d\n" % self.learning.batch_size)
            file.write("local_steps,%d\n" % self.learning.local_steps)
            file.write("algorithm,%s\n" % self.algorithm)
            file.write("seed,%d\n" % self.seed)
            file.write("dataset,%s\n" % self.dataset)
            file.write("participants,%d\n" % self.participants)
            file.write("synchronous,%s\n" % self.synchronous)
            file.write("model,%s\n" % self.model)
            file.write("alpha,%f\n" % self.alpha)
            file.write("validation_set_fraction,%f\n" % self.validation_set_fraction)
            file.write("compute_validation_loss_global_model,%s\n" % self.compute_validation_loss_global_model)
            file.write("partitioner,%s\n" % self.partitioner)
            file.write("torch_device_name,%s\n" % self.torch_device_name)
            file.write("test_interval,%d\n" % self.test_interval)
            file.write("scheduler,%s\n" % self.scheduler)
            file.write("brokers,%d\n" % self.brokers)
            file.write("rounds,%d\n" % self.rounds)
            file.write("torch_threads,%d\n" % self.torch_threads)
            file.write("dry_run,%s\n" % self.dry_run)
            file.write("unit_testing,%s\n" % self.unit_testing)
            file.write("duration,%d\n" % self.duration)
            file.write("gl_period,%d\n" % self.gl_period)
            file.write("test_period,%d\n" % self.test_period)
            file.write("compute_graph_plot_size,%d\n" % self.compute_graph_plot_size)
            file.write("agg,%s\n" % self.agg)
            file.write("stop,%s\n" % self.stop)
            file.write("wait,%s\n" % self.wait)
            file.write("el,%s\n" % self.el)
            file.write("k,%d\n" % self.k)
            file.write("no_weights,%s\n" % self.no_weights)
            file.write("stragglers,%f\n" % self.stragglers_proportion)
            file.write("stragglers_ratio,%f\n" % self.stragglers_ratio)
