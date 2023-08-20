import os
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json


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
    work_dir: str
    dataset: str
    learning: LearningSettings
    participants: int
    target_participants: int
    model: Optional[str] = None
    alpha: float = 1
    dataset_base_path: str = None
    partitioner: str = "iid"  # iid, shards or dirichlet
    train_device_name: str = "cpu"


def dump_settings(settings: SessionSettings):
    """
    Dump the session settings if they do not exist yet.
    """
    settings_file_path = os.path.join(settings.work_dir, "settings.json")
    if not os.path.exists(settings_file_path):
        with open(settings_file_path, "w") as settings_file:
            settings_file.write(settings.to_json())
