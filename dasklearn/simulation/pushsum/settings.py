from dataclasses import dataclass
from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class PushSumSettings(SessionSettings):
    sample_size: int = 0
    chunks_in_sample: int = 10
    push_sum_duration: float = 50
