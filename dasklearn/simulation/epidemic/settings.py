from dataclasses import dataclass
from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class ELSettings(SessionSettings):
    sample_size: int = 0
    el: str = "oracle"  # oracle, local
