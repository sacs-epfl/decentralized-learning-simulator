from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dasklearn.session_settings import SessionSettings

@dataclass_json
@dataclass
class TeleportationSettings(SessionSettings):
    sample_size: int = 8
    topology: str = "ring"
    shuffle_tokens: bool = True
