from dataclasses import dataclass
from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class DPSGDSettings(SessionSettings):
    topology: str = "kreg"  # ring, kreg
    k: int = 0  # Default value of 0 means log2(participants)
    el: str = "oracle"  # oracle, local
