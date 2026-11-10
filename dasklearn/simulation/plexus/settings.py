from dataclasses import dataclass

from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class PlexusSettings(SessionSettings):
    sample_size: int = 10
    success_fraction: float = 1.0
