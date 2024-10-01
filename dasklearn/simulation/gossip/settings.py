from dataclasses import dataclass
from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class GLSettings(SessionSettings):
    gl_period: int = 10
    agg: str = "default"  # default, average or age
