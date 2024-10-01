from dataclasses import dataclass
from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class LuborSettings(SessionSettings):
    k: int = 0
    no_weights: bool = False
