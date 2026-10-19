from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class MoshpitSettings(SessionSettings):
    group_size: int = 8
    averaging_steps: int = 1
    load_balancing: bool = True
