from dataclasses import dataclass
from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class ShatterSettings(SessionSettings):
    k: int = 0  # Number of virtual nodes to spawn per real node
    r: int = 0  # The degree of the virtual topology
