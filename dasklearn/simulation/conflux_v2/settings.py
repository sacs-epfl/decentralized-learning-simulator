from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dasklearn.simulation.conflux.settings import ConfluxSettings


@dataclass_json
@dataclass
class ConfluxV2Settings(ConfluxSettings):
    relay_count: int = 0
