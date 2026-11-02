from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dasklearn.simulation.conflux_v2.settings import ConfluxV2Settings


@dataclass_json
@dataclass
class ShardedPSSettings(ConfluxV2Settings):
    reducer_epoch_rounds: int = 0
    immediate_uploads: bool = True


@dataclass_json
@dataclass
class AutoDirectSettings(ShardedPSSettings):
    """Chunk/reducer flags are ignored: the measured planner chooses both."""
