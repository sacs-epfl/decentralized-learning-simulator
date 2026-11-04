from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dasklearn.simulation.motef.settings import MoTEFSettings


@dataclass_json
@dataclass
class MoTEFVRSettings(MoTEFSettings):
    """Algorithm 2 uses the shared --motef-* options; momentum_weight is lambda."""
