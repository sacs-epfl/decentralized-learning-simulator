from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class DeTAGSettings(SessionSettings):
    topology: str = 'ring'
    degree: int = 4
    phase_length: int = 2
    eta: Optional[float] = None  # Algorithm 3's spectral choice, or an explicit tuned value.


def add_detag_arguments(parser):
    parser.add_argument('--detag-topology', choices=['ring', 'complete', 'kreg'], default='ring')
    parser.add_argument('--detag-degree', type=int, default=4)
    parser.add_argument('--detag-phase-length', type=int, default=2,
                        help='R overlapped gradient/gossip iterations at fixed parameters per phase')
    parser.add_argument('--detag-eta', type=float, default=None,
                        help='Gossip acceleration in [0,1); default derives it from the mixing spectrum')


def detag_options(args):
    return {field: getattr(args, 'detag_' + field, default) for field, default in [
        ('topology', 'ring'), ('degree', 4), ('phase_length', 2), ('eta', None)]}
