from dataclasses import dataclass
from typing import Optional, List

from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class SwiftSettings(SessionSettings):
    topology: str = 'ring'
    degree: int = 4
    averaging_interval: int = 1  # C_s with s = averaging_interval - 1
    influence: Optional[List[float]] = None  # Uniform CIS, as in the paper experiments.


def add_swift_arguments(parser):
    parser.add_argument('--swift-topology', choices=['ring', 'complete', 'kreg'], default='ring')
    parser.add_argument('--swift-degree', type=int, default=4)
    parser.add_argument('--swift-averaging-interval', type=int, default=1,
                        help='Average every H single-minibatch SWIFT updates; broadcast every update')
    parser.add_argument('--swift-influence', type=float, nargs='+', default=None,
                        help='Positive client influence scores in peer order (normalized); default uniform')


def swift_options(args):
    return {field: getattr(args, 'swift_' + field, default) for field, default in [
        ('topology', 'ring'), ('degree', 4), ('averaging_interval', 1), ('influence', None)]}
