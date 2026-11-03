from dataclasses import dataclass

from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class SGPSettings(SessionSettings):
    topology: str = 'exponential'


def add_sgp_arguments(parser):
    parser.add_argument('--sgp-topology', choices=['exponential', 'ring', 'irregular', 'complete'],
                        default='exponential', help='Directed SGP graph; exponential rotates forward power-of-two hops')


def sgp_options(args):
    return {'topology': getattr(args, 'sgp_topology', 'exponential')}
