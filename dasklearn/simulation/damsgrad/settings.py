from dataclasses import dataclass

from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class DAMSGradSettings(SessionSettings):
    topology: str = 'ring'
    degree: int = 4
    beta1: float = .9
    beta2: float = .999
    epsilon: float = 1e-6


def add_damsgrad_arguments(parser):
    parser.add_argument('--damsgrad-topology', choices=['ring', 'complete', 'kreg'], default='ring')
    parser.add_argument('--damsgrad-degree', type=int, default=4)
    parser.add_argument('--damsgrad-beta1', type=float, default=.9, help='Local first-moment decay, in [0,1)')
    parser.add_argument('--damsgrad-beta2', type=float, default=.999, help='Local second-moment decay, in [0,1)')
    parser.add_argument('--damsgrad-epsilon', type=float, default=1e-6,
                        help='Positive floor in squared-gradient units: denominator sqrt(max(mixed_u,epsilon))')


def damsgrad_options(args):
    return {field: getattr(args, 'damsgrad_' + field, default) for field, default in [
        ('topology', 'ring'), ('degree', 4), ('beta1', .9), ('beta2', .999), ('epsilon', 1e-6)]}
