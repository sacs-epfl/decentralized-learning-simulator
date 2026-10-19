from dataclasses import dataclass
from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class ChocoSettings(SessionSettings):
    topology: str = 'ring'
    degree: int = 4
    compressor: str = 'topk'
    keep_ratio: float = .01
    consensus_stepsize: float = .1


def add_choco_arguments(parser):
    parser.add_argument('--choco-topology', choices=['ring', 'complete', 'kreg'], default='ring')
    parser.add_argument('--choco-degree', type=int, default=4)
    parser.add_argument('--choco-compressor', choices=['topk', 'randomk', 'identity'], default='topk')
    parser.add_argument('--choco-keep-ratio', type=float, default=.01,
                        help='Fraction retained per trainable parameter tensor, not the fraction discarded')
    parser.add_argument('--choco-consensus-stepsize', type=float, default=.1)


def choco_options(args):
    return {field: getattr(args, 'choco_' + field, default) for field, default in [
        ('topology', 'ring'), ('degree', 4), ('compressor', 'topk'),
        ('keep_ratio', .01), ('consensus_stepsize', .1)]}
