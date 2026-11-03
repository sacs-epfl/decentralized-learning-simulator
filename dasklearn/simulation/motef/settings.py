from dataclasses import dataclass

from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class MoTEFSettings(SessionSettings):
    topology: str = 'ring'
    degree: int = 4
    compressor: str = 'topk'
    keep_ratio: float = .01
    consensus_stepsize: float = .1
    momentum_weight: float = .1


def add_motef_arguments(parser):
    parser.add_argument('--motef-topology', choices=['ring', 'complete', 'kreg'], default='ring')
    parser.add_argument('--motef-degree', type=int, default=4)
    parser.add_argument('--motef-compressor', choices=['topk', 'randomk', 'identity'], default='topk')
    parser.add_argument('--motef-keep-ratio', type=float, default=.01,
                        help='Fraction retained over the whole parameter vector in each compressed channel')
    parser.add_argument('--motef-consensus-stepsize', type=float, default=.1)
    parser.add_argument('--motef-momentum-weight', type=float, default=.1,
                        help='Lambda: weight of the fresh gradient in MoTEF momentum, in (0,1]')


def motef_options(args):
    return {field: getattr(args, 'motef_' + field, default) for field, default in [
        ('topology', 'ring'), ('degree', 4), ('compressor', 'topk'), ('keep_ratio', .01),
        ('consensus_stepsize', .1), ('momentum_weight', .1)]}
