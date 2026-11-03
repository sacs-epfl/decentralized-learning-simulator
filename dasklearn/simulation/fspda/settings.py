from dataclasses import dataclass

from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class FSPDASettings(SessionSettings):
    topology: str = 'ring'
    degree: int = 4
    sampling: str = 'bernoulli'
    edge_probability: float = .5
    keep_ratio: float = .1
    consensus_stepsize: float = .1
    dual_weight: float = .01
    dual_stepsize: float = .1


def add_fspda_arguments(parser):
    parser.add_argument('--fspda-topology', choices=['ring', 'complete', 'kreg'], default='ring')
    parser.add_argument('--fspda-degree', type=int, default=4)
    parser.add_argument('--fspda-sampling', choices=['bernoulli', 'one-edge'], default='bernoulli')
    parser.add_argument('--fspda-edge-probability', type=float, default=.5,
                        help='Independent undirected edge activation probability (Bernoulli mode only)')
    parser.add_argument('--fspda-keep-ratio', type=float, default=.1,
                        help='Shared random coordinate fraction per edge; unscaled projection, 1 is dense')
    parser.add_argument('--fspda-consensus-stepsize', type=float, default=.1, help='Gamma: unnormalized Laplacian weight')
    parser.add_argument('--fspda-dual-weight', type=float, default=.01, help='Eta: primal weight of the persistent dual')
    parser.add_argument('--fspda-dual-stepsize', type=float, default=.1, help='Beta: dual ascent stepsize')


def fspda_options(args):
    return {field: getattr(args, 'fspda_' + field, default) for field, default in [
        ('topology', 'ring'), ('degree', 4), ('sampling', 'bernoulli'), ('edge_probability', .5),
        ('keep_ratio', .1), ('consensus_stepsize', .1), ('dual_weight', .01), ('dual_stepsize', .1)]}
