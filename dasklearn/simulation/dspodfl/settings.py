from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class DSpodFLSettings(SessionSettings):
    degree: int = 4
    sgd_probability: float = .5
    communication_probability: float = .5
    probability_mode: str = "capacity"
    minimum_round_ms: float = 20


def add_dspodfl_arguments(parser):
    parser.add_argument('--dspod-degree', type=int, default=4)
    parser.add_argument('--dspod-sgd-probability', type=float, default=.5)
    parser.add_argument('--dspod-communication-probability', type=float, default=.5)
    parser.add_argument('--dspod-probability-mode', choices=['capacity','constant','beta'], default='capacity')


def dspodfl_options(args):
    return {name:getattr(args, flag, default) for name,flag,default in [
        ('degree','dspod_degree',4), ('sgd_probability','dspod_sgd_probability',.5),
        ('communication_probability','dspod_communication_probability',.5),
        ('probability_mode','dspod_probability_mode','capacity')]}
