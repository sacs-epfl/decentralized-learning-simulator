from dataclasses import dataclass

from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class GTDSGDSettings(SessionSettings):
    topology: str = 'ring'
    degree: int = 4


def add_gt_dsgd_arguments(parser):
    parser.add_argument('--gt-dsgd-topology', choices=['ring', 'complete', 'kreg'], default='ring')
    parser.add_argument('--gt-dsgd-degree', type=int, default=4)


def gt_dsgd_options(args):
    return {field: getattr(args, 'gt_dsgd_' + field, default)
            for field, default in [('topology', 'ring'), ('degree', 4)]}
