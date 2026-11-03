from dataclasses import dataclass

from dataclasses_json import dataclass_json

from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class RelaySGDSettings(SessionSettings):
    topology: str = 'balanced'
    normalization: str = 'counts'  # Paper Algorithm 1; 'initial' matches the reference's world_size mode.


def add_relaysgd_arguments(parser):
    parser.add_argument('--relaysgd-topology', choices=['balanced', 'path', 'star'], default='balanced')
    parser.add_argument('--relaysgd-normalization', choices=['counts', 'initial'], default='counts',
                        help='Startup: average received contributions, or pad missing ones with the initial model')


def relaysgd_options(args):
    return {field: getattr(args, 'relaysgd_' + field, default)
            for field, default in [('topology', 'balanced'), ('normalization', 'counts')]}
