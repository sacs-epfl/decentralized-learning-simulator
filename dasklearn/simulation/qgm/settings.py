from dataclasses import dataclass
from typing import Optional
from dataclasses_json import dataclass_json
from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class QGMSettings(SessionSettings):
    topology: str = 'ring'
    degree: int = 4
    mode: str = 'quasi'  # quasi, local, plain
    beta: float = .9
    mu: Optional[float] = None  # Paper experiments use mu=beta.
    nesterov: bool = False


def add_qgm_arguments(parser):
    parser.add_argument('--qgm-topology', choices=['ring','complete','kreg'], default='ring')
    parser.add_argument('--qgm-degree', type=int, default=4)
    parser.add_argument('--qgm-mode', choices=['quasi','local','plain'], default='quasi')
    parser.add_argument('--qgm-beta', type=float, default=.9)
    parser.add_argument('--qgm-mu', type=float, default=None)
    parser.add_argument('--qgm-nesterov', action='store_true')


def qgm_options(args):
    return {field:getattr(args,'qgm_'+field,default) for field,default in [
        ('topology','ring'),('degree',4),('mode','quasi'),('beta',.9),('mu',None),('nesterov',False)]}
