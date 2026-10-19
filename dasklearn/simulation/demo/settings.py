from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dasklearn.session_settings import SessionSettings


@dataclass_json
@dataclass
class DeMoSettings(SessionSettings):
    compression_decay: float = .999
    compression_chunk: int = 64
    compression_topk: int = 32


def add_demo_arguments(parser):
    parser.add_argument('--demo-decay', type=float, default=.999)
    parser.add_argument('--demo-chunk', type=int, default=64)
    parser.add_argument('--demo-topk', type=int, default=32)


def demo_options(args):
    return {name: getattr(args, flag, default) for name, flag, default in [
        ('compression_decay', 'demo_decay', .999), ('compression_chunk', 'demo_chunk', 64),
        ('compression_topk', 'demo_topk', 32)]}
