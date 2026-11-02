from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dasklearn.simulation.conflux.settings import ConfluxSettings


@dataclass_json
@dataclass
class ConfluxV2Settings(ConfluxSettings):
    relay_count: int = 0
    # Full-participation rounds can overlap uploads with other peers' training.
    # Opt-in until the scheduling ablation has been evaluated.
    immediate_uploads: bool = False
    server_optimizer: str = "fedavg"
    server_learning_rate: float = 1.0
    server_beta1: float = 0.9
    server_beta2: float = 0.99
    server_tau: float = 0.001
    # 0 fixes owners; 1 preserves the original per-round rotation.
    reducer_epoch_rounds: int = 1
    state_bits: int = 32
    state_block_size: int = 256
    server_step_ms: float = 0.0
    state_pack_ms: float = 0.0


def add_conflux_v2_arguments(parser):
    import argparse
    parser.add_argument("--immediate-uploads", action=argparse.BooleanOptionalAction, default=False,
                        help="Upload as soon as training finishes in full-participation V2 rounds; "
                             "partial-contribution rounds retain readiness coordination")
    parser.add_argument("--server-optimizer", choices=["fedavg", "fedavgm", "fedadam"], default="fedavg")
    parser.add_argument("--server-learning-rate", type=float, default=1.0)
    parser.add_argument("--server-beta1", type=float, default=0.9)
    parser.add_argument("--server-beta2", type=float, default=0.99)
    parser.add_argument("--server-tau", type=float, default=0.001)
    parser.add_argument("--reducer-epoch-rounds", type=int, default=1,
                        help="Rounds per reducer layout; 0 fixes ownership, 1 rotates every round")
    parser.add_argument("--state-bits", type=int, choices=[8, 32], default=32,
                        help="Moment precision on ownership handoff; active owners compute in FP32")
    parser.add_argument("--state-block-size", type=int, default=256)
    parser.add_argument("--server-step-ms", type=float, default=0,
                        help="Simulated optimizer compute milliseconds per chunk")
    parser.add_argument("--state-pack-ms", type=float, default=0,
                        help="Simulated quantization compute milliseconds per migrated chunk")


def fedopt_options(args):
    return {key: getattr(args, key, ConfluxV2Settings.__dataclass_fields__[key].default) for key in (
        "server_optimizer", "server_learning_rate", "server_beta1", "server_beta2", "server_tau",
        "reducer_epoch_rounds", "state_bits", "state_block_size", "server_step_ms", "state_pack_ms")}
