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


def add_conflux_v2_arguments(parser):
    import argparse
    parser.add_argument("--immediate-uploads", action=argparse.BooleanOptionalAction, default=False,
                        help="Upload as soon as training finishes in full-participation V2 rounds; "
                             "partial-contribution rounds retain readiness coordination")
