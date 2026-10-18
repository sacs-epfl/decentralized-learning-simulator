"""Named, explicitly configured literature baselines for comparison runners."""
from dasklearn.simulation.teleportation.settings import TeleportationSettings
from dasklearn.simulation.teleportation.simulation import TeleportationSimulation
from dasklearn.simulation.moshpit.settings import MoshpitSettings
from dasklearn.simulation.moshpit.simulation import MoshpitSimulation
from dasklearn.simulation.dspodfl.settings import DSpodFLSettings
from dasklearn.simulation.dspodfl.simulation import DSpodFLSimulation


BASELINE_VARIANTS = ("teleportation-ring", "teleportation-hypercube", "d-local-sgd-ring", "moshpit", "dspodfl")


def baseline_spec(variant, peers, sample_size, group_size=8, averaging_steps=1):
    if variant == 'dspodfl':
        return DSpodFLSimulation,DSpodFLSettings,{}
    if variant == "moshpit":
        return MoshpitSimulation, MoshpitSettings, {
            "group_size": group_size, "averaging_steps": averaging_steps}
    if variant in BASELINE_VARIANTS:
        return TeleportationSimulation, TeleportationSettings, {
            "sample_size": peers if variant == "d-local-sgd-ring" else sample_size,
            "topology": "hypercube" if variant == "teleportation-hypercube" else "ring",
            "shuffle_tokens": variant != "d-local-sgd-ring"}
    raise ValueError("Unknown baseline: " + variant)
