"""Named, explicitly configured literature baselines for comparison runners."""
from dasklearn.simulation.teleportation.settings import TeleportationSettings
from dasklearn.simulation.teleportation.simulation import TeleportationSimulation
from dasklearn.simulation.moshpit.settings import MoshpitSettings
from dasklearn.simulation.moshpit.simulation import MoshpitSimulation
from dasklearn.simulation.dspodfl.settings import DSpodFLSettings
from dasklearn.simulation.dspodfl.simulation import DSpodFLSimulation


from dasklearn.simulation.demo.settings import DeMoSettings
from dasklearn.simulation.demo.simulation import DeMoSimulation
from dasklearn.simulation.choco.settings import ChocoSettings
from dasklearn.simulation.choco.simulation import ChocoSimulation
from dasklearn.simulation.qgm.settings import QGMSettings
from dasklearn.simulation.qgm.simulation import QGMSimulation
from dasklearn.simulation.swift.settings import SwiftSettings
from dasklearn.simulation.swift.simulation import SwiftSimulation
from dasklearn.simulation.gt_dsgd.settings import GTDSGDSettings
from dasklearn.simulation.gt_dsgd.simulation import GTDSGDSimulation
from dasklearn.simulation.detag.settings import DeTAGSettings
from dasklearn.simulation.detag.simulation import DeTAGSimulation


BASELINE_VARIANTS = ("detag", "gt-dsgd", "swift", "qgm", "dsgdm", "dsgd", "choco", "demo", "teleportation-ring", "teleportation-hypercube", "d-local-sgd-ring", "moshpit", "dspodfl")


def baseline_spec(variant, peers, sample_size, group_size=8, averaging_steps=1):
    if variant == 'detag':
        return DeTAGSimulation, DeTAGSettings, {}
    if variant == 'gt-dsgd':
        return GTDSGDSimulation, GTDSGDSettings, {}
    if variant == 'swift':
        return SwiftSimulation, SwiftSettings, {}
    if variant in {'qgm','dsgdm','dsgd'}:
        return QGMSimulation, QGMSettings, {'mode':{'qgm':'quasi','dsgdm':'local','dsgd':'plain'}[variant]}
    if variant == 'choco':
        return ChocoSimulation, ChocoSettings, {}
    if variant == 'demo':
        return DeMoSimulation, DeMoSettings, {}
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
