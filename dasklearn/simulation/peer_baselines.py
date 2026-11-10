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
from dasklearn.simulation.relaysgd.settings import RelaySGDSettings
from dasklearn.simulation.relaysgd.simulation import RelaySGDSimulation
from dasklearn.simulation.sgp.settings import SGPSettings
from dasklearn.simulation.sgp.simulation import SGPSimulation
from dasklearn.simulation.damsgrad.settings import DAMSGradSettings
from dasklearn.simulation.fspda.settings import FSPDASettings
from dasklearn.simulation.motef.settings import MoTEFSettings
from dasklearn.simulation.damsgrad.simulation import DAMSGradSimulation
from dasklearn.simulation.fspda.simulation import FSPDASimulation
from dasklearn.simulation.motef.simulation import MoTEFSimulation


from dasklearn.simulation.motef_vr.settings import MoTEFVRSettings
from dasklearn.simulation.motef_vr.simulation import MoTEFVRSimulation


from dasklearn.simulation.plexus.settings import PlexusSettings
from dasklearn.simulation.plexus.simulation import PlexusSimulation


BASELINE_VARIANTS = ("plexus", "motef-vr", "damsgrad", "fspda", "motef", "sgp", "relaysgd", "detag", "gt-dsgd", "swift", "qgm", "dsgdm", "dsgd", "choco", "demo", "teleportation-ring", "teleportation-hypercube", "d-local-sgd-ring", "moshpit", "dspodfl")


def baseline_spec(variant, peers, sample_size, group_size=8, averaging_steps=1):
    if variant == "plexus":
        return PlexusSimulation, PlexusSettings, {"sample_size": sample_size}
    if variant == 'damsgrad':
        return DAMSGradSimulation, DAMSGradSettings, {}
    if variant == 'fspda':
        return FSPDASimulation, FSPDASettings, {}
    if variant == 'motef-vr':
        return MoTEFVRSimulation, MoTEFVRSettings, {}
    if variant == 'motef':
        return MoTEFSimulation, MoTEFSettings, {}
    if variant == 'sgp':
        return SGPSimulation, SGPSettings, {}
    if variant == 'relaysgd':
        return RelaySGDSimulation, RelaySGDSettings, {}
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
