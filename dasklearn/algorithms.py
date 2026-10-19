"""Algorithm registrations shared by the CLI and distributed brokers.

Keep imports lazy: listing algorithms does not load datasets or create models.
Register new protocols here so their settings survive broker serialization.
"""
from importlib import import_module


ALGORITHMS = {
    'fl': ('fl', 'FLSettings', 'FLSimulation'),
    'dpsgd': ('dpsgd', 'DPSGDSettings', 'DPSGDSimulation'),
    'epidemic': ('dpsgd', 'DPSGDSettings', 'DPSGDSimulation'),
    'gossip': ('gossip', 'GLSettings', 'GossipSimulation'),
    'super-gossip': ('super_gossip', 'SuperGossipSettings', 'SuperGossipSimulation'),
    'adpsgd': ('adpsgd', 'ADPSGDSettings', 'ADPSGDSimulation'),
    'lubor': ('lubor', 'LuborSettings', 'LuborSimulation'),
    'conflux': ('conflux', 'ConfluxSettings', 'ConfluxSimulation'),
    'teleportation': ('teleportation', 'TeleportationSettings', 'TeleportationSimulation'),
    'moshpit': ('moshpit', 'MoshpitSettings', 'MoshpitSimulation'),
    'shatter': ('shatter', 'ShatterSettings', 'ShatterSimulation'),
    'dspodfl': ('dspodfl', 'DSpodFLSettings', 'DSpodFLSimulation'),
    'demo': ('demo', 'DeMoSettings', 'DeMoSimulation'),
}


def class_paths(algorithm):
    package, settings, simulation = ALGORITHMS[algorithm]
    base = 'dasklearn.simulation.' + package
    return base + '.settings.' + settings, base + '.simulation.' + simulation


def import_class(path):
    module, name = path.rsplit('.', 1)
    return getattr(import_module(module), name)


def settings_classes():
    from dasklearn.session_settings import SessionSettings
    result = {'SessionSettings': SessionSettings}
    for algorithm in ALGORITHMS:
        cls = import_class(class_paths(algorithm)[0])
        result[cls.__name__] = cls
    return result
