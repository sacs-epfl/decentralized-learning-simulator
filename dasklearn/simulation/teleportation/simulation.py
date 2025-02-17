from math import log2
import networkx as nx

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.teleportation.client import TeleportationClient
from dasklearn.simulation.events import *
from dasklearn.simulation.asynchronous_simulation import AsynchronousSimulation


class TeleportationSimulation(AsynchronousSimulation):
    CLIENT_CLASS = TeleportationClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        settings.sample_size = min(settings.sample_size, settings.participants)
        self.register_event_callback(START_ROUND, "start_round")

        # The sample size must be a power of 2
        if settings.sample_size & (settings.sample_size - 1) != 0:
            raise ValueError("Sample size must be a power of 2")
        
        nbs_in_k_topology: int = int(log2(settings.sample_size))
        self.G_k = nx.random_regular_graph(nbs_in_k_topology, settings.sample_size)
