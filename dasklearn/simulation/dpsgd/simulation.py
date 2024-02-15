import math

import networkx as nx

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.dpsgd.client import DPSGDClient
from dasklearn.simulation.events import *
from dasklearn.simulation.simulation import Simulation


class DPSGDSimulation(Simulation):
    CLIENT_CLASS = DPSGDClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        k = math.floor(math.log2(self.settings.participants))
        self.topology = nx.random_regular_graph(k, self.settings.participants, seed=self.settings.seed)

        self.register_event_callback(FINISH_TRAIN, "finish_train")
        self.register_event_callback(AGGREGATE, "aggregate")

    def initialize_clients(self):
        super().initialize_clients()
        for client in self.clients:
            client.topology = self.topology
