import math
from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.events import *
from dasklearn.simulation.shatter.client import ShatterClient
from dasklearn.simulation.simulation import Simulation

import networkx as nx


class ShatterSimulation(Simulation):
    CLIENT_CLASS = ShatterClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        self.topologies = []
        self.finished: Dict[int, int] = {}
        if self.settings.r == 0:
            self.settings.r = math.floor(math.log2(self.settings.participants * self.settings.k))

        self.register_event_callback(START_ROUND, "start_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")
        self.register_event_callback(SEND_CHUNKS, "send_chunks")

    def set_finished(self, round_nr: int):
        if round_nr not in self.finished:
            self.finished[round_nr] = 0
        self.finished[round_nr] += 1

        if self.finished[round_nr] == self.settings.participants:
            self.logger.info("All clients finished round %d (t=%.3f)", round_nr, self.cur_time_in_sec())
            self.finished.pop(round_nr)

    def add_topology(self):
        seed = self.settings.seed + len(self.topologies)
        g = nx.random_regular_graph(self.settings.r, self.settings.participants * self.settings.k, seed=seed)
        g = g.to_directed()
        self.topologies.append(g)
