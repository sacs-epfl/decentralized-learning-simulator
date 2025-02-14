import math
import random
import os
from collections import defaultdict
from typing import List, Tuple
from datetime import datetime

import networkx as nx

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.epidemic.client import EpidemicClient
from dasklearn.simulation.events import *
from dasklearn.simulation.simulation import Simulation


class EpidemicSimulation(Simulation):
    CLIENT_CLASS = EpidemicClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        self.topologies = []
        self.participants = [i for i in range(self.settings.participants)]
        self.s = math.floor(math.log2(self.settings.participants))
        self.clients_ready_for_round: Dict[int, List[Tuple[int, Dict]]] = defaultdict(lambda: [])

        random.seed(self.settings.seed)

        self.register_event_callback(START_ROUND, "start_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")
        self.register_event_callback(AGGREGATE, "aggregate")

    def setup_data_dir(self, settings: SessionSettings) -> None:
        self.data_dir = os.path.join(settings.work_dir, "data", "%s_%s_%s_n%d_b%d_s%d_%s" %
                                     (settings.algorithm, settings.el, settings.dataset, settings.participants,
                                      settings.brokers, settings.seed, datetime.now().strftime("%Y%m%d%H%M")))
        settings.data_dir = self.data_dir

    def add_topology(self) -> None:
        # Generate a topology
        g = nx.DiGraph()
        if self.settings.el == "local":
            for node in self.participants:
                nodes = self.participants[:]
                nodes.remove(node)
                connections = random.sample(nodes, self.s)
                for other_node in connections:
                    g.add_edge(node, other_node)
        else:
            undirected = nx.random_regular_graph(d=self.s, n=self.settings.participants,
                                                 seed=self.settings.seed + len(self.topologies))
            for u, v in undirected.edges():
                g.add_edge(u, v)
                g.add_edge(v, u)
        self.topologies.append(g)

    def client_ready_for_round(self, client_id: int, round_nr: int, start_round_info: Dict):
        """
        This function is used if synchronous mode is turned on. In this setting, the simulator will initiate the next
        round if all nodes are ready to start it.
        """
        self.clients_ready_for_round[round_nr].append((client_id, start_round_info))
        if len(self.clients_ready_for_round[round_nr]) == self.settings.participants:
            # Some sanity checking - 1) all nodes should be finished training in the previous round
            for client in self.clients:
                assert not client.bw_scheduler.incoming_requests
                assert not client.bw_scheduler.outgoing_requests
                assert not client.bw_scheduler.incoming_transfers
                assert not client.bw_scheduler.outgoing_transfers

            self.logger.info("Simulator starting round %d", round_nr)
            for client_id, info in self.clients_ready_for_round[round_nr]:
                start_round_event = Event(self.current_time, client_id, START_ROUND, data=info)
                self.schedule(start_round_event)
            self.clients_ready_for_round.pop(round_nr)
