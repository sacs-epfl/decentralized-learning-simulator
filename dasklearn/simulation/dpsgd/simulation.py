from copy import deepcopy
import math
from collections import defaultdict
import random
from typing import List, Tuple

import networkx as nx

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.dpsgd.client import DPSGDClient
from dasklearn.simulation.events import *
from dasklearn.simulation.simulation import Simulation


class DPSGDSimulation(Simulation):
    CLIENT_CLASS = DPSGDClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        self.topologies = []
        if self.settings.k == 0:
                    self.settings.k = math.floor(math.log2(self.settings.participants))
        self.clients_ready_for_round: Dict[int, List[Tuple[int, Dict]]] = defaultdict(lambda: [])

        self.register_event_callback(START_ROUND, "start_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")
        self.register_event_callback(AGGREGATE, "aggregate")

    def add_topology(self) -> None:
        # Generate a new topology
        if self.settings.algorithm == "dpsgd" or (self.settings.algorithm == "epidemic" and self.settings.el == "oracle"):
            seed = self.settings.seed if self.settings.algorithm == "dpsgd" else self.settings.seed + len(self.topologies)
            r = random.Random(seed)

            if self.settings.topology == "kreg":
                g = nx.random_regular_graph(self.settings.k, self.settings.participants, seed=seed)
            elif self.settings.topology == "ring":
                g = nx.cycle_graph(self.settings.participants)
                nodes = list(g.nodes())
                mapping = dict(zip(g.nodes(), r.sample(nodes, len(nodes))))
                g = nx.relabel_nodes(g, mapping)
            else:
                raise ValueError("Unknown topology %s" % self.settings.topology)
            g = g.to_directed()
        else:
            g = nx.DiGraph()
            if self.settings.algorithm == "epidemic" and self.settings.el == "local":
                for node in self.participants:
                    nodes = self.participants[:]
                    nodes.remove(node)
                    connections = r.sample(nodes, self.k)
                    for other_node in connections:
                        g.add_edge(node, other_node)

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
