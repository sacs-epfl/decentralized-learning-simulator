import math
from collections import defaultdict
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
        self.k = math.floor(math.log2(self.settings.participants))
        self.topology = nx.random_regular_graph(self.k, self.settings.participants, seed=self.settings.seed)
        self.clients_ready_for_round: Dict[int, List[Tuple[int, Dict]]] = defaultdict(lambda: [])

        self.register_event_callback(START_ROUND, "start_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")
        self.register_event_callback(AGGREGATE, "aggregate")

    def initialize_clients(self):
        super().initialize_clients()
        for client in self.clients:
            client.topology = self.topology

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
