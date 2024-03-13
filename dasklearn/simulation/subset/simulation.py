from collections import defaultdict
from typing import List, Tuple

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.events import *
from dasklearn.simulation.simulation import Simulation
from dasklearn.simulation.subset.client import SubsetDLClient


class SubsetDLSimulation(Simulation):
    CLIENT_CLASS = SubsetDLClient

    def __init__(self, settings: SessionSettings, sample_size: int):
        super().__init__(settings)
        self.sample_size: int = sample_size
        self.clients_ready_for_round: Dict[int, List[Tuple[int, Dict]]] = defaultdict(lambda: [])
        self.register_event_callback(START_ROUND, "start_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")

    def client_ready_for_round(self, client_id: int, round_nr: int, start_round_info: Dict):
        """
        This function is used if synchronous mode is turned on. In this setting, the simulator will initiate the next
        round if all nodes are ready to start it.
        """
        self.logger.info("Simulator starting round %d", round_nr)
        self.clients_ready_for_round[round_nr].append((client_id, start_round_info))
        if len(self.clients_ready_for_round[round_nr]) == self.sample_size:
            for client_id, info in self.clients_ready_for_round[round_nr]:
                start_round_event = Event(self.current_time, client_id, START_ROUND, data=info)
                self.schedule(start_round_event)
            self.clients_ready_for_round.pop(round_nr)
