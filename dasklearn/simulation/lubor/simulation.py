import random
import math
import os
from datetime import datetime
from typing import List, Set

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.lubor.client import LuborClient
from dasklearn.simulation.events import *
from dasklearn.simulation.asynchronous_simulation import AsynchronousSimulation


class LuborSimulation(AsynchronousSimulation):
    CLIENT_CLASS = LuborClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        if self.settings.agg == "default":
            self.settings.agg = "age"

        self.speeds: Dict[int, int] = {}
        self.register_event_callback(DISSEMINATE, "disseminate")

    def setup_data_dir(self, settings: SessionSettings) -> None:
        self.data_dir = os.path.join(settings.work_dir, "data", "%s_%s_%s_n%d_b%d_s%d_%s" %
                                     (settings.algorithm, settings.agg, settings.dataset, settings.participants,
                                      settings.brokers, settings.seed, datetime.now().strftime("%Y%m%d%H%M")))
        settings.data_dir = self.data_dir

    def get_send_period(self, index: int) -> int:
        """
        Initialize the clients
        """
        # Set the speeds and sending period
        if len(self.speeds) == 0:
            self.speeds = {client.index: client.get_train_time() for client in self.clients}
        client_speed: int = self.speeds[index]
        del self.speeds[index]
        average_speed: int = int(sum(self.speeds.values()) / len(self.speeds))
        self.speeds[index] = client_speed
        return average_speed

    def get_send_set(self, index: int) -> Set[int]:
        """
        Returns a set of a single index, who will receive a model
        @index - index of the sender
        """
        weights: Dict[int, float] = {client: 1 / speed for client, speed in self.speeds.items()}
        del weights[index]
        weights = {client: x / sum(weights.values()) for client, x in weights.items()}
        k: int = math.floor(math.log2(self.settings.participants))
        peers: List[int] = random.choices(list(weights.keys()), list(weights.values()), k=k)
        return set(peers)
