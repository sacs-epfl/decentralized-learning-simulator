import random
import math
import os
from datetime import datetime
from typing import List, Set, Optional

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

        self.training_lengths: Optional[List[int]] = None
        self.speeds: Optional[List[float]] = None
        self.participants: List[int] = [i for i in range(self.settings.participants)]
        self.weights: List[float] = [1 / (self.settings.participants - 1) for _ in range(self.settings.participants)]
        self.settings.k = math.floor(math.log2(self.settings.participants)) if self.settings.k <= 0 else self.settings.k

        self.register_event_callback(DISSEMINATE, "disseminate")

    def setup_data_dir(self, settings: SessionSettings) -> None:
        weights_string: str = "no_weights" if settings.no_weights else "weights"
        self.data_dir = os.path.join(settings.work_dir, "data", "%s_%s_%d_%s_%s_n%d_b%d_s%d_%s" %
                                     (settings.algorithm, settings.agg, settings.k, weights_string,
                                      settings.dataset, settings.participants, settings.brokers, settings.seed,
                                      datetime.now().strftime("%Y%m%d%H%M")))
        settings.data_dir = self.data_dir

    def get_send_period(self, index: int) -> int:
        """
        Initialize the clients
        """
        # Set the speeds and sending period
        if self.training_lengths is None:
            self.training_lengths = [client.get_train_time() for client in self.clients]
            self.speeds = [1 / x for x in self.training_lengths]
        total_speed: int = sum(self.training_lengths) - self.training_lengths[index]
        average_speed: int = int(total_speed / self.settings.participants)
        return average_speed

    def get_send_set(self, index: int) -> Set[int]:
        """
        Returns a set of a single index, who will receive a model
        @index - index of the sender
        """
        if self.settings.no_weights:
            self.weights[index] = 0
            peers: List[int] = random.choices(self.participants, weights=self.weights, k=self.settings.k)
            self.weights[index] = 1 / (self.settings.participants - 1)
        else:
            my_speed: float = self.speeds[index]
            self.speeds[index] = 0
            total_speed: float = sum(self.speeds)
            weights = [speed / total_speed for speed in self.speeds]
            peers: List[int] = random.choices(self.participants, weights=weights, k=self.settings.k)
            self.speeds[index] = my_speed
        return set(peers)
