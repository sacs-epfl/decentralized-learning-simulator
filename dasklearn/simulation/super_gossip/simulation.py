import math
import random
import os
from datetime import datetime
from typing import List

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.super_gossip.client import SuperGossipClient
from dasklearn.simulation.asynchronous_simulation import AsynchronousSimulation


class SuperGossipSimulation(AsynchronousSimulation):
    CLIENT_CLASS = SuperGossipClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        if settings.agg == "default":
            settings.agg = "age"
        self.participants: List[int] = [i for i in range(self.settings.participants)]
        self.weights: List[float] = [1 / (self.settings.participants - 1) for _ in range(self.settings.participants)]
        self.k: int = math.floor(math.log2(self.settings.participants))

    def setup_data_dir(self, settings: SessionSettings) -> None:
        wait_string: str = "wait" if settings.wait else "no_wait"
        self.data_dir = os.path.join(settings.work_dir, "data", "%s_%s_%s_n%d_b%d_s%d_%s" %
                                     (settings.algorithm, wait_string, settings.dataset, settings.participants,
                                      settings.brokers, settings.seed, datetime.now().strftime("%Y%m%d%H%M")))
        settings.data_dir = self.data_dir

    def get_send_set(self, index: int):
        """
        Returns a set of a single index, who will receive a model
        @index - index of the sender
        """
        self.weights[index] = 0
        peers: List[int] = random.choices(self.participants, weights=self.weights, k=self.k)
        self.weights[index] = 1 / (self.settings.participants - 1)
        return set(peers)
