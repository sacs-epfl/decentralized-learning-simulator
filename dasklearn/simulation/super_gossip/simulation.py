import os
import random

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.super_gossip.client import SuperGossipClient
from dasklearn.simulation.asynchronous_simulation import AsynchronousSimulation


class SuperGossipSimulation(AsynchronousSimulation):
    CLIENT_CLASS = SuperGossipClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        if settings.agg == "default":
            settings.agg = "age"

    def get_send_set(self, index: int):
        """
        Returns a set of a single index, who will receive a model
        @index - index of the sender
        """
        candidate = index
        while candidate == index:
            candidate = random.randint(0, self.settings.participants - 1)
        return {candidate}
