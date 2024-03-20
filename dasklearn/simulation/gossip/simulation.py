import random

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.gossip.client import GossipClient
from dasklearn.simulation.events import *
from dasklearn.simulation.simulation import Simulation


class GossipSimulation(Simulation):
    CLIENT_CLASS = GossipClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)

        if settings.period == 0:
            raise RuntimeError("Period needs to be larger than 0 for gossip to work")

        self.register_event_callback(FINISH_TRAIN, "finish_train")
        self.register_event_callback(DISSEMINATE, "disseminate")
        self.register_event_callback(TEST, "test")
        self.register_event_callback(AGGREGATE, "aggregate")

        random.seed(settings.seed)

    def get_random_participant(self, index: int):
        """
        Returns a random participant other than oneself (index)
        """
        candidate = index
        while candidate == index:
            candidate = random.randint(0, self.settings.participants - 1)
        return candidate
