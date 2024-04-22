import random

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.gossip.client import GossipClient
from dasklearn.simulation.events import *
from dasklearn.simulation.asynchronous_simulation import AsynchronousSimulation


class GossipSimulation(AsynchronousSimulation):
    CLIENT_CLASS = GossipClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        if self.settings.agg == "default":
            self.settings.agg = "age"
        if self.settings.gl_period == 0:
            raise RuntimeError("Period needs to be larger than 0 for gossip to work")

        self.register_event_callback(DISSEMINATE, "disseminate")

    def get_send_set(self, index: int):
        """
        Returns a set of a single index, who will receive a model
        @index - index of the sender
        """
        candidate = index
        while candidate == index:
            candidate = random.randint(0, self.settings.participants - 1)
        return {candidate}
