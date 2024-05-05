import random
import os
from datetime import datetime

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.gossip.client import GossipClient
from dasklearn.simulation.events import *
from dasklearn.simulation.asynchronous_simulation import AsynchronousSimulation
from dasklearn.util import MICROSECONDS


class GossipSimulation(AsynchronousSimulation):
    CLIENT_CLASS = GossipClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        if self.settings.agg == "default":
            self.settings.agg = "age"
        if self.settings.gl_period <= 0:
            raise RuntimeError("Period needs to be larger than 0 for gossip to work")

        self.register_event_callback(DISSEMINATE, "disseminate")

    def setup_data_dir(self, settings: SessionSettings) -> None:
        self.data_dir = os.path.join(settings.work_dir, "data", "%s_%d_%s_n%d_b%d_s%d_%s" %
                                     (settings.algorithm, settings.gl_period // MICROSECONDS, settings.dataset,
                                      settings.participants, settings.brokers,
                                      settings.seed, datetime.now().strftime("%Y%m%d%H%M")))
        settings.data_dir = self.data_dir

    def get_send_set(self, index: int):
        """
        Returns a set of a single index, who will receive a model
        @index - index of the sender
        """
        candidate = index
        while candidate == index:
            candidate = random.randint(0, self.settings.participants - 1)
        return {candidate}
