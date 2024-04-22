import os
import random

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.super_gossip.client import SuperGossipClient
from dasklearn.simulation.events import *
from dasklearn.simulation.simulation import Simulation


class SuperGossipSimulation(Simulation):
    CLIENT_CLASS = SuperGossipClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        if settings.agg == "default":
            settings.agg = "age"

        self.register_event_callback(FINISH_TRAIN, "finish_train")
        self.register_event_callback(TEST, "test")

        random.seed(settings.seed)

    def schedule(self, event: Event):
        # Don't schedule events after the running duration elapses, transfers are allowed to satisfy sanity checks,
        # but their results are not considered
        if event.time > self.settings.duration and event.action != "start_transfer" and event.action != "finish_outgoing_transfer":
            return
        super().schedule(event)

    def get_random_participant(self, index: int):
        """
        Returns a random participant other than oneself (index)
        """
        candidate = index
        while candidate == index:
            candidate = random.randint(0, self.settings.participants - 1)
        return candidate

    def save_measurements(self) -> None:
        super().save_measurements()
        # Write queue size
        with open(os.path.join(self.data_dir, "queue_size.csv"), "w") as file:
            file.write("client,queue_size,time\n")
            for client in self.clients:
                for size, time in client.queue_size:
                    file.write("%s,%d,%d\n" % (client.index, size, time))

