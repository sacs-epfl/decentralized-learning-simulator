import random
from abc import ABC
from typing import Set

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.events import *
from dasklearn.simulation.simulation import Simulation


class AsynchronousSimulation(Simulation, ABC):
    """
    This is an abstract class for asynchronous simulation (gossip, adpsgd...)
    """
    CLIENT_CLASS = AsynchronousClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)

        self.register_event_callback(FINISH_TRAIN, "finish_train")
        self.register_event_callback(TEST, "test")

        random.seed(settings.seed)

    def schedule(self, event: Event):
        """
        Schedule an event as normal, unless it is time to stop.
        In that case transfers are still allowed to satisfy sanity checks,
        but their results are won't be considered (other events cannot be scheduled).
        """
        if event.time > self.settings.duration and event.action != "start_transfer" and\
                event.action != "finish_outgoing_transfer":
            return
        super().schedule(event)

    def get_send_set(self, index: int) -> Set[int]:
        """
        Returns a set of indices of clients, who will receive a model
        @index - index of the sender
        """
        pass
