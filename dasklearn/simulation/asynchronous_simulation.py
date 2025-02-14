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

    def get_send_set(self, index: int) -> Set[int]:
        """
        Returns a set of indices of clients, who will receive a model
        @index - index of the sender
        """
        pass
