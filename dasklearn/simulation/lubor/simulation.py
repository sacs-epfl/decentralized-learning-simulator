import random
import math

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

    def get_send_period(self, index: int):
        """
        Initialize the clients
        """
        # Set the speeds and sending period
        if len(self.speeds) == 0:
            self.speeds = {client.index: client.get_train_time() for client in self.clients}
        client_speed: int = self.speeds[index]
        del self.speeds[index]
        average_speed: int = int(sum(self.speeds.values()) / len(self.speeds) / math.log2(self.settings.participants))
        self.speeds[index] = client_speed
        return average_speed

    def get_send_set(self, index: int):
        """
        Returns a set of a single index, who will receive a model
        @index - index of the sender
        """
        weights: Dict[int, float] = {client: 1 / speed for client, speed in self.speeds.items()}
        del weights[index]
        weights = {client: x / sum(weights.values()) for client, x in weights.items()}
        peer: int = random.choices(list(weights.keys()), list(weights.values()))[0]
        return {peer}
