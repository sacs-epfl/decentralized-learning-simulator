import random
from typing import Optional, Tuple

from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.events import *


class GossipClient(AsynchronousClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        random.seed(index)

        # Not available by default
        self.available: bool = False
        self.last_model: Optional[Tuple[int, str, int, Dict[int, float]]] = None

    def init_client(self, event: Event):
        """
        Schedule train, test and disseminate events
        """
        super().init_client(event)
        # Randomly select shift
        start_time: int = random.randint(0, self.simulator.settings.gl_period)
        disseminate_event = Event(event.time + start_time + self.simulator.settings.gl_period, self.index, DISSEMINATE)
        self.simulator.schedule(disseminate_event)

    def finish_train(self, event: Event):
        """
        We finished training. Update the model after the training was finished
        """
        super().finish_train(event)
        self.available = True
        if self.last_model is not None:
            self.clear()

    def process_incoming_model(self, event: Event):
        """
        We received a model. Perform one aggregate and training step.
        """
        self.last_model = (event.data["from"], event.data["model"], event.data["metadata"]["age"],
                           event.data["metadata"]["contribution"])
        # Accept the incoming model if no other model is being trained
        if self.available:
            self.clear()

    def clear(self):
        # Lock
        self.available = False
        self.aggregate([self.last_model])
        self.last_model = None

        # Schedule a train action
        start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN, data={
            "model": self.own_model, "round": self.age})
        self.simulator.schedule(start_train_event)

    def disseminate(self, event: Event):
        """
        Send the model to a random peer.
        """
        # Check if the model is initialized
        if self.own_model is not None:
            self.send()

        # Schedule next disseminate action
        disseminate_event = Event(event.time + self.simulator.settings.gl_period, self.index, DISSEMINATE)
        self.simulator.schedule(disseminate_event)
