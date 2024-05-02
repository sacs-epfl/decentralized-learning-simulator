from typing import Optional, Tuple

from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.events import *


class LuborClient(AsynchronousClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.queue: Dict[int, Tuple[int, str, int, Dict[int, float]]] = {}  # sender, [sender, model, age, contribution]
        self.send_period: Optional[int] = None

    def init_client(self, event: Event):
        """
        Schedule train, test and disseminate events
        """
        super().init_client(event)

        self.send_period = self.simulator.get_send_period(self.index)
        disseminate_event = Event(event.time + self.send_period, self.index, DISSEMINATE)
        self.simulator.schedule(disseminate_event)

    def finish_train(self, event: Event):
        """
        We finished training. Aggregate and train again
        """
        super().finish_train(event)
        self.aggregate(list(self.queue.values()))

        start_train_event = Event(event.time, self.index, START_TRAIN,
                                  data={"model": self.own_model, "round": self.age})
        self.simulator.schedule(start_train_event)

    def process_incoming_model(self, event: Event):
        """
        We received a model. Update the queue
        """
        self.queue[event.data["from"]] = (event.data["from"], event.data["model"], event.data["metadata"]["age"],
                                          event.data["metadata"]["contribution"])

    def disseminate(self, event: Event):
        """
        Send the model to a random peer.
        """
        # Check if the model is initialized
        if self.own_model is not None:
            self.send()
            self.queue = {}

        # Schedule next disseminate action
        disseminate_event = Event(event.time + self.send_period, self.index, DISSEMINATE)
        self.simulator.schedule(disseminate_event)
