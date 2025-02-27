from typing import Tuple

from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.events import *


class SuperGossipClient(AsynchronousClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)

        self.queue: Dict[int, Tuple[int, str, int]] = {}  # sender, [sender, model, age]
        self.waiting: bool = False

    def finish_train(self, event: Event):
        """
        We finished training. Aggregate and send the model to a random peer
        """
        super().finish_train(event)
        # Special case - send the model in the first iteration
        if self.age == 1:
            self.send()

        if len(self.queue) == 0 and self.simulator.settings.wait:
            # Wait to receive a model if queue is empty
            self.waiting = True
        else:
            self.clear_queue()

    def process_incoming_model(self, event: Event):
        """
        We received a model. Put it into a queue
        """
        sender_id: int = event.data["from"]
        self.queue[sender_id] = (sender_id, event.data["model"], event.data["metadata"]["age"])
        # We were previously waiting to receive a model, aggregate and continue training
        if self.waiting:
            self.clear_queue()
            self.waiting = False

    def clear_queue(self):
        """
        Aggregate all models in the queue and schedule training
        """
        self.aggregate(list(self.queue.values()))
        self.queue = {}
        self.send()

        # Schedule a train action
        start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN,
                                  data={"model": self.own_model, "round": self.age})
        self.simulator.schedule(start_train_event)
