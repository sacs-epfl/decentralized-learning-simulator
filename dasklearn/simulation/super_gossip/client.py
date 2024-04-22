from typing import Tuple

from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.events import *
from queue import Queue


class SuperGossipClient(AsynchronousClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)

        self.queue: Queue[Tuple[int, str, int]] = Queue()  # sender, model, age

    def finish_train(self, event: Event):
        """
        We finished training. Aggregate and send the model to a random peer
        """
        super().finish_train(event)
        if self.queue.qsize() > 0:
            self.aggregate([self.queue.get()])
        self.send()

        # Schedule a train action
        start_train_event = Event(event.time, self.index, START_TRAIN,
                                  data={"model": self.own_model, "round": self.age})
        self.simulator.schedule(start_train_event)

    def process_incoming_model(self, event: Event):
        """
        We received a model. Put it into a queue
        """
        self.queue.put((event.data["model"], event.data["metadata"]["index"], event.data["metadata"]["rounds"]))
        # Deque if we reached maximum size of the queue
        if self.queue.qsize() > self.simulator.settings.queue_max_size:
            self.queue.get()
