from typing import Optional, Tuple, List

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from queue import Queue


class SuperGossipClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)

        self.own_model: Optional[str] = None
        self.age: int = 0
        self.queue: Queue[Tuple[str, int, int]] = Queue()  # Model, sender, age
        self.queue_size: List[Tuple[int, int]] = []  # Log queue size at different times

    def init_client(self, event: Event):
        """
        Schedule train and test events
        """
        start_train_event = Event(event.time, self.index, START_TRAIN,
                                  data={"model": self.own_model, "round": self.age})
        self.simulator.schedule(start_train_event)

        # No testing when test_period == 0
        if self.simulator.settings.test_period > 0:
            test_event = Event(event.time + self.simulator.settings.test_period, self.index, TEST)
            self.simulator.schedule(test_event)

    def finish_train(self, event: Event):
        """
        We finished training. Aggregate and send the model to a random peer
        """
        self.compute_time += event.data["train_time"]
        self.own_model = event.data["model"]
        self.age += 1

        if self.queue.qsize() > 0:
            # Aggregate the incoming and own models
            other_model, other_index, other_age = self.queue.get()
            model_names = [other_model, self.own_model]
            self.client_log("Client %d will aggregate (%s)" % (self.index, model_names))

            # Compute weights
            rounds = [other_age, self.age]
            self.aggregations.append((other_index, self.age, other_age))
            weights = list(map(lambda x: x / sum(rounds), rounds))
            self.age = max(rounds)
            self.own_model = self.aggregate_models(model_names, self.age, weights)

        # Select random peer and send them the model
        peer = self.simulator.get_random_participant(self.index)
        self.client_log("Client %d will send model %s to %d" % (self.index, self.own_model, peer))
        metadata = dict(rounds=self.age, index=self.index)
        self.send_model(peer, self.own_model, metadata)

        # Proceed to the training again
        start_train_event = Event(event.time, self.index, START_TRAIN,
                                  data={"model": self.own_model, "round": self.age})
        self.simulator.schedule(start_train_event)

    def on_incoming_model(self, event: Event):
        """
        We received a model. Perform one aggregate and training step.
        """
        self.client_log("Client %d received from %d model %s" % (self.index, event.data["from"], event.data["model"]))
        self.queue.put((event.data["model"], event.data["metadata"]["index"], event.data["metadata"]["rounds"]))
        self.queue_size.append((self.queue.qsize(), event.time))
        # Maximal size of queue
        if self.queue.qsize() > self.simulator.settings.queue_max_size:
            self.queue.get()

    def test(self, event: Event):
        """
        Test model's performance
        """
        # Check if the model is initialized
        if self.age > 0:
            self.client_log("Client %d will test its model %s" % (self.index, self.own_model))

            test_task_name = "test_%d_%d" % (self.index, event.time // MICROSECONDS)
            task = Task(test_task_name, "test", data={
                "model": self.own_model, "time": self.simulator.current_time,
                "peer": self.index, "round": self.age})
            self.add_compute_task(task)

        # Schedule next test action
        test_event = Event(event.time + self.simulator.settings.test_period, self.index, TEST)
        self.simulator.schedule(test_event)
