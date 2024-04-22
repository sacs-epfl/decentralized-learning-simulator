from abc import ABC
from typing import List, Optional, Tuple, Set

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS


class AsynchronousClient(BaseClient, ABC):
    """
    This is an abstract class for asynchronous clients (gossip, adpsgd...)
    """

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)

        self.own_model: Optional[str] = None
        self.age: int = 0

    def init_client(self, event: Event):
        """
        Schedule initial (train) and test events
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
        We finished training. Update the model after the training was finished
        """
        self.compute_time += event.data["train_time"]
        self.own_model = event.data["model"]
        self.age += 1

    def on_incoming_model(self, event: Event):
        """
        We received a model.
        """
        if event.time > self.simulator.settings.duration:
            return
        self.client_log("Client %d received from %d model %s" % (self.index, event.data["from"], event.data["model"]))
        self.process_incoming_model(event)

    def process_incoming_model(self, event: Event):
        pass

    def aggregate(self, models: List[Tuple[int, str, int]]):
        """
        Aggregate the incoming and own models
        @models: tuple of - index of the sender, model of the sender, age of the sender
        """
        # Add own model to the aggregation
        models.append((self.index, self.own_model, self.age))
        model_names: List[str] = list(map(lambda x: x[1], models))
        self.aggregations.append(models)
        self.client_log("Client %d will aggregate and train (%s)" % (self.index, model_names))

        if self.simulator.settings.agg == "age":
            # Compute weights
            ages = list(map(lambda x: x[2], models))
            weights = list(map(lambda x: x / sum(ages), ages))
            self.age = max(ages)
        else:
            weights = None
        self.own_model = self.aggregate_models(model_names, self.age, weights)

    def send(self):
        """
        Send own model to a set of receiver clients
        """
        clients: Set[int] = self.simulator.get_send_set(self.index)
        for client in clients:
            self.client_log("Client %d will send model %s to %d" % (self.index, self.own_model, client))
            self.send_model(client, self.own_model, metadata=dict(age=self.age))

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
