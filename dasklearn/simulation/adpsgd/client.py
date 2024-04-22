from typing import Optional

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS


class ADPSGDClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)

        self.own_model: Optional[str] = None
        self.age: int = 0

        # active/passive peer
        self.active: bool = True
        self.steps_remaining: int = 0
        self.train_time: int = 0

    def init_client(self, event: Event):
        """
        Schedule a training and testing
        """
        self.steps_remaining += self.simulator.settings.learning.local_steps
        compute_gradient_event = Event(event.time, self.index, COMPUTE_GRADIENT,
                                       data={"model": self.own_model, "round": self.age})
        self.simulator.schedule(compute_gradient_event)

        # Schedule testing, no testing when test_period == 0
        if self.simulator.settings.test_period > 0:
            test_event = Event(event.time + self.simulator.settings.test_period, self.index, TEST)
            self.simulator.schedule(test_event)

    def compute_gradient(self, event: Event):
        """
        We started training. Schedule when the training has ended.
        """
        task_name = Task.generate_name("compute_gradient")
        task = Task(task_name, "compute_gradient", data={
            "model": event.data["model"], "round": event.data["round"],
            "time": self.simulator.current_time, "peer": self.index})
        self.add_compute_task(task)

        if self.age % self.simulator.settings.learning.local_steps == 0:
            # Divide time by number of local steps, because we schedule only 1 at a time
            self.train_time = self.get_train_time()
            self.train_time //= self.simulator.settings.learning.local_steps
        gradient_update_event = Event(self.simulator.current_time + self.train_time, self.index, GRADIENT_UPDATE,
                                      data={"model": task_name, "round": event.data["round"], "train_time": self.train_time})
        self.simulator.schedule(gradient_update_event)

    def gradient_update(self, event: Event):
        """
        We started training. Schedule when the training has ended.
        """
        # Special case for model initialization
        if self.age == 0:
            self.own_model = event.data["model"]
        self.compute_time += event.data["train_time"]

        task_name = Task.generate_name("gradient_update")
        task = Task(task_name, "gradient_update", data={
            "model": self.own_model, "round": event.data["round"],
            "time": self.simulator.current_time, "peer": self.index, "gradient_model": event.data["model"]})
        self.add_compute_task(task)

        # We assume gradient update time is negligible compared to gradient compute time
        finish_train_event = Event(self.simulator.current_time, self.index, FINISH_TRAIN,
                                   data={"model": task_name, "round": event.data["round"]})
        self.simulator.schedule(finish_train_event)

    def finish_train(self, event: Event):
        """
        We finished a single local step
        """
        self.steps_remaining -= 1
        self.age += 1
        self.own_model = event.data["model"]

        if self.active and self.steps_remaining == 0:
            # Active peer send the model to a random neighbour to aggregate
            neighbour = self.simulator.get_random_participant(self.active)
            self.client_log("Client %d will send model %s to %d" % (self.index, self.own_model, neighbour))
            metadata = dict(age=self.age // self.simulator.settings.learning.local_steps, index=self.index)
            self.send_model(neighbour, self.own_model, metadata=metadata)
        elif self.steps_remaining > 0:
            # Schedule a next local steps
            compute_gradient_event = Event(event.time, self.index, COMPUTE_GRADIENT,
                                           data={"model": self.own_model, "round": self.age})
            self.simulator.schedule(compute_gradient_event)

    def on_incoming_model(self, event: Event):
        """
        We received a model. Aggregate with own model and schedule training again
        """
        self.client_log("Client %d received from %d model %s" % (self.index, event.data["from"], event.data["model"]))

        if not self.active:
            # Passive peer sends back its own model
            self.client_log("Client %d will send model %s to %d" % (self.index, self.own_model, event.data["from"]))
            metadata = dict(age=self.age // self.simulator.settings.learning.local_steps, index=self.index)
            self.send_model(event.data["from"], self.own_model, metadata=metadata)

        # Compute weights
        weights = None
        if self.simulator.settings.agg == "age":
            ages = [event.data["metadata"]["age"], self.age]
            weights = list(map(lambda x: x / sum(ages), ages))
            self.age = max(ages)

        # Aggregate the incoming and own models
        model_names = [event.data["model"], self.own_model]
        self.client_log("Client %d will aggregate and train (%s)" % (self.index, model_names))
        self.aggregations.append((event.data["metadata"]["index"], self.age, event.data["metadata"]["age"]))
        self.own_model = self.aggregate_models(model_names, self.age, weights)

        # Schedule the next training step
        if self.active or self.steps_remaining == 0:
            compute_gradient_event = Event(event.time, self.index, COMPUTE_GRADIENT,
                                           data={"model": self.own_model, "round": self.age})
            self.simulator.schedule(compute_gradient_event)

        # Increase the remaining steps count
        self.steps_remaining += self.simulator.settings.learning.local_steps

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

        # Schedule the next test action
        test_event = Event(event.time + self.simulator.settings.test_period, self.index, TEST)
        self.simulator.schedule(test_event)
