from typing import List

from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task


class ADPSGDClient(AsynchronousClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)

        # active/passive peer
        self.active: bool = True
        self.steps_remaining: int = self.simulator.settings.learning.local_steps
        self.waiting: List[Event] = []
        self.last_train_time: int = 0
        self.train_function = "compute_gradient"

    def get_train_time(self) -> int:
        if self.steps_remaining % self.simulator.settings.learning.local_steps == 0:
            # Divide time by number of local steps, because we schedule only 1 at a time
            self.last_train_time = super().get_train_time() // self.simulator.settings.learning.local_steps
        return self.last_train_time

    def finish_train(self, event: Event):
        """
        We computed gradient. Update the model and schedule another training or spread the model
        """
        self.compute_time += event.data["train_time"]
        self.steps_remaining -= 1
        self.contribution[self.index] += 1
        # Special case for model initialization
        if self.own_model is None:
            self.own_model = event.data["model"]
            self.process_waiting()
        if self.steps_remaining % self.simulator.settings.learning.local_steps == 0:
            self.age += 1

        task_name = Task.generate_name("gradient_update")
        task = Task(task_name, "gradient_update", data={
            "model": self.own_model, "round": event.data["round"],
            "time": self.simulator.current_time, "peer": self.index, "gradient_model": event.data["model"]})
        self.add_compute_task(task)
        self.own_model = task_name

        if self.steps_remaining > 0:
            # Schedule next local step
            start_train_event = Event(event.time, self.index, START_TRAIN,
                                      data={"model": self.own_model, "round": self.age})
            self.simulator.schedule(start_train_event)
        elif self.active:
            # Active peer send the model to a random passive neighbour
            self.send()

    def process_incoming_model(self, event: Event):
        """
        We received a model. Aggregate with own model and schedule training again
        """
        # Wait for initialization
        if self.own_model is None:
            self.waiting.append(event)
            return

        if not self.active:
            # Passive peer sends back its own model
            self.client_log("Client %d will send model %s to %d" % (self.index, self.own_model, event.data["from"]))
            metadata = dict(age=self.age, contribution=self.contribution)
            self.send_model(event.data["from"], self.own_model, metadata=metadata)

        self.aggregate([(event.data["from"], event.data["model"], event.data["metadata"]["age"],
                         event.data["metadata"]["contribution"])])

        # Schedule next local step
        if self.active or self.steps_remaining == 0:
            start_train_event = Event(event.time, self.index, START_TRAIN,
                                      data={"model": self.own_model, "round": self.age})
            self.simulator.schedule(start_train_event)

        # Increase the remaining steps count
        self.steps_remaining += self.simulator.settings.learning.local_steps

    def process_waiting(self):
        """
        Process all incoming models waiting in the queue
        """
        for event in self.waiting:
            self.process_incoming_model(event)
        self.waiting = []
