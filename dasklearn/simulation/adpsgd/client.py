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
        self.start_train_time: int = 0
        self.is_training: bool = False

    def get_train_time(self) -> int:
        train_time: int = super().get_train_time()
        if self.active:
            return train_time
        return train_time // self.simulator.settings.learning.local_steps

    def start_train(self, event: Event):
        if self.active:
            super().start_train(event)
        else:
            self.start_train_time = event.time

    def finish_train(self, event: Event):
        """
        We computed gradient. Update the model and schedule another training or spread the model
        """
        if "local_steps" in event.data:
            local_steps: int = event.data["local_steps"]
        else:
            local_steps: int = self.simulator.settings.learning.local_steps
        self.compute_time += event.data["train_time"]
        self.steps_remaining -= local_steps
        self.opportunity[self.index] += (local_steps / self.simulator.settings.learning.local_steps)
        self.age += (local_steps / self.simulator.settings.learning.local_steps)

        if self.own_model is None:
            self.own_model = event.data["model"]
            self.process_waiting(event)
        elif self.active or event.data["function"] == "train":
            self.own_model = event.data["model"]

        if self.active:
            # Active peer send the model to a random passive neighbour
            self.send()
        elif event.data["function"] == "compute_gradient":
            task_name = Task.generate_name("gradient_update")
            task = Task(task_name, "gradient_update", data={
                "model": self.own_model, "round": event.data["round"],
                "time": self.simulator.current_time, "peer": self.index, "gradient_model": event.data["model"]})
            self.add_compute_task(task)
            self.own_model = task_name
            self.start_train_time = event.time
            self.is_training = False

    def process_incoming_model(self, event: Event):
        """
        We received a model. Aggregate with own model and schedule training again
        """
        if not self.active and not self.is_training:
            local_steps: int = min((event.time - self.start_train_time) // self.get_train_time(), self.steps_remaining)
            train_time: int = 0
            if local_steps > 0:
                task_name = Task.generate_name("train")
                task = Task(task_name, "train", data={
                    "model": self.own_model, "round": int(self.age),
                    "time": self.start_train_time, "peer": self.index,
                    "local_steps": local_steps})
                self.add_compute_task(task)

                train_time = self.get_train_time() * local_steps
                finish_train_event = Event(self.start_train_time + train_time, self.index, FINISH_TRAIN,
                                           data={"model": task_name, "round": int(self.age), "train_time": train_time,
                                                 "function": task.func, "local_steps": local_steps})
                self.finish_train(finish_train_event)
            time: int = self.start_train_time + train_time
            if self.steps_remaining > 0 and time < event.time:
                self.is_training = True
                task_name = Task.generate_name("compute_gradient")
                task = Task(task_name, "compute_gradient", data={
                    "model": self.own_model, "round": int(self.age),
                    "time": time, "peer": self.index,
                    "local_steps": 1})
                self.add_compute_task(task)

                train_time = self.get_train_time()
                finish_train_event = Event(time + train_time, self.index, FINISH_TRAIN,
                                           data={"model": task_name, "round": int(self.age), "train_time": train_time,
                                                 "function": task.func, "local_steps": 1})
                self.simulator.schedule(finish_train_event)
            else:
                self.start_train_time = event.time

        # Wait for initialization
        if self.own_model is None:
            self.waiting.append(event)
            return

        if not self.active:
            # Passive peer sends back its own model
            self.client_log("Client %d will send model %s to %d" % (self.index, self.own_model, event.data["from"]))
            metadata = dict(age=int(self.age), opportunity=self.opportunity)
            self.send_model(event.data["from"], self.own_model, metadata=metadata)

        self.aggregate([(event.data["from"], event.data["model"], event.data["metadata"]["age"],
                         event.data["metadata"]["opportunity"])])

        # Schedule next local step
        if self.active:
            start_train_event = Event(event.time, self.index, START_TRAIN,
                                      data={"model": self.own_model, "round": int(self.age)})
            self.simulator.schedule(start_train_event)

        # Increase the remaining steps count
        self.steps_remaining += self.simulator.settings.learning.local_steps

    def process_waiting(self, event: Event):
        """
        Process all incoming models waiting in the queue
        """
        for waiting_event in self.waiting:
            waiting_event.time = event.time
            self.process_incoming_model(waiting_event)
        self.waiting = []
