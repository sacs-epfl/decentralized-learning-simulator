from typing import Optional

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS


class GossipClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)

        self.own_model: Optional[str] = None
        self.age = 0
        # Not available by default
        self.available = False

    def init_client(self, event: Event):
        """
        Schedule train, test and disseminate events
        """
        start_train_event = Event(event.time, self.index, START_TRAIN,
                                  data={"model": self.own_model, "round": self.age})
        self.simulator.schedule(start_train_event)

        disseminate_event = Event(event.time + self.simulator.settings.period, self.index, DISSEMINATE)
        self.simulator.schedule(disseminate_event)

        # No testing when test_period == 0
        if self.simulator.settings.test_period > 0:
            test_event = Event(event.time + self.simulator.settings.test_period, self.index, TEST)
            self.simulator.schedule(test_event)

    def finish_train(self, event: Event):
        """
        We finished training. Update the model after the training was finished
        """
        # Check if we are still running
        if event.time > self.simulator.settings.duration:
            return

        self.own_model = event.data["model"]
        self.available = True
        self.age += 1

    def on_incoming_model(self, event: Event):
        """
        We received a model. Perform one aggregate and training step.
        """
        # Check if we are still running
        if event.time > self.simulator.settings.duration:
            return

        self.client_log("Client %d received from %d model %s" % (self.index, event.data["from"], event.data["model"]))

        # Accept the incoming model if available
        if self.available and event.time:
            # Lock
            self.available = False

            # Aggregate the incoming and own models
            model_names = [event.data["model"], self.own_model]
            self.client_log("Client %d will aggregate and train (%s)" % (self.index, model_names))

            # Compute weights
            rounds = [event.data["metadata"]["rounds"], self.age]
            weights = list(map(lambda x: x / sum(rounds), rounds))
            self.age = max(rounds)
            self.own_model = self.aggregate_models(model_names, self.age, weights)

            # Schedule a train action
            start_train_event = Event(event.time, self.index, START_TRAIN, data={
                "model": self.own_model, "round": self.age})
            self.simulator.schedule(start_train_event)

    def disseminate(self, event: Event):
        """
        Send the model to a random peer.
        """
        # Check if we are still running
        if event.time > self.simulator.settings.duration:
            return

        # Check if the model is initialized
        if self.age > 0:
            peer = self.simulator.get_random_participant(self.index)
            metadata = dict(rounds=self.age)

            self.client_log("Client %d will send model %s to %d" % (self.index, self.own_model, peer))
            self.send_model(peer, self.own_model, metadata)

        # Schedule next disseminate action
        disseminate_event = Event(event.time + self.simulator.settings.period, self.index, DISSEMINATE)
        self.simulator.schedule(disseminate_event)

    def test(self, event: Event):
        """
        Test model's performance
        """
        # Check if we are still running
        if event.time > self.simulator.settings.duration:
            return

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
