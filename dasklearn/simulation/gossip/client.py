from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.events import *


class GossipClient(AsynchronousClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)

        # Not available by default
        self.available = False

    def init_client(self, event: Event):
        """
        Schedule train, test and disseminate events
        """
        super().init_client(event)

        disseminate_event = Event(event.time + self.simulator.settings.gl_period, self.index, DISSEMINATE)
        self.simulator.schedule(disseminate_event)

    def finish_train(self, event: Event):
        """
        We finished training. Update the model after the training was finished
        """
        super().finish_train(event)
        self.available = True

    def process_incoming_model(self, event: Event):
        """
        We received a model. Perform one aggregate and training step.
        """
        # Accept the incoming model if no other model is being trained
        if self.available:
            # Lock
            self.available = False

            self.aggregate([(event.data["from"], event.data["model"], event.data["metadata"]["age"])])

            # Schedule a train action
            start_train_event = Event(event.time, self.index, START_TRAIN, data={
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
