from typing import Optional

from dasklearn.model_trainer import AUGMENTATION_FACTOR_SIM
from dasklearn.models import create_model
from dasklearn.simulation.bandwidth_scheduler import BWScheduler
from dasklearn.simulation.events import *


class Client:

    def __init__(self, simulator, index: int):
        self.simulator = simulator
        self.initial_model = None
        self.index = index
        self.bw_scheduler = BWScheduler(self)
        self.other_nodes_bws: Dict[bytes, int] = {}

        self.simulated_speed: Optional[float] = None
        self.round: int = 1
        self.models_received: int = 0

    def init_model(self, event: Event):
        self.initial_model = create_model("cifar10", architecture=self.simulator.settings.model)

        # Schedule a train action
        start_train_event = Event(event.time, self.index, START_TRAIN)
        self.simulator.schedule(start_train_event)

    def start_train(self, event: Event):
        """
        We started training. Schedule when the training has ended.
        """
        train_time: float = 0.0
        if self.simulated_speed:
            local_steps: int = self.simulator.settings.learning.local_steps
            batch_size: int = self.simulator.settings.learning.batch_size
            train_time = AUGMENTATION_FACTOR_SIM * local_steps * batch_size * (self.simulated_speed / 1000)

        finish_train_event = Event(event.time + train_time, self.index, FINISH_TRAIN)
        self.simulator.schedule(finish_train_event)

    def finish_train(self, event: Event):
        """
        We finished training. Select a neighbour node and send it the model.
        """
        for neighbour in self.simulator.topology.neighbors(self.index):
            start_transfer_event = Event(event.time, self.index, START_TRANSFER, {"from": self.index, "to": neighbour})
            self.simulator.schedule(start_transfer_event)

    def start_transfer(self, event: Event):
        """
        We started a transfer operation. Compute how much time it takes to complete the transfer and schedule the
        completion of the transfer.
        """
        receiver_scheduler: BWScheduler = self.simulator.clients[event.data["to"]].bw_scheduler
        self.bw_scheduler.add_transfer(receiver_scheduler, self.simulator.model_size)

    def finish_outgoing_transfer(self, event: Event):
        """
        An outgoing transfer has finished.
        """
        self.bw_scheduler.on_outgoing_transfer_complete(event.data["transfer"])

    def on_incoming_model(self, event: Event):
        """
        We received a model.
        """
        num_nb = len(list(self.simulator.topology.neighbors(self.index)))
        self.models_received += 1
        if self.models_received == num_nb:
            # Wrap up this round
            aggregate_event = Event(self.simulator.current_time, self.index, AGGREGATE)
            self.simulator.schedule(aggregate_event)

    def aggregate(self, event: Event):
        self.round += 1
        self.models_received = 0

        if self.round <= self.simulator.settings.rounds:
            # Schedule a train action
            start_train_event = Event(event.time, self.index, START_TRAIN)
            self.simulator.schedule(start_train_event)
