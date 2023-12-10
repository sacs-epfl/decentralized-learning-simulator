import copy
import logging
from typing import Optional

from dasklearn.model_trainer import AUGMENTATION_FACTOR_SIM
from dasklearn.simulation.bandwidth_scheduler import BWScheduler
from dasklearn.simulation.events import *
from dasklearn.functions import *
from dasklearn.tasks.task import Task


class Client:

    def __init__(self, simulator, index: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.simulator = simulator
        self.index = index
        self.bw_scheduler = BWScheduler(self)
        self.other_nodes_bws: Dict[bytes, int] = {}
        self.incoming_models: Dict[int, str] = {}

        self.simulated_speed: Optional[float] = None
        self.round: int = 1

        self.own_model: Optional[str] = None
        self.latest_task: Optional[str] = None  # Keep track of the latest task

    def add_compute_task(self, task: Task):
        self.simulator.workflow_dag.tasks[task.name] = task
        self.latest_task = task

        # Link inputs/outputs of the task
        if (task.func == "train" and task.data["model"] is not None) or task.func == "test":
            preceding_task: Task = self.simulator.workflow_dag.tasks[task.data["model"]]
            preceding_task.outputs.append(task)
            task.inputs.append(preceding_task)
        elif task.func == "aggregate":
            for _, model_name in task.data["models"].items():
                preceding_task: Task = self.simulator.workflow_dag.tasks[model_name]
                preceding_task.outputs.append(task)
                task.inputs.append(preceding_task)

    def init_model(self, event: Event):
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

        task_name = "train_%d_%d" % (self.index, self.round)
        task = Task(task_name, "train", data={"model": self.own_model, "round": self.round, "peer": self.index})
        self.add_compute_task(task)

        finish_train_event = Event(event.time + train_time, self.index, FINISH_TRAIN)
        self.simulator.schedule(finish_train_event)

    def finish_train(self, event: Event):
        """
        We finished training. Select a neighbour node and send it the model.
        """
        self.own_model = "train_%d_%d" % (self.index, self.round)
        for neighbour in list(self.simulator.topology.neighbors(self.index)) + [self.index]:
            event_data = {"from": self.index, "to": neighbour, "model": self.own_model}
            start_transfer_event = Event(event.time, self.index, START_TRANSFER, data=event_data)
            self.simulator.schedule(start_transfer_event)

    def start_transfer(self, event: Event):
        """
        We started a transfer operation. Compute how much time it takes to complete the transfer and schedule the
        completion of the transfer.
        """
        receiver_scheduler: BWScheduler = self.simulator.clients[event.data["to"]].bw_scheduler
        self.bw_scheduler.add_transfer(receiver_scheduler, self.simulator.model_size, event.data["model"])

    def finish_outgoing_transfer(self, event: Event):
        """
        An outgoing transfer has finished.
        """
        self.bw_scheduler.on_outgoing_transfer_complete(event.data["transfer"])

    def on_incoming_model(self, event: Event):
        """
        We received a model.
        """
        self.logger.debug("Client %d received from %d model %s", self.index, event.data["from"], event.data["model"])
        num_nb = len(list(self.simulator.topology.neighbors(self.index)))
        self.incoming_models[event.data["from"]] = event.data["model"]
        if len(self.incoming_models) == num_nb + 1:
            # Global synchronization - check if all clients are done with the transfers
            all_done = True
            for client in self.simulator.clients:
                c_nb = len(list(self.simulator.topology.neighbors(client.index)))
                if len(client.incoming_models) < c_nb + 1:
                    all_done = False
                    break

            if all_done:
                self.on_all_round_transfers_done()

    def on_all_round_transfers_done(self):
        for client in self.simulator.clients:
            aggregate_event = Event(self.simulator.current_time, client.index, AGGREGATE)
            self.simulator.schedule(aggregate_event)

    def aggregate(self, event: Event):
        model_names = [model_name for model_name in self.incoming_models.values()]
        self.logger.debug("Client %d will aggregate in round %d (%s)", self.index, self.round, model_names)

        models = copy.deepcopy(self.incoming_models)
        self.incoming_models = {}
        agg_task_name = "agg_%d_%d" % (self.index, self.round)
        task = Task(agg_task_name, "aggregate", data={"models": models, "round": self.round, "peer": self.index})
        self.add_compute_task(task)

        self.own_model = agg_task_name

        # Should we test?
        if self.round % self.simulator.settings.test_interval == 0:
            test_task_name = "test_%d_%d" % (self.index, self.round)
            task = Task(test_task_name, "test", data={"model": self.own_model, "round": self.round, "time": self.simulator.current_time, "peer": self.index})
            self.add_compute_task(task)
            self.own_model = test_task_name

        self.round += 1

        if self.round <= self.simulator.settings.rounds:
            # Schedule a train action
            start_train_event = Event(event.time, self.index, START_TRAIN)
            self.simulator.schedule(start_train_event)
