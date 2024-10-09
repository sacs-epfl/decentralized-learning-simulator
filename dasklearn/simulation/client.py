import logging
from typing import Optional, Any, Dict, List, Tuple
from collections import Counter

from dasklearn import AUGMENTATION_FACTOR_SIM
from dasklearn.simulation.bandwidth_scheduler import BWScheduler
from dasklearn.simulation.events import FINISH_TRAIN, Event, START_TRANSFER
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS, time_to_sec


class BaseClient:

    def __init__(self, simulator, index: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.simulator = simulator
        self.index = index
        self.bw_scheduler = BWScheduler(self)
        self.other_nodes_bws: Dict[bytes, int] = {}

        self.simulated_speed: Optional[float] = None
        self.struggler = False

        self.latest_task: Optional[str] = None  # Keep track of the latest task
        self.train_function: str = "train"

        self.compute_time: int = 0  # Total time spent training
        # Log of aggregations (client, model, age, opportunity)
        self.aggregations: List[List[Tuple[int, str, int, Dict[int, float]]]] = []
        self.incoming_counter: Dict[int, int] = Counter()
        self.opportunity: Dict[int, float] = Counter()  # Opportunity of clients to contribute to the model

    def client_log(self, msg: str):
        self.logger.info("[t=%.3f] %s", time_to_sec(self.simulator.current_time), msg)

    def get_train_time(self) -> int:
        train_time: float = 0.0
        if self.simulated_speed:
            local_steps: int = self.simulator.settings.learning.local_steps
            batch_size: int = self.simulator.settings.learning.batch_size
            train_time = float(AUGMENTATION_FACTOR_SIM * local_steps * batch_size * (self.simulated_speed / 1000))
        return int(train_time * MICROSECONDS)

    def start_train(self, event: Event):
        """
        We started training. Schedule when the training has ended.
        """
        task_name = Task.generate_name(self.train_function)
        task = Task(task_name, self.train_function, data={
            "model": event.data["model"], "round": event.data["round"],
            "time": self.simulator.current_time, "peer": self.index})
        self.add_compute_task(task)

        train_time: int = self.get_train_time()
        finish_train_event = Event(self.simulator.current_time + train_time, self.index, FINISH_TRAIN,
                                   data={"model": task_name, "round": event.data["round"], "train_time": train_time})
        self.simulator.schedule(finish_train_event)

    def send_model(self, to: int, model: str, metadata: Optional[Dict[Any, Any]] = None, send_time: Optional[int] = None) -> None:
        metadata = metadata or {}
        event_data = {"from": self.index, "to": to, "model": model, "metadata": metadata}
        start_transfer_event = Event(send_time or self.simulator.current_time, self.index, START_TRANSFER, data=event_data)
        self.simulator.schedule(start_transfer_event)

    def start_transfer(self, event: Event):
        """
        We started a transfer operation. Compute how much time it takes to complete the transfer and schedule the
        completion of the transfer.
        """
        receiver_scheduler: BWScheduler = self.simulator.clients[event.data["to"]].bw_scheduler
        self.bw_scheduler.add_transfer(receiver_scheduler, self.simulator.model_size, event.data["model"],
                                       event.data["metadata"])

    def finish_outgoing_transfer(self, event: Event):
        """
        An outgoing transfer has finished.
        """
        self.bw_scheduler.on_outgoing_transfer_complete(event.data["transfer"])

    def aggregate_models(self, models: List[str], round_nr: int, weights: List[float] = None) -> str:
        task_name = Task.generate_name("agg")
        data = {"models": models, "round": round_nr, "peer": self.index}
        if weights:
            data["weights"] = weights
        task = Task(task_name, "aggregate", data=data)
        self.add_compute_task(task)
        return task_name

    def merge_opportunity(self, opportunities: List[Dict[int, float]], weights: Optional[List[float]] = None) -> None:
        result_opportunity: Dict[int, float] = Counter()
        if weights is None:
            weights = [1 / len(opportunities)] * len(opportunities)
        for cont_dict, weight in zip(opportunities, weights):
            for client, opportunity in cont_dict.items():
                result_opportunity[client] += (opportunity * weight)
        self.opportunity = result_opportunity

    def add_compute_task(self, task: Task):
        self.simulator.workflow_dag.tasks[task.name] = task
        self.latest_task = task
        # Link inputs/outputs of the task
        if (task.func == "train" and task.data["model"] is not None) or task.func == "test" or \
                (task.func == "compute_gradient" and task.data["model"] is not None) or task.func == "gradient_update":
            preceding_task: Task = self.simulator.workflow_dag.tasks[task.data["model"]]
            preceding_task.outputs.append(task)
            task.inputs.append(preceding_task)
            if task.func == "gradient_update" and task.data["gradient_model"] != task.data["model"]:
                preceding_task: Task = self.simulator.workflow_dag.tasks[task.data["gradient_model"]]
                preceding_task.outputs.append(task)
                task.inputs.append(preceding_task)
        elif task.func == "aggregate":
            for model_name in task.data["models"]:
                preceding_task: Task = self.simulator.workflow_dag.tasks[model_name]
                preceding_task.outputs.append(task)
                task.inputs.append(preceding_task)
