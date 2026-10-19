import logging
import copy
import pickle
from typing import Optional, Any, Dict, List, Tuple
from collections import Counter

from dasklearn import AUGMENTATION_FACTOR_SIM
from dasklearn.simulation.bandwidth_scheduler import BWScheduler
from dasklearn.simulation.events import FINISH_TRAIN, Event, SEND_MESSAGE, ComputeCompletion, PayloadDelivery
from dasklearn.tasks.task import Task
from dasklearn.tasks.references import TaskRef, references
from dasklearn.tasks.operations import operation_contract
from dasklearn.simulation.protocol import ParticipantContext, Timer
from dasklearn.simulation.events import (COMPUTE_DONE, START_PAYLOAD_TRANSFER,
                                         INCOMING_PAYLOAD, PROTOCOL_TIMER)
from dasklearn.util import MICROSECONDS, time_to_sec


class BaseClient:

    def __init__(self, simulator, index: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.simulator = simulator
        self.index = index
        self.bw_scheduler = BWScheduler(self)
        self.other_nodes_bws: Dict[bytes, int] = {}
        self.online: bool = True  # Whether the node is online or not

        self.simulated_speed: float = 200  # Simulated speed in milliseconds
        self.struggler = False

        self.latest_task: Optional[Task] = None
        self.train_function: str = "train"

        self.compute_time: int = 0  # Total time spent training
        self.training_until: int = 0
        # Log of aggregations (client, model, age)
        self.aggregations: List[List[Tuple[int, str, int]]] = []
        self.incoming_counter: Dict[int, int] = Counter()
        self.context = ParticipantContext(self)
        self.resource_until = {}

    def create_compute_task(self, operation, data, *, name=None):
        """Declare numerical work; callers of this low-level API own its timing."""
        task = Task(name or Task.generate_name(operation), operation, copy.deepcopy(data),
                    operation=operation_contract(operation, data),
                    available_at=self.payload_available_at(data))
        self.simulator.workflow_dag.add(task)
        self.latest_task = task
        return task

    def compute(self, operation, inputs, *, duration, on_complete=None, params=None,
                resource="compute", feedback=False):
        """Queue an operation; duration uses integer simulated microseconds.

        Output references are available immediately for graph construction, but
        consumers and callbacks are scheduled no earlier than their availability.
        feedback=True delivers numerical values rather than references.
        """
        if type(duration) is not int or duration < 0:
            raise ValueError("Compute duration must be nonnegative integer microseconds")
        if on_complete is not None and (not isinstance(on_complete, str) or
                                         not callable(getattr(self, on_complete, None))):
            raise ValueError("on_complete must name a participant method")
        if feedback and on_complete is None:
            raise ValueError("Feedback operations require a completion handler")
        if not isinstance(resource, str) or not resource:
            raise ValueError("A compute resource needs a nonempty name")
        data = copy.deepcopy(params or {})
        if set(data) & set(inputs):
            raise ValueError("Compute inputs and params must have distinct keys")
        data.update(inputs)
        now = self.simulator.current_time
        available = self.payload_available_at(data)
        start = max(now, available, self.resource_until.get(resource, now))
        # Default compute shares the existing training resource.
        if resource == "compute":
            start = max(start, self.training_until)
        data.update(peer=self.index, time=start)
        task = self.create_compute_task(operation, data)
        task.available_at = start + duration
        self.resource_until[resource] = task.available_at
        if resource == "compute":
            self.training_until = task.available_at
        refs = tuple(TaskRef(task.name, i) for i in range(task.operation["outputs"]))
        self.simulator.schedule(Event(task.available_at, self.index, COMPUTE_DONE, {
            "task": task.name, "outputs": refs, "duration": duration,
            "callback": on_complete, "feedback": feedback, "params": copy.deepcopy(params or {}),
        }))
        return refs[0] if len(refs) == 1 else refs

    def payload_available_at(self, payload):
        times = [self.simulator.current_time]
        for ref in references(payload):
            task = self.simulator.workflow_dag.tasks.get(ref.task)
            if task is None:
                raise ValueError(f"Unknown payload reference: {ref}")
            if task.operation and not 0 <= ref.output < task.operation["outputs"]:
                raise ValueError(f"Invalid output reference: {ref}")
            times.append(task.available_at if task.available_at is not None else task.data.get("time", 0))
        return max(times)

    def on_compute_complete(self, event: Event[ComputeCompletion]):
        self.compute_time += event.data["duration"]
        self._observe("on_compute_complete", event)
        callback = event.data["callback"]
        if callback:
            getattr(self, callback)(event)

    def set_timer(self, delay, callback, data=None):
        if type(delay) is not int or delay < 0:
            raise ValueError("Timer delay must be nonnegative integer microseconds")
        if not isinstance(callback, str) or not callable(getattr(self, callback, None)):
            raise ValueError("Timer callback must name a participant method")
        timer = Timer()
        self.simulator.schedule(Event(self.context.now + delay, self.index, PROTOCOL_TIMER,
            {"timer": timer, "callback": callback, "payload": copy.deepcopy(data)}))
        return timer

    def on_protocol_timer(self, event):
        if not event.data["timer"].cancelled:
            getattr(self, event.data["callback"])(Event(event.time, self.index,
                event.data["callback"], event.data["payload"]))

    def send(self, to, payload, *, size_bytes, metadata=None, send_time=None,
             on_receive="on_incoming_model", overhead_bytes=None, immediate=False):
        """Send payload bytes plus one header and serialized metadata overhead.

        Self-delivery is a queued local event with zero wire bytes and latency.
        Ordinary mutable payloads and metadata are frozen at the call boundary.
        """
        if type(to) is not int or not 0 <= to < len(self.simulator.clients):
            raise ValueError("Unknown destination participant")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError("Payload size must be a nonnegative integer")
        if not isinstance(on_receive, str):
            raise ValueError("on_receive must name a participant method")
        if overhead_bytes is not None and (type(overhead_bytes) is not int or overhead_bytes < 0):
            raise ValueError("Explicit overhead must be a nonnegative integer")
        start = self.context.now if send_time is None else send_time
        if type(start) is not int or start < self.context.now:
            raise ValueError("Send time must be an integer at or after current time")
        start = max(start, self.payload_available_at(payload))
        metadata = copy.deepcopy(metadata or {})
        data = {"from": self.index, "to": to, "payload": copy.deepcopy(payload),
                "metadata": metadata, "size_bytes": size_bytes, "callback": on_receive,
                "overhead_bytes": overhead_bytes}
        event = Event(start, to if to == self.index else self.index,
                      INCOMING_PAYLOAD if to == self.index else START_PAYLOAD_TRANSFER, data)
        if immediate and to != self.index and start == self.context.now:
            self.start_payload_transfer(event)
        else:
            self.simulator.schedule(event)

    def start_payload_transfer(self, event):
        data = event.data
        size = (self.data_transfer_size(data["size_bytes"], data["metadata"])
                if data["overhead_bytes"] is None else data["size_bytes"] + data["overhead_bytes"])
        self.bw_scheduler.add_transfer(self.simulator.clients[data["to"]].bw_scheduler, size,
            data["payload"], data["metadata"], delivery_action=INCOMING_PAYLOAD,
            delivery_callback=data["callback"])

    def on_payload(self, event: Event[PayloadDelivery]):
        self._observe("on_payload", event)
        # Model protocols use the same delivery path as arbitrary payloads.
        event.data["model"] = event.data["payload"]
        getattr(self, event.data["callback"])(event)

    def _observe(self, method, *args):
        for observer in getattr(self.simulator, "observers", ()):
            callback = getattr(observer, method, None)
            if callback is not None:
                callback(self, *args)

    def come_online(self, event: Event):
        if not self.online:
            self.simulator.update_active_clients(True)
        self.online = True

    def go_offline(self, event: Event):
        if self.online:
            self.simulator.update_active_clients(False)
        self.online = False

    def client_log(self, msg: str):
        self.logger.info("[t=%.3f] %s", time_to_sec(self.simulator.current_time), msg)

    def get_train_time(self) -> int:
        train_time: float = 0.0
        local_steps: int = self.simulator.settings.learning.local_steps
        train_time = float(AUGMENTATION_FACTOR_SIM * local_steps * (self.simulated_speed / 1000))
        return int(train_time * MICROSECONDS)

    def start_train(self, event: Event):
        """
        We started training. Schedule when the training has ended.
        """
        if self.simulator.current_time < self.training_until:
            self.simulator.schedule(Event(self.training_until, self.index, "start_train", event.data))
            return
        self.compute(self.train_function, {"model": event.data["model"]},
                     params={"round": event.data["round"]}, duration=self.get_train_time(),
                     on_complete="training_finished")

    def training_finished(self, event):
        self.finish_train(Event(event.time, self.index, FINISH_TRAIN, {
            "model": event.data["outputs"][0], "round": event.data["params"]["round"],
            "train_time": event.data["duration"],
        }))

    def finish_outgoing_transfer(self, event: Event):
        """
        An outgoing transfer has finished.
        """
        self.bw_scheduler.on_outgoing_transfer_complete(event.data["transfer"])

    def aggregate_models(self, models: List[TaskRef], round_nr: int, weights: List[float] = None) -> TaskRef:
        task_name = Task.generate_name("agg")
        data = {"models": models, "round": round_nr, "peer": self.index}
        if weights:
            data["weights"] = weights
        task = self.create_compute_task("aggregate", data, name=task_name)
        return TaskRef(task_name, 0)

    def send_control(self, to: int, message_type: str, message: Dict, delay: int = 0) -> None:
        # Freeze the payload: later mutations of a local population view must not
        # change an already sent advertisement or an evaluation snapshot.
        if type(delay) is not int or delay < 0:
            raise ValueError("Control delay must be nonnegative integer microseconds")
        event_data = {"from": self.index, "to": to, "type": message_type, "message": copy.deepcopy(message)}
        self.send(to, event_data, size_bytes=len(pickle.dumps(event_data)),
                  overhead_bytes=self.simulator.settings.control_message_bytes,
                  send_time=self.context.now + delay, on_receive="receive_control")

    def receive_control(self, event):
        self.on_message(Event(event.time, self.index, SEND_MESSAGE, event.data["payload"]))

    def chunk_transfer_size(self, index: int, count: int) -> int:
        # Charge all bytes, including the last chunk's remainder.
        size, remainder = divmod(self.simulator.model_size, count)
        return size + (remainder if index == count - 1 else 0)

    def data_transfer_size(self, payload_size, metadata):
        return payload_size + self.simulator.settings.control_message_bytes + len(pickle.dumps(metadata))

    def publish_model(self, round_nr, model):
        previous = self.simulator.latest_models.get(self.index)
        if previous is None or round_nr >= previous[0]:
            self.simulator.latest_models[self.index] = (round_nr, model)
            self._observe("on_model_published", round_nr, model)
