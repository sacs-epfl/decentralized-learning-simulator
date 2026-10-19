import logging
from typing import Generic, TypeVar, TypedDict, Any, NotRequired, cast
from dasklearn.tasks.references import TaskRef


class ComputeCompletion(TypedDict):
    task: str
    outputs: tuple[TaskRef, ...]
    duration: int
    callback: str | None
    feedback: bool
    params: dict
    values: NotRequired[Any]


class PayloadDelivery(TypedDict):
    payload: Any
    metadata: dict
    callback: str


EventData = TypeVar('EventData')


class Event(Generic[EventData]):
    COUNTER = 0

    def __init__(self, time: int, client_id: int, action: str, data: EventData | None = None, is_global: bool = False):
        assert isinstance(time, int), "%s" % type(time)
        self.time: int = time
        self.index = Event.COUNTER
        self.client_id: int = client_id
        self.action: str = action
        self.data: EventData = cast(EventData, data if data is not None else {})
        self.is_global: bool = is_global  # Global events are not associated with a particular client and handled by the simulator

        Event.COUNTER += 1

    def __str__(self):
        return "Event(%d, %d, %s)" % (self.time, self.client_id, self.action)


# Client-specific events
INIT_CLIENT = "init_client"
START_TRAIN = "start_train"
FINISH_TRAIN = "finish_train"
TEST_SNAPSHOT = "test_snapshot"
FINISH_OUTGOING_TRANSFER = "finish_outgoing_transfer"
INCOMING_MODEL = "incoming_model"
AGGREGATE = "aggregate"
DISSEMINATE = "disseminate"
SEND_MESSAGE = "send_message"
TEST = "test"
START_ROUND = "start_round"
COMPUTE_GRADIENT = "compute_gradient"
GRADIENT_UPDATE = "gradient_update"
ONLINE = "online"
OFFLINE = "offline"
SEND_CHUNKS = "send_chunks"
FINISH_PUSH_SUM = "finish_push_sum"
COMPUTE_DONE = "compute_done"
START_PAYLOAD_TRANSFER = "start_payload_transfer"
INCOMING_PAYLOAD = "incoming_payload"
PROTOCOL_TIMER = "protocol_timer"

# Global events
MONITOR_BANDWIDTH_UTILIZATION = "monitor_bandwidth_utilization"
CHECKPOINT_DAG = "checkpoint_dag"
