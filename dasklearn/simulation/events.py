import logging
from typing import Dict


class Event:
    COUNTER = 0

    def __init__(self, time: int, client_id: int, action: str, data: Dict = None):
        assert isinstance(time, int), "%s" % type(time)
        self.time: int = time
        self.index = Event.COUNTER
        self.client_id: int = client_id
        self.action: str = action
        self.data: Dict = data or {}

        Event.COUNTER += 1

    def __str__(self):
        return "Event(%d, %d, %s)" % (self.time, self.client_id, self.action)


INIT_CLIENT = "init_client"
START_TRAIN = "start_train"
FINISH_TRAIN = "finish_train"
START_TRANSFER = "start_transfer"
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
