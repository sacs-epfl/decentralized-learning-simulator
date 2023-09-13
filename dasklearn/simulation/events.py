from functools import total_ordering
from typing import NamedTuple, Dict


@total_ordering
class Event(NamedTuple):
    time: float
    client_id: int
    action: str
    data: Dict = {}

    def __lt__(self, other: 'Event') -> bool:
        # Compare primarily by time, then by client_id and action for tiebreakers.
        # This logic can be adjusted based on the intended ordering.
        return self.time < other.time

    # Define __eq__ as well, since total_ordering requires it.
    def __eq__(self, other: 'Event') -> bool:
        return (self.time, self.client_id, self.action) == (other.time, other.client_id, other.action)


MODEL_INIT = "model_init"
START_TRAIN = "start_train"
FINISH_TRAIN = "finish_train"
START_TRANSFER = "start_transfer"
FINISH_OUTGOING_TRANSFER = "finish_outgoing_transfer"
INCOMING_MODEL = "incoming_model"
AGGREGATE = "aggregate"
