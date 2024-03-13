from typing import Optional

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, START_TRAIN
from dasklearn.simulation.fl.server import FLServer


class FLClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.server: Optional[FLServer] = None

    def init_client(self, _: Event):
        pass

    def on_incoming_model(self, event: Event):
        start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN, data={
            "model": event.data["model"], "round": event.data["metadata"]["round"]})
        self.simulator.schedule(start_train_event)

    def finish_train(self, event: Event):
        """
        We're done training - send the model back to the server.
        """
        model: str = event.data["model"]
        self.send_model(self.server.index, model, metadata={"round": event.data["round"]})
