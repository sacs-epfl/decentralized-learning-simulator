from typing import Optional

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.events import Event, INIT_CLIENT, START_ROUND, FINISH_TRAIN
from dasklearn.simulation.fl.client import FLClient
from dasklearn.simulation.fl.server import FLServer
from dasklearn.simulation.simulation import Simulation


class FLSimulation(Simulation):
    CLIENT_CLASS = FLClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        self.server: Optional[FLServer] = None

        self.register_event_callback(START_ROUND, "start_next_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")

    def initialize_clients(self):
        super().initialize_clients()

        # Setup the server
        server_id: int = len(self.clients)
        self.server = FLServer(self, server_id)

        for client in self.clients:
            client.server = self.server

        self.clients.append(self.server)
        init_client_event = Event(0, server_id, INIT_CLIENT)
        self.schedule(init_client_event)
