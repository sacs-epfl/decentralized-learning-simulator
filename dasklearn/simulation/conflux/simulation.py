from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.conflux.client import ConfluxClient
from dasklearn.simulation.events import *
from dasklearn.simulation.asynchronous_simulation import AsynchronousSimulation


class ConfluxSimulation(AsynchronousSimulation):
    CLIENT_CLASS = ConfluxClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        settings.sample_size = min(settings.sample_size, settings.participants)
        self.register_event_callback(START_ROUND, "start_round")
