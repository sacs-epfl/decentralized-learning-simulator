from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.events import *
from dasklearn.simulation.simulation import Simulation
from dasklearn.simulation.subset.client import SubsetDLClient


class SubsetDLSimulation(Simulation):
    CLIENT_CLASS = SubsetDLClient

    def __init__(self, settings: SessionSettings, sample_size: int):
        super().__init__(settings)
        self.sample_size: int = sample_size
        self.register_event_callback(START_ROUND, "start_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")
