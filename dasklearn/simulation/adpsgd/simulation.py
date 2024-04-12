import random

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.adpsgd.client import ADPSGDClient
from dasklearn.simulation.events import *
from dasklearn.simulation.simulation import Simulation


class ADPSGDSimulation(Simulation):
    CLIENT_CLASS = ADPSGDClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        if settings.agg == "default":
            settings.agg = "average"
        random.seed(settings.seed)

        # randomly choose active and passive peers
        peers = list(range(settings.participants))
        self.active_peers = random.sample(peers, len(peers) // 2)
        self.passive_peers = list(filter(lambda x: x not in self.active_peers, peers))

        self.register_event_callback(FINISH_TRAIN, "finish_train")
        self.register_event_callback(COMPUTE_GRADIENT, "compute_gradient")
        self.register_event_callback(GRADIENT_UPDATE, "gradient_update")
        self.register_event_callback(TEST, "test")

    def initialize_clients(self):
        super().initialize_clients()
        for client in self.clients:
            client.active = client.index in self.active_peers
            self.logger.info("Client %d is %s peer" % (client.index, "an active" if client.active else "a passive"))

    def schedule(self, event: Event):
        # Don't schedule events after the running duration elapses, transfers are allowed to satisfy sanity checks,
        # but their results are not considered
        if event.time > self.settings.duration and event.action != "start_transfer" and event.action != "finish_outgoing_transfer":
            return
        super().schedule(event)

    def get_random_participant(self, active: bool):
        """
        Returns a random participant from the other group (active/passive)
        """
        if active:
            return random.choice(self.passive_peers)
        else:
            return random.choice(self.active_peers)
