import os
import random
from typing import Set

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.adpsgd.client import ADPSGDClient
from dasklearn.simulation.asynchronous_simulation import AsynchronousSimulation


class ADPSGDSimulation(AsynchronousSimulation):
    CLIENT_CLASS = ADPSGDClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        if settings.agg == "default":
            settings.agg = "average"

        # Randomly choose active and passive peers
        peers = list(range(settings.participants))
        self.active_peers = random.sample(peers, len(peers) // 2)
        self.passive_peers = list(filter(lambda x: x not in self.active_peers, peers))

    def initialize_clients(self):
        super().initialize_clients()
        for client in self.clients:
            client.active = client.index in self.active_peers

    def get_send_set(self, index: int) -> Set[int]:
        """
        Returns a random passive client
        """
        return {random.choice(self.passive_peers)}

    def save_measurements(self) -> None:
        super().save_measurements()
        # Write active/passive log
        with open(os.path.join(self.data_dir, "active_passive.csv"), "w") as file:
            file.write("client,active\n")
            for client in self.clients:
                # Supports only duration based algorithms
                file.write("%d,%s\n" % (client.index, client.active))
