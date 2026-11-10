import csv
import math
from pathlib import Path

from dasklearn.simulation.simulation import Simulation

from .client import PlexusClient


class PlexusSimulation(Simulation):
    CLIENT_CLASS = PlexusClient

    def __init__(self, settings):
        if (type(settings.participants) is not int or settings.participants < 1
                or type(settings.sample_size) is not int
                or not 1 <= settings.sample_size <= settings.participants):
            raise ValueError("Plexus sample_size must be in [1, participants]")
        if not math.isfinite(settings.success_fraction) or not 0 < settings.success_fraction <= 1:
            raise ValueError("Plexus success_fraction must be in (0, 1]")
        if settings.learning.local_steps < 1 or settings.learning.momentum != 0:
            raise ValueError("Plexus requires positive local_steps and zero trainer momentum")
        if settings.churn != "none":
            raise ValueError("Plexus currently requires stable membership")
        if settings.synchronous:
            raise ValueError("Plexus is message-driven; synchronous barriers are unsupported")
        if settings.test_method == "global":
            raise ValueError("Use snapshot or individual evaluation for Plexus")
        super().__init__(settings)
        self.initial_model = None
        self.round_records = []
        self.delivery_records = []

    def initialize_clients(self):
        capacities = {client.index: client.bw_scheduler.bw_limit for client in self.clients}
        if any(not math.isfinite(value) or value <= 0 for value in capacities.values()):
            raise ValueError("Plexus requires positive finite bandwidth capacities")
        identities = tuple(f"plexus:{self.settings.seed}:{peer}".encode()
                           for peer in range(self.settings.participants))
        for client in self.clients:
            client.peer_ids = identities
            client.other_nodes_bws = capacities.copy()
        super().initialize_clients()

    def save_measurements(self):
        super().save_measurements()
        with (Path(self.data_dir) / "plexus-rounds.csv").open("w", newline="") as stream:
            fields = ["round", "aggregator", "time_us", "sample", "contributors", "next_sample", "aggregate_task"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in self.round_records:
                writer.writerow({key: " ".join(map(str, value)) if isinstance(value, tuple) else value
                                 for key, value in record.items()})
        with (Path(self.data_dir) / "plexus-deliveries.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["round", "client", "time_us"])
            writer.writerows(self.delivery_records)
