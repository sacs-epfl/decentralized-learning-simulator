import csv
import os
from dasklearn.simulation.simulation import Simulation
from dasklearn.simulation.conflux_v2.client import ConfluxV2Client
from dasklearn.simulation.events import START_ROUND, FINISH_TRAIN


class ConfluxV2Simulation(Simulation):
    CLIENT_CLASS = ConfluxV2Client

    def __init__(self, settings):
        if settings.churn != "none":
            raise ValueError("Conflux v2 currently requires stable membership (use --churn none).")
        if not 0 < settings.success_fraction <= 1 or settings.sample_size < 1 or settings.chunks_in_sample < 1 or settings.relay_count < 0:
            raise ValueError("Invalid Conflux v2 sample, chunk, relay or success-fraction setting")
        if settings.test_method == "global":
            raise ValueError("Use --test-method snapshot or individual for Conflux v2")
        settings.sample_size = min(settings.sample_size, settings.participants)
        super().__init__(settings)
        self.contributor_sets = {}
        self.completed_rounds = []
        self.register_event_callback(START_ROUND, "start_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")

    def initialize_clients(self):
        capacities = {client.index: client.bw_scheduler.bw_limit for client in self.clients}
        if min(capacities.values()) <= 0:
            raise ValueError("Conflux v2 requires positive bandwidth capacities")
        for client in self.clients:
            client.other_nodes_bws = capacities.copy()
        super().initialize_clients()

    def save_measurements(self):
        super().save_measurements()
        with open(os.path.join(self.data_dir, "conflux_v2_rounds.csv"), "w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["round", "client", "time_us", "contributors"])
            for round_nr, client, time in self.completed_rounds:
                writer.writerow([round_nr, client, time, " ".join(map(str, self.contributor_sets[round_nr]))])
