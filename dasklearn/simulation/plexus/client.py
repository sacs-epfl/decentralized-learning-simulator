"""Message-driven implementation of Plexus paper Algorithms 1 and 2.

Only a received aggregate starts training. Each full trained model goes directly
to the highest-bandwidth peer in that round's sample; the first floor(s * sf)
distinct uploads form one common uniform average for the next sample.
"""
from collections import defaultdict
from math import floor

from dasklearn.simulation.client import BaseClient
from dasklearn.tasks.references import TaskRef

from .sampling import sample


class PlexusClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.started_rounds = set()
        self.latest_received_round = 0
        self.completed_aggregations = set()
        self.uploads = defaultdict(dict)

    def sample(self, round_nr):
        return sample(round_nr, self.peer_ids, self.simulator.settings.sample_size)

    def aggregator(self, round_nr):
        return max(self.sample(round_nr), key=self.other_nodes_bws.__getitem__)

    def init_client(self, event):
        if self.index in self.sample(1):
            if self.simulator.initial_model is None:
                # Reuse the shared initial checkpoint when snapshot evaluation is on.
                if self.simulator.latest_models:
                    self.simulator.initial_model = next(iter(self.simulator.latest_models.values()))[1]
                else:
                    task = self.create_compute_task("initialize", {
                        "peer": self.index, "round": 0, "time": self.context.now})
                    self.simulator.initial_model = TaskRef(task.name, 0)
            self.begin_round(1, self.simulator.initial_model)

    def begin_round(self, round_nr, model):
        if round_nr in self.started_rounds:
            return
        self.started_rounds.add(round_nr)
        # Shared compute scheduling serializes overlapping partial rounds locally.
        self.compute("train", {"model": model}, params={"round": round_nr},
                     duration=self.get_train_time(), on_complete="trained")

    def trained(self, event):
        round_nr = event.data["params"]["round"]
        self.send(self.aggregator(round_nr), event.data["outputs"][0],
                  size_bytes=self.simulator.model_size,
                  metadata={"phase": "upload", "round": round_nr},
                  on_receive="received_upload")

    def received_upload(self, event):
        round_nr = event.data["metadata"]["round"]
        sender = event.data["from"]
        if (round_nr < 1 or self.index != self.aggregator(round_nr)
                or sender not in self.sample(round_nr)
                or round_nr in self.completed_aggregations):
            return
        inbox = self.uploads[round_nr]
        if sender in inbox:
            return
        inbox[sender] = event.data["payload"]
        required = max(1, floor(self.simulator.settings.sample_size * self.simulator.settings.success_fraction))
        if len(inbox) < required:
            return
        self.completed_aggregations.add(round_nr)
        # Canonical summation order; membership is frozen by actual upload arrival.
        contributors = tuple(sorted(inbox))
        aggregate = self.create_compute_task("aggregate", {
            "models": [inbox[peer] for peer in contributors],
            "round": round_nr, "peer": self.index, "time": self.context.now,
        })
        self.simulator.round_records.append({
            "round": round_nr, "aggregator": self.index, "time_us": self.context.now,
            "sample": self.sample(round_nr), "contributors": contributors,
            "next_sample": self.sample(round_nr + 1), "aggregate_task": aggregate.name,
        })
        del self.uploads[round_nr]
        # Full model to every recipient, sharing the aggregator's finite endpoint.
        for peer in self.sample(round_nr + 1):
            self.send(peer, TaskRef(aggregate.name, 0), size_bytes=self.simulator.model_size,
                      metadata={"phase": "aggregate", "round": round_nr},
                      on_receive="received_aggregate")

    def received_aggregate(self, event):
        round_nr = event.data["metadata"]["round"]
        if (round_nr < 1 or event.data["from"] != self.aggregator(round_nr)
                or self.index not in self.sample(round_nr + 1)
                or round_nr <= self.latest_received_round):
            return
        self.latest_received_round = round_nr
        model = event.data["payload"]
        self.publish_model(round_nr, model)
        self.simulator.delivery_records.append((round_nr, self.index, self.context.now))
        if (self.simulator.settings.test_method == "individual"
                and self.simulator.settings.test_interval > 0
                and round_nr % self.simulator.settings.test_interval == 0):
            self.create_compute_task("test", {
                "model": model, "round": round_nr, "peer": self.index, "time": self.context.now,
            })
        if self.simulator.settings.rounds <= 0 or round_nr < self.simulator.settings.rounds:
            self.begin_round(round_nr + 1, model)
