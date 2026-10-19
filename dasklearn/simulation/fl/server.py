from random import Random
from typing import List

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, START_ROUND
from dasklearn.tasks.task import Task
from dasklearn.tasks.references import TaskRef


class FLServer(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.round: int = 0
        self.global_model = None
        self.incoming_models: List[str] = []
        self.sample_rng = Random(simulator.settings.seed)
        self.selected_clients = set()
        self.responded_clients = set()
        self.incoming_peers = []

    def init_client(self, _: Event):
        self.logger.info("Initializing FL server")
        self.bw_scheduler.bw_limit = 10000000000000000
        initial = self.create_compute_task("initialize",
                       {"peer": 0, "round": 0, "time": self.simulator.current_time})
        self.global_model = TaskRef(initial.name, 0)
        start_next_round_event = Event(self.simulator.current_time, self.index, START_ROUND)
        self.simulator.schedule(start_next_round_event)

    def sample_clients(self) -> List[int]:
        sample_size: int = self.simulator.settings.sample_size
        clients: List[int] = list(range(self.simulator.settings.participants))
        return self.sample_rng.sample(clients, min(self.simulator.settings.participants, sample_size))

    def start_next_round(self, event: Event):
        self.round += 1
        self.incoming_models = []
        self.incoming_peers = []
        self.responded_clients = set()
        self.client_log("Server initializing round %d" % self.round)

        # Send the model to all clients
        sampled = self.sample_clients()
        self.selected_clients = set(sampled)
        for client_index in sampled:
            self.send(client_index, self.global_model, size_bytes=self.simulator.model_size, metadata={"round": self.round}, overhead_bytes=0)

    def on_incoming_model(self, event: Event):
        peer = event.data["from"]
        if (event.data["metadata"]["round"] != self.round or peer not in self.selected_clients
                or peer in self.responded_clients):
            return
        self.responded_clients.add(peer)
        self.incoming_peers.append(peer)
        self.incoming_models.append(event.data["model"])
        if len(self.incoming_models) == min(self.simulator.settings.participants, self.simulator.settings.sample_size):
            self.aggregate()

    def aggregate(self):
        self.client_log("Server will aggregate models in round %d (%s)" % (self.round, self.incoming_models))
        self.global_model = self.aggregate_models(self.incoming_models, self.round)
        aggregation = self.simulator.workflow_dag.tasks[self.global_model[0]]
        if self.simulator.settings.aggregation_weighting == "examples":
            aggregation.data.update(weight_by_examples=True, sample_peers=list(self.incoming_peers))

        # Should we test?
        if self.simulator.settings.test_interval > 0 and self.round % self.simulator.settings.test_interval == 0:
            self.client_log("Server testing model in round %d" % self.round)
            test_task_name = "test_%d" % self.round
            task = self.create_compute_task("test", {"model": self.global_model, "round": self.round,
                "time": self.simulator.current_time, "peer": self.index}, name=test_task_name)
            self.global_model = TaskRef(test_task_name, 0)

        if self.round < self.simulator.settings.rounds:
            start_next_round_event = Event(self.simulator.current_time, self.index, START_ROUND)
            self.simulator.schedule(start_next_round_event)
