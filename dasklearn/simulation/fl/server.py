from typing import List

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, START_ROUND
from dasklearn.tasks.task import Task


class FLServer(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.round: int = 0
        self.global_model = None
        self.incoming_models: List[str] = []

    def init_client(self, _: Event):
        self.logger.info("Initializing FL server")
        self.bw_scheduler.bw_limit = 10000000000000000
        start_next_round_event = Event(self.simulator.current_time, self.index, START_ROUND)
        self.simulator.schedule(start_next_round_event)

    def start_next_round(self, event: Event):
        self.round += 1
        self.incoming_models = []
        self.client_log("Server initializing round %d" % self.round)

        # Send the model to all clients
        for client_index in range(self.simulator.settings.participants):
            self.send_model(client_index, self.global_model, metadata={"round": self.round})

    def on_incoming_model(self, event: Event):
        self.incoming_models.append(event.data["model"])
        if len(self.incoming_models) == self.simulator.settings.participants:
            self.aggregate()

    def aggregate(self):
        self.client_log("Server will aggregate models in round %d (%s)" % (self.round, self.incoming_models))
        self.global_model = self.aggregate_models(self.incoming_models, self.round)

        # Should we test?
        if self.simulator.settings.test_interval > 0 and self.round % self.simulator.settings.test_interval == 0:
            self.client_log("Server testing model in round %d" % self.round)
            test_task_name = "test_%d" % self.round
            task = Task(test_task_name, "test", data={"model": self.global_model, "round": self.round, "time": self.simulator.current_time, "peer": self.index})
            self.add_compute_task(task)
            self.global_model = test_task_name

        if self.round < self.simulator.settings.rounds:
            start_next_round_event = Event(self.simulator.current_time, self.index, START_ROUND)
            self.simulator.schedule(start_next_round_event)
