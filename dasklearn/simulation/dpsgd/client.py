from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task


class DPSGDClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.topology = None
        self.incoming_models: Dict[int, str] = {}

    def init_client(self, event: Event):
        # Schedule a train action
        self.round = 1
        start_train_event = Event(event.time, self.index, START_TRAIN)
        self.simulator.schedule(start_train_event)

    def finish_train(self, event: Event):
        """
        We finished training. Select a neighbour node and send it the model.
        """
        self.own_model = event.data["model"]
        for neighbour in list(self.simulator.topology.neighbors(self.index)):
            self.send_model(neighbour, self.own_model)

    def on_incoming_model(self, event: Event):
        """
        We received a model.
        """
        self.client_log("Client %d received from %d model %s in round %d" % (self.index, event.data["from"], event.data["model"], self.round))
        num_nb = len(list(self.simulator.topology.neighbors(self.index)))
        self.incoming_models[event.data["from"]] = event.data["model"]
        if len(self.incoming_models) == num_nb:
            # Global synchronization - check if all clients are done with the transfers
            all_done = True
            for client in self.simulator.clients:
                c_nb = len(list(self.simulator.topology.neighbors(client.index)))
                if len(client.incoming_models) < c_nb:
                    all_done = False
                    break

            if all_done:
                self.on_all_round_transfers_done()

    def on_all_round_transfers_done(self):
        for client in self.simulator.clients:
            aggregate_event = Event(self.simulator.current_time, client.index, AGGREGATE)
            self.simulator.schedule(aggregate_event)

    def aggregate(self, event: Event):
        model_names = [model_name for model_name in self.incoming_models.values()] + [self.own_model]
        self.client_log("Client %d will aggregate in round %d (%s)" % (self.index, self.round, model_names))

        self.incoming_models = {}
        self.own_model = self.aggregate_models(model_names)

        # Should we test?
        if self.round % self.simulator.settings.test_interval == 0:
            test_task_name = "test_%d_%d" % (self.index, self.round)
            task = Task(test_task_name, "test", data={"model": self.own_model, "round": self.round, "time": self.simulator.current_time, "peer": self.index})
            self.add_compute_task(task)
            self.own_model = test_task_name

        self.round += 1

        if self.round <= self.simulator.settings.rounds:
            # Schedule a train action
            start_train_event = Event(event.time, self.index, START_TRAIN)
            self.simulator.schedule(start_train_event)
