from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.dpsgd.round import Round
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task


class DPSGDClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.topology = None
        self.round_info: Dict[int, Round] = {}

    def init_client(self, _: Event):
        self.schedule_next_round({"round": 1, "model": None})

    def schedule_next_round(self, round_data: Dict):
        if round_data["round"] <= self.simulator.settings.rounds:
            is_sync: bool = self.simulator.settings.synchronous
            if not is_sync:
                start_round_event = Event(self.simulator.current_time, self.index, START_ROUND, data=round_data)
                self.simulator.schedule(start_round_event)
            else:
                # We operate in synchronous mode, so the start of the next round is initiated by the simulator.
                self.simulator.client_ready_for_round(self.index, round_data["round"], round_data)

    def start_round(self, event: Event):
        round_nr: int = event.data["round"]
        self.client_log("Peer %d starting round %d" % (self.index, round_nr))

        new_round = Round(round_nr)
        new_round.model = event.data["model"]
        new_round.incoming_models = event.data["incoming_models"] if "incoming_models" in event.data else {}
        self.round_info[round_nr] = new_round

        if new_round.model or round_nr == 1:
            # Start training if we have the next-sample model or if it's the first round
            self.schedule_train(new_round)

    def schedule_train(self, round_info: Round):
        round_info.is_training = True
        start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN, data={
            "model": round_info.model, "round": round_info.round_nr})
        self.simulator.schedule(start_train_event)

    def finish_train(self, event: Event):
        """
        We finished training. Select a neighbour node and send it the model.
        """
        cur_round: int = event.data["round"]
        self.client_log("Client %d finished model training in round %d" % (self.index, cur_round))
        if cur_round not in self.round_info:
            raise RuntimeError("Peer %d does not know about round %d after training finished!" %
                               (self.index, cur_round))

        round_info: Round = self.round_info[cur_round]
        round_info.model = event.data["model"]
        round_info.is_training = False
        round_info.train_done = True

        for index, neighbour in enumerate(list(self.simulator.topology.neighbors(self.index))):
            send_time = self.simulator.current_time + 0.001 * index
            self.send_model(neighbour, event.data["model"], metadata={"round": event.data["round"]}, send_time=send_time)

        # Do we have all incoming models for this round? If so, aggregate.
        num_nb = len(list(self.simulator.topology.neighbors(self.index)))
        if round_info.train_done and len(round_info.incoming_models) == num_nb:
            aggregate_event = Event(self.simulator.current_time, self.index, AGGREGATE, data={"round": cur_round})
            self.simulator.schedule(aggregate_event)

    def on_incoming_model(self, event: Event):
        """
        We received a model.
        """
        round_nr: int = event.data["metadata"]["round"]
        round_info: Round = self.round_info[round_nr]
        self.client_log("Client %d received from %d model %s in round %d" %
                        (self.index, event.data["from"], event.data["model"], round_nr))

        num_nb = len(list(self.simulator.topology.neighbors(self.index)))
        round_info.incoming_models[event.data["from"]] = event.data["model"]

        # Are we done training our own model in this round? If so, aggregate everything. Otherwise, wait until we are
        # done training.
        if round_info.train_done and len(round_info.incoming_models) == num_nb:
            aggregate_event = Event(self.simulator.current_time, self.index, AGGREGATE, data={"round": round_nr})
            self.simulator.schedule(aggregate_event)

    def aggregate(self, event: Event):
        round_nr = event.data["round"]
        round_info: Round = self.round_info[round_nr]
        model_names = [model_name for model_name in round_info.incoming_models.values()] + [round_info.model]
        self.client_log("Client %d will aggregate in round %d (%s)" % (self.index, round_nr, model_names))
        round_info.model = self.aggregate_models(model_names, round_nr)

        # Should we test?
        if round_nr % self.simulator.settings.test_interval == 0:
            test_task_name = "test_%d_%d" % (self.index, round_nr)
            task = Task(test_task_name, "test", data={"model": round_info.model, "round": round_nr, "time": self.simulator.current_time, "peer": self.index})
            self.add_compute_task(task)
            round_info.model = test_task_name

        self.round_info.pop(round_nr)

        # Schedule the start of the next round
        self.schedule_next_round({"round": round_nr + 1, "model": round_info.model})
