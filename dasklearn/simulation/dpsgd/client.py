from typing import Optional, Tuple, List

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.dpsgd.round import Round
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS


class DPSGDClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.topology = None
        self.round_info: Dict[int, Round] = {}

    def init_client(self, event: Event):
        self.schedule_next_round({"round": 1, "model": (None, 0)})
        # Schedule time based testing if we use time based stopping
        if self.simulator.settings.stop == "duration" and self.simulator.settings.test_period > 0:
            test_event = Event(event.time + self.simulator.settings.test_period, self.index, TEST)
            self.simulator.schedule(test_event)

    def schedule_next_round(self, round_data: Dict):
        if self.simulator.settings.rounds > 0 and round_data["round"] > self.simulator.settings.rounds:
            return
        
        round_nr: int = round_data["round"]
        model: Tuple[Optional[str], int] = round_data["model"]
        incoming_models: Dict[int, Tuple[str, List[float]]] = round_data["incoming_models"]\
            if "incoming_models" in round_data else {}

        new_round = Round(round_nr)
        new_round.model = model
        new_round.incoming_models = incoming_models
        new_round.should_ignore = not self.online
        self.round_info[round_nr] = new_round

        is_sync: bool = self.simulator.settings.synchronous
        if not is_sync:
            start_round_event = Event(self.simulator.current_time, self.index, START_ROUND, data=round_data)
            self.simulator.schedule(start_round_event)
        else:
            # We operate in synchronous mode, so the start of the next round is initiated by the simulator.
            self.simulator.client_ready_for_round(self.index, round_data["round"], round_data)

    def start_round(self, event: Event):
        round_nr: int = event.data["round"]

        if round_nr not in self.round_info:
            return

        round_info: Round = self.round_info[round_nr]

        self.client_log("Client %d starting round %d" % (self.index, round_nr))

        if round_info.model and not round_info.is_training:
            self.schedule_train(round_info)

    def schedule_train(self, round_info: Round):
        cur_round: int = round_info.round_nr
        round_info.is_training = True
        if not round_info.should_ignore:
            start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN, data={
                "model": round_info.model, "round": cur_round})
            self.start_train(start_train_event)
        else:
            # We should ignore this round, but we still need to notify the simulator that we're done training.
            finish_train_event = Event(self.simulator.current_time, self.index, FINISH_TRAIN, data={
                "round": cur_round, "model": round_info.model, "train_time": 0})
            self.finish_train(finish_train_event)

    def finish_train(self, event: Event):
        """
        We finished training. Select a neighbour node and send it the model.
        """
        cur_round: int = event.data["round"]
        self.compute_time += event.data["train_time"]
        self.client_log("Client %d finished model training in round %d (online? %s)" % (self.index, cur_round, self.online))
        if cur_round not in self.round_info:
            raise RuntimeError("Client %d does not know about round %d after training finished!" %
                               (self.index, cur_round))

        round_info: Round = self.round_info[cur_round]
        round_info.model = event.data["model"]
        round_info.is_training = False
        round_info.train_done = True
        
        self.send_model_to_nbs(round_info)

        # Do we have all incoming models for this round? If so, aggregate.
        num_nb = len(list(self.simulator.topologies[cur_round - 1].predecessors(self.index)))
        if len(round_info.incoming_models) == num_nb:
            aggregate_event = Event(self.simulator.current_time, self.index, AGGREGATE, data={"round": cur_round})
            self.aggregate(aggregate_event)

    def send_model_to_nbs(self, round_info):
        cur_round = round_info.round_nr
        if len(self.simulator.topologies) < cur_round:
            self.simulator.add_topology()
        for neighbour in self.simulator.topologies[round_info.round_nr - 1].successors(self.index):
            if not round_info.should_ignore and self.simulator.clients[neighbour].online:
                self.send_model(neighbour, round_info.model, metadata={"round": cur_round})
            else:
                self.simulator.clients[neighbour].on_offline_sentinel(self.index, cur_round)

    def on_offline_sentinel(self, from_client: int, round_nr: int):
        event = {
            "from": from_client,
            "model": None,
            "metadata": {"round": round_nr},
        }
        self.on_incoming_model(Event(self.simulator.current_time, self.index, INCOMING_MODEL, data=event))

    def on_incoming_model(self, event: Event):
        """
        We received a model.
        """
        round_nr: int = event.data["metadata"]["round"]
        self.client_log("Client %d received from %d model %s in round %d" %
                        (self.index, event.data["from"], event.data["model"], round_nr))
        self.incoming_counter[event.data["from"]] += 1

        if round_nr not in self.round_info:
            # We do not know about this round yet - schedule it to start.
            round_data: Dict = {"round": round_nr, "model": None,
                                "incoming_models": {event.data["from"]: event.data["model"]}}
            self.schedule_next_round(round_data)
        else:
            round_info: Round = self.round_info[round_nr]

            if len(self.simulator.topologies) < round_nr:
                self.simulator.add_topology()
            num_nb = len(list(self.simulator.topologies[round_nr - 1].predecessors(self.index)))
            round_info.incoming_models[event.data["from"]] = event.data["model"]

            # Are we done training our own model in this round and have we received all nb models?
            # If so, aggregate everything. Otherwise, wait until we are done training.
            if round_info.train_done and len(round_info.incoming_models) == num_nb:
                aggregate_event = Event(self.simulator.current_time, self.index, AGGREGATE, data={"round": round_nr})
                self.aggregate(aggregate_event)

    def aggregate(self, event: Event):
        round_nr = event.data["round"]
        round_info: Round = self.round_info[round_nr]
        model_names = [m for m in round_info.incoming_models.values() if m] + [round_info.model]
        self.client_log("Client %d will aggregate in round %d (%s)" % (self.index, round_nr, model_names))
        if not round_info.should_ignore:
            other = [(sender_id, model_name, round_nr)
                    for sender_id, model_name in round_info.incoming_models.items()]
            self.aggregations.append(other + [(self.index, round_info.model, round_nr)])
            round_info.model = self.aggregate_models(model_names, round_nr)

            # Should we test?
            if self.simulator.settings.stop == "rounds" and self.simulator.settings.test_interval > 0 \
                    and round_nr % self.simulator.settings.test_interval == 0:
                test_task_name = "test_%d_%d" % (self.index, round_nr)
                task = Task(test_task_name, "test", data={"model": round_info.model, "round": round_nr, "time": self.simulator.current_time, "peer": self.index})
                self.add_compute_task(task)
                round_info.model = (test_task_name, 0)

        self.logger.debug("Client %d finished round %d (t=%d)" % (self.index, round_nr, self.simulator.current_time // MICROSECONDS))
        self.last_round_completed = round_nr
        self.round_info.pop(round_nr)

        next_round_nr: int = round_nr + 1
        if next_round_nr not in self.round_info:
            self.schedule_next_round({"round": next_round_nr, "model": round_info.model})
        else:
            next_round_info: Round = self.round_info[next_round_nr]
            if next_round_info.model is None and not next_round_info.is_training:
                next_round_info.model = round_info.model
                self.schedule_train(next_round_info)

    def test(self, event: Event):
        """
        Test model's performance
        """
        # Find latest round with model
        latest_model_round: int = -1
        for round_nr, info in self.round_info.items():
            if info.model is not None:
                latest_model_round = max(latest_model_round, round_nr)
        # Test the model if it exists
        if latest_model_round > -1:
            self.client_log("Client %d will test its model %s" % (self.index, self.round_info[latest_model_round].model))
            test_task_name = "test_%d_%d" % (self.index, event.time // MICROSECONDS)
            task = Task(test_task_name, "test", data={
                "model": self.round_info[latest_model_round].model, "time": self.simulator.current_time,
                "peer": self.index, "round": latest_model_round})
            self.add_compute_task(task)

        # Schedule next test action
        test_event = Event(event.time + self.simulator.settings.test_period, self.index, TEST)
        self.simulator.schedule(test_event)
