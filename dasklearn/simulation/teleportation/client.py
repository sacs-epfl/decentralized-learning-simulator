from typing import List
from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.teleportation.round import Round
from dasklearn.simulation.teleportation.sample_manager import SampleManager
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task


class TeleportationClient(AsynchronousClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.round_info: Dict[int, Round] = {}

    def init_client(self, _: Event):
        sample: List[int] = SampleManager.get_sample(1, len(self.simulator.clients), self.simulator.settings.sample_size)

        if self.index in sample:
            round = Round(1)
            start_round_event = Event(self.simulator.current_time, self.index, START_ROUND, data=round)
            self.simulator.schedule(start_round_event)

    async def start_round(self, event: Event):
        round_info: Round = event.data
        self.round_info[round_info.round_nr] = round_info
        round_nr: int = round_info.round_nr
        if round_nr < 1:
            raise RuntimeError("Round number %d invalid!" % round_nr)
        
        # Should we test the model?
        if round_nr > 1 and (round_nr - 1) % self.simulator.settings.test_interval == 0:
            test_task_name = "test_%d_%d" % (self.index, round_nr)
            task = Task(test_task_name, "test", data={"model": round_info.model, "round": round_nr, "time": self.simulator.current_time, "peer": self.index})
            self.add_compute_task(task)
            round_info.model = (test_task_name, 0)

        self.client_log(f"Node {self.index} starts training in round {round_nr}")

        # 1. Train the model
        self.schedule_train(round_info)
        round_info.model = await round_info.train_future
        round_info.is_training = False
        round_info.train_done = True

        # 2. Start sharing the models in the G_k topology related ot the currente sample
        sample: List[int] = SampleManager.get_sample(round_nr, len(self.simulator.clients), self.simulator.settings.sample_size)
        index_in_sample: int = sample.index(self.index)
        nbs = list(self.simulator.G_k.neighbors(index_in_sample))
        
        # Send the model to the neighbors
        for nb_idx_in_graph in nbs:
            nb = sample[nb_idx_in_graph]
            self.send_model(nb, round_info.model, metadata={"round": round_nr, "type": "in_sample"})

    def on_incoming_model(self, event: Event):
        if event.data["metadata"]["type"] == "in_sample":
            self.handle_incoming_model_in_sample(event)
        elif event.data["metadata"]["type"] == "next_sample":
            self.handle_incoming_model_next_sample(event)

    def handle_incoming_model_in_sample(self, event: Event):
        model_round = event.data["metadata"]["round"]
        self.client_log("Peer %d received in-sample model from %d in round %d" %
                        (self.index, event.data["from"], model_round))
        current_sample = SampleManager.get_sample(model_round, len(self.simulator.clients), self.simulator.settings.sample_size)
        if self.index not in current_sample:
            raise RuntimeError("Client %d not in sample of round %d!" % (self.index, model_round))

        if model_round not in self.round_info:
            # We were not activated yet by the previous sample - start a new round
            round_info: Round = Round(model_round)
            self.round_info[model_round] = round_info
        else:
            # We are currently working on this round.
            round_info: Round = self.round_info[model_round]
        round_info.incoming_models[event.data["from"]] = event.data["model"]

        # Do we have all the models?
        index_in_sample: int = current_sample.index(self.index)
        if len(round_info.incoming_models) == len(list(self.simulator.G_k.neighbors(index_in_sample))):
            # We are done with training in this round - aggregate and send the model to the next sample.
            round_info.model = self.aggregate_models(
                list(round_info.incoming_models.values()) + [round_info.model], round_info.round_nr)
            
            # Send it to the next sample
            next_sample: int = SampleManager.get_sample(model_round + 1, len(self.simulator.clients), self.simulator.settings.sample_size)
            self.send_model(next_sample[index_in_sample], round_info.model, metadata={"round": model_round + 1, "type": "next_sample"})

    def handle_incoming_model_next_sample(self, event: Event):
        model_round = event.data["metadata"]["round"]
        self.client_log("Peer %d received next-sample model from %d for round %d" %
                        (self.index, event.data["from"], model_round))
        current_sample = SampleManager.get_sample(model_round, len(self.simulator.clients), self.simulator.settings.sample_size)
        if self.index not in current_sample:
            raise RuntimeError("Client %d not in sample of round %d!" % (self.index, model_round))

        next_round: Round = Round(model_round)
        next_round.model = event.data["model"]
        self.round_info[model_round + 1] = next_round
        start_round_event = Event(self.simulator.current_time, self.index, START_ROUND, data=next_round)
        self.simulator.schedule(start_round_event)

    def schedule_train(self, round_info: Round):
        round_info.is_training = True
        start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN, data={
            "model": round_info.model, "round": round_info.round_nr})
        self.simulator.schedule(start_train_event)

    def finish_train(self, event: Event):
        """
        We finished training.
        """
        cur_round: int = event.data["round"]
        round_info: Round = self.round_info[cur_round]
        round_info.is_training = False
        round_info.train_done = True
        round_info.train_future.set_result(event.data["model"])
