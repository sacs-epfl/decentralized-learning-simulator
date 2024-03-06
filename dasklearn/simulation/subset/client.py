from typing import List

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import *
from dasklearn.simulation.subset.round import Round
from dasklearn.simulation.subset.sample_manager import SampleManager


class SubsetDLClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.round_info: Dict[int, Round] = {}

    def init_client(self, event: Event):
        self.schedule_next_round({"round": 1, "model": None})

    def start_round(self, event: Event):
        round_nr: int = event.data["round"]
        sample: List[int] = SampleManager.get_sample(round_nr, len(self.simulator.clients), self.simulator.sample_size)
        if self.index in sample:
            self.client_log("Peer %d starting round %d" % (self.index, round_nr))

            round_nr: int = event.data["round"]
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

    def schedule_next_round(self, round_data: Dict):
        if round_data["round"] <= self.simulator.settings.rounds:
            is_sync: bool = self.simulator.settings.synchronous
            if not is_sync:
                start_round_event = Event(self.simulator.current_time, self.index, START_ROUND, data=round_data)
                self.simulator.schedule(start_round_event)
            else:
                # We operate in synchronous mode, so the start of the next round is initiated by the simulator.
                self.simulator.client_ready_for_round(self.index, round_data["round"], round_data)

    def is_training(self) -> bool:
        return any([r.is_training for r in self.round_info.values()])

    def finish_train(self, event: Event):
        """
        We finished training. Send the model to the next node in the sample.
        """
        cur_round: int = event.data["round"]
        if cur_round not in self.round_info:
            raise RuntimeError("Peer %d does not know about round %d after training finished!" %
                               (self.index, cur_round))

        round_info: Round = self.round_info[cur_round]
        round_info.model = event.data["model"]
        round_info.is_training = False
        round_info.train_done = True
        current_sample: List[int] = SampleManager.get_sample(cur_round, len(self.simulator.clients),
                                                             self.simulator.sample_size)
        index_in_current_sample: int = current_sample.index(self.index)
        nb_peer = current_sample[(index_in_current_sample + 1) % len(current_sample)]
        self.client_log("Peer %d finished model training in round %d and sends model to peer %d" %
                        (self.index, cur_round, nb_peer))
        self.send_model(nb_peer, round_info.model, metadata={"type": "transfer_in_sample", "round": cur_round})

        # Check if we already received a model from a neighbour - if so, aggregate and proceed.
        # TODO assume a single incoming model
        if round_info.incoming_models:
            round_info.model = self.aggregate_models(
                list(round_info.incoming_models.values()) + [round_info.model], round_info.round_nr)
            self.send_aggregated_model_to_next_sample(round_info, current_sample)

        # We're now done training. Should we start training in another round?
        for active_round_nr in self.round_info.keys():
            round_info: Round = self.round_info[active_round_nr]
            if not round_info.train_done and not round_info.is_training and round_info.model:
                self.schedule_train(round_info)
                break

    def send_aggregated_model_to_next_sample(self, round_info: Round, current_sample: List[int]):
        next_round_nr: int = round_info.round_nr + 1
        index_in_current_sample: int = current_sample.index(self.index)
        next_sample: List[int] = SampleManager.get_sample(next_round_nr, len(self.simulator.clients),
                                                          self.simulator.sample_size)
        next_peer: int = next_sample[index_in_current_sample]
        self.client_log("Peer %d sending aggregated model to peer %d in next sample (round %d)" %
                        (self.index, next_peer, next_round_nr))
        self.send_model(next_peer, round_info.model, metadata={"type": "transfer_next_sample", "round": next_round_nr})
        self.round_info.pop(round_info.round_nr)

    def on_incoming_model(self, event: Event):
        if event.data["metadata"]["type"] == "transfer_in_sample":
            self.handle_incoming_model_in_sample(event)
        elif event.data["metadata"]["type"] == "transfer_next_sample":
            self.handle_incoming_model_next_sample(event)

    def handle_incoming_model_in_sample(self, event: Event):
        model_round = event.data["metadata"]["round"]
        self.client_log("Peer %d received in-sample model from %d in round %d" %
                        (self.index, event.data["from"], model_round))
        current_sample = SampleManager.get_sample(model_round, len(self.simulator.clients), self.simulator.sample_size)
        if self.index not in current_sample:
            raise RuntimeError("Client %d not in sample of round %d!" % (self.index, model_round))

        if model_round not in self.round_info:
            # We were not activated yet by the previous sample - start a new round
            round_data: Dict = {"round": model_round, "model": None,
                                "incoming_models": {event.data["from"]: event.data["model"]}}
            self.schedule_next_round(round_data)
        else:
            # We are currently working on this round.
            round_info: Round = self.round_info[model_round]
            if round_info.train_done:
                # We are done with training in this round - aggregate and send the model to the next sample.
                round_info.incoming_models[event.data["from"]] = event.data["model"]
                round_info.model = self.aggregate_models(
                    list(round_info.incoming_models.values()) + [round_info.model], round_info.round_nr)
                self.send_aggregated_model_to_next_sample(round_info, current_sample)
            else:
                round_info.incoming_models[event.data["from"]] = event.data["model"]

    def handle_incoming_model_next_sample(self, event: Event):
        model_round = event.data["metadata"]["round"]
        self.client_log("Peer %d received next-sample model from %d for round %d" %
                        (self.index, event.data["from"], model_round))
        current_sample = SampleManager.get_sample(model_round, len(self.simulator.clients), self.simulator.sample_size)
        if self.index not in current_sample:
            raise RuntimeError("Client %d not in sample of round %d!" % (self.index, model_round))

        if model_round in self.round_info:
            # We are aware of this round. Store the incoming model and see if we can start training.
            round_info: Round = self.round_info[model_round]
            round_info.model = event.data["model"]
            if not self.is_training():
                self.schedule_train(round_info)
        else:
            # Start a new round.
            self.schedule_next_round({"round": model_round, "model": event.data["model"]})
