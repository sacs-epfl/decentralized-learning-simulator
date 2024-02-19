from collections import defaultdict
from typing import List, Optional, Tuple

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import *
from dasklearn.simulation.subset.sample_manager import SampleManager


class SubsetDLClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.train_done: bool = False
        self.round_done: bool = False
        self.incoming_models: Dict[int, Dict[int, str]] = defaultdict(lambda: {})
        self.early_incoming_next_sample_models: List[Tuple[int, str]] = []

    def init_client(self, event: Event):
        start_round_event = Event(event.time, self.index, START_ROUND)
        self.simulator.schedule(start_round_event)
        self.round = 1

    def start_round(self, _: Event):
        self.train_done = False
        self.round_done = False
        sample: List[int] = SampleManager.get_sample(self.round, len(self.simulator.clients), self.simulator.sample_size)
        if self.index in sample:
            self.client_log("Peer %d starting round %d" % (self.index, self.round))
            start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN)
            self.simulator.schedule(start_train_event)
        else:
            self.round_done = True

    def finish_train(self, event: Event):
        """
        We finished training. Send the model to the next node in the sample.
        """
        self.train_done = True
        self.own_model = event.data["model"]
        current_sample: List[int] = SampleManager.get_sample(self.round, len(self.simulator.clients), self.simulator.sample_size)
        index_in_current_sample: int = current_sample.index(self.index)
        nb_peer = current_sample[(index_in_current_sample + 1) % len(current_sample)]
        self.client_log("Peer %d finished model training in round %d and sends model to peer %d" % (self.index, self.round, nb_peer))
        self.send_model(nb_peer, self.own_model, metadata={"type": "transfer_in_sample", "round": self.round})

        # Check if we already received a model from a neighbour - if so, aggregate and proceed.
        if self.incoming_models[self.round]:
            self.own_model = self.aggregate_models(list(self.incoming_models[self.round].values()) + [self.own_model])
            self.send_aggregated_model_to_next_sample(current_sample)

    def on_incoming_model(self, event: Event):
        if event.data["metadata"]["type"] == "transfer_in_sample":
            self.handle_incoming_model_in_sample(event)
        elif event.data["metadata"]["type"] == "transfer_next_sample":
            self.handle_incoming_model_next_sample(event)

    def send_aggregated_model_to_next_sample(self, current_sample: List[int]):
        index_in_current_sample: int = current_sample.index(self.index)
        next_sample: List[int] = SampleManager.get_sample(self.round + 1, len(self.simulator.clients), self.simulator.sample_size)
        next_peer: int = next_sample[index_in_current_sample]
        self.client_log("Peer %d sending aggregated model to peer %d in next sample (round %d)" %
                        (self.index, next_peer, self.round + 1))
        self.send_model(next_peer, self.own_model, metadata={"type": "transfer_next_sample", "round": self.round + 1})
        del self.incoming_models[self.round]
        self.round_done = True

        # Did we receive an early next-sample model? If so, start the next round already
        if self.early_incoming_next_sample_models:
            model_round, model = self.early_incoming_next_sample_models[0]
            self.client_log("Peer %d processing early next-sample model for round %d" % (self.index, model_round))
            self.round = model_round
            self.own_model = model
            self.early_incoming_next_sample_models.pop(0)
            self.start_round(None)

    def handle_incoming_model_in_sample(self, event: Event):
        model_round = event.data["metadata"]["round"]
        self.client_log("Peer %d received in-sample model from %d in round %d" %
                        (self.index, event.data["from"], model_round))
        current_sample = SampleManager.get_sample(model_round, len(self.simulator.clients), self.simulator.sample_size)
        if self.index not in current_sample:
            raise RuntimeError("Client %d not in sample of round %d!" % (self.index, model_round))

        if self.round < model_round:
            # We were not activated yet by the previous sample - simply save this model for later.
            self.incoming_models[model_round][event.data["from"]] = event.data["model"]
        elif self.round == model_round and self.train_done:
            # We are done with training in this round - aggregate and send the model to the next sample.
            self.incoming_models[self.round][event.data["from"]] = event.data["model"]
            self.own_model = self.aggregate_models(list(self.incoming_models[self.round].values()) + [self.own_model])
            self.send_aggregated_model_to_next_sample(current_sample)
        elif self.round == model_round and not self.train_done:
            # We are not done training yet for this round - save this model for later.
            self.incoming_models[model_round][event.data["from"]] = event.data["model"]

    def handle_incoming_model_next_sample(self, event: Event):
        model_round = event.data["metadata"]["round"]
        self.client_log("Peer %d received next-sample model from %d for round %d" %
                        (self.index, event.data["from"], model_round))
        current_sample = SampleManager.get_sample(model_round, len(self.simulator.clients) , self.simulator.sample_size)
        if self.index not in current_sample:
            raise RuntimeError("Client %d not in sample of round %d!" % (self.index, model_round))

        # Are we still working on another round? If so, we cannot interrupt everything and start the new round.
        if self.round != model_round and not self.round_done:
            self.early_incoming_next_sample_models.append((model_round, event.data["model"]))
            self.early_incoming_next_sample_models.sort()
        else:
            self.round = model_round
            self.own_model = event.data["model"]
            self.start_round(event)
