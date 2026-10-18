"""Causal, token-preserving SGD -> same-round gossip -> model handoff."""
from collections import defaultdict
from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, START_ROUND, START_TRAIN, INCOMING_MODEL
from dasklearn.tasks.task import Task


class TeleportationClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(lambda: {"incoming": {}, "started": False,
                                          "trained": False, "mixed": False})

    def init_client(self, event):
        if self.index in self.simulator.sample(1):
            self.simulator.schedule(Event(event.time, self.index, START_ROUND,
                                          {"round": 1, "model": (None, 0)}))

    def start_round(self, event):
        state = self.rounds[event.data["round"]]
        if state["started"]:
            return
        state["started"] = True
        self.simulator.schedule(Event(self.simulator.current_time, self.index, START_TRAIN, event.data))

    def transmit(self, peer, model, round_nr, phase):
        metadata = {"round": round_nr, "phase": phase}
        if peer == self.index:
            self.simulator.schedule(Event(self.simulator.current_time, peer, INCOMING_MODEL,
                {"from": self.index, "to": peer, "model": model, "metadata": metadata}))
        else:
            self.send_model(peer, model, metadata)

    def start_transfer(self, event):
        self.bw_scheduler.add_transfer(self.simulator.clients[event.data["to"]].bw_scheduler,
            self.data_transfer_size(self.simulator.model_size, event.data["metadata"]),
            event.data["model"], event.data["metadata"])

    def finish_train(self, event):
        round_nr = event.data["round"]
        state = self.rounds[round_nr]
        if state["trained"]:
            return
        state["trained"] = True
        state["model"] = event.data["model"]
        self.compute_time += event.data["train_time"]
        sample = self.simulator.sample(round_nr)
        for token in self.simulator.neighbors(sample.index(self.index)):
            self.transmit(sample[token], state["model"], round_nr, "mix")
        self.try_mix(round_nr)

    def on_incoming_model(self, event):
        round_nr = event.data["metadata"]["round"]
        if event.data["metadata"]["phase"] == "handoff":
            self.simulator.schedule(Event(self.simulator.current_time, self.index, START_ROUND,
                {"round": round_nr, "model": event.data["model"]}))
        else:
            self.rounds[round_nr]["incoming"][event.data["from"]] = event.data["model"]
            self.try_mix(round_nr)

    def try_mix(self, round_nr):
        state = self.rounds[round_nr]
        sample = self.simulator.sample(round_nr)
        token = sample.index(self.index)
        neighbors = [sample[i] for i in self.simulator.neighbors(token)]
        if state["mixed"] or not state["trained"] or not set(neighbors).issubset(state["incoming"]):
            return
        state["mixed"] = True
        models = [state["model"]] + [state["incoming"][peer] for peer in neighbors]
        # Symmetric regular graph plus self-loop: W is doubly stochastic.
        task = Task(Task.generate_name("teleport_mix"), "aggregate", {
            "models": models, "round": round_nr, "peer": self.index,
            "time": self.simulator.current_time})
        self.add_compute_task(task)
        model = (task.name, 0)
        self.publish_model(round_nr, model)
        if self.simulator.settings.test_method == "individual" and self.simulator.settings.test_interval > 0 and round_nr % self.simulator.settings.test_interval == 0:
            self.add_compute_task(Task(Task.generate_name("test"), "test", {
                "model": model, "round": round_nr, "peer": self.index, "time": self.simulator.current_time}))
        if self.simulator.settings.rounds <= 0 or round_nr < self.simulator.settings.rounds:
            self.transmit(self.simulator.sample(round_nr + 1)[token], model, round_nr + 1, "handoff")
