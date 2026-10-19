"""Group-local butterfly reduce-scatter/all-gather, with explicit readiness."""
from collections import defaultdict

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, START_ROUND, START_TRAIN, START_TRANSFER, INCOMING_MODEL
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.tasks.task import Task


class MoshpitClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.started = set()
        self.phases = defaultdict(lambda: {"ready": set(), "planned": False, "uploads": {},
                                          "received": {}, "reduced": False, "done": False})

    def init_client(self, event):
        self.simulator.schedule(Event(event.time, self.index, START_ROUND, {"round": 1, "model": (None, 0)}))

    def start_round(self, event):
        round_nr = event.data["round"]
        if round_nr in self.started:
            return
        self.started.add(round_nr)
        self.simulator.schedule(Event(self.simulator.current_time, self.index, START_TRAIN, event.data))

    def finish_train(self, event):
        self.compute_time += event.data["train_time"]
        self.begin_phase(event.data["round"], 0, event.data["model"])

    def begin_phase(self, round_nr, phase, model):
        state = self.phases[round_nr, phase]
        if "chunked" in state:
            return
        members, weights = self.simulator.layout(self.index, round_nr, phase)
        task = Task(Task.generate_name("chunk"), "chunk", {"model": model, "n": len(members),
            "weights": weights, "round": round_nr, "phase": phase, "peer": self.index,
            "time": self.simulator.current_time})
        self.add_compute_task(task)
        state["chunked"] = task.name
        self.send_message_to_client(members[0], "ready", {"round": round_nr, "phase": phase})

    def on_message(self, event):
        message = event.data["message"]
        round_nr, phase = message["round"], message["phase"]
        state = self.phases[round_nr, phase]
        members, _ = self.simulator.layout(self.index, round_nr, phase)
        if event.data["type"] == "ready":
            if self.index != members[0] or event.data["from"] not in members:
                raise ValueError("Invalid group readiness")
            state["ready"].add(event.data["from"])
            if len(state["ready"]) == len(members) and not state["planned"]:
                state["planned"] = True
                for peer in members:
                    self.send_message_to_client(peer, "plan", message)
        elif event.data["type"] == "plan":
            if event.data["from"] != members[0] or state.get("uploaded"):
                return
            state["uploaded"] = True
            for chunk, owner in enumerate(members):
                self.send_part(owner, (state["chunked"], chunk), round_nr, phase, chunk, "upload")
        else:
            raise ValueError("Unknown Moshpit control message")

    def send_part(self, peer, model, round_nr, phase, chunk, kind):
        data = {"from": self.index, "to": peer, "model": model,
                "metadata": {"round": round_nr, "phase": phase, "chunk": chunk, "kind": kind}}
        if peer == self.index:
            self.simulator.schedule(Event(self.simulator.current_time, peer, INCOMING_MODEL, data))
        else:
            self.simulator.schedule(Event(self.simulator.current_time, self.index, START_TRANSFER, data))

    def start_transfer(self, event):
        meta = event.data["metadata"]
        _, weights = self.simulator.layout(self.index, meta["round"], meta["phase"])
        size = ChunkManager.partition_sizes(self.simulator.model_size, weights)[meta["chunk"]]
        self.bw_scheduler.add_transfer(self.simulator.clients[event.data["to"]].bw_scheduler,
            self.data_transfer_size(size, meta), event.data["model"], meta)

    def on_incoming_model(self, event):
        meta = event.data["metadata"]
        round_nr, phase, chunk = meta["round"], meta["phase"], meta["chunk"]
        state = self.phases[round_nr, phase]
        members, _ = self.simulator.layout(self.index, round_nr, phase)
        if meta["kind"] == "upload":
            state["uploads"][event.data["from"]] = event.data["model"]
            if not state["reduced"] and len(state["uploads"]) == len(members):
                state["reduced"] = True
                task = Task(Task.generate_name("aggregate_chunk"), "aggregate_chunk", {
                    "chunks": [state["uploads"][peer] for peer in members], "round": round_nr,
                    "phase": phase, "peer": self.index, "time": self.simulator.current_time})
                self.add_compute_task(task)
                for peer in members:
                    self.send_part(peer, (task.name, 0), round_nr, phase, chunk, "gather")
        else:
            state["received"][chunk] = event.data["model"]
            if len(state["received"]) == len(members) and not state["done"]:
                state["done"] = True
                task = Task(Task.generate_name("reconstruct_from_chunks"), "reconstruct_from_chunks", {
                    "chunks": [[state["received"][i]] for i in range(len(members))],
                    "round": round_nr, "phase": phase, "peer": self.index,
                    "time": self.simulator.current_time})
                self.add_compute_task(task)
                model = (task.name, 0)
                # Only publish the configured averaging block's completed output.
                if phase + 1 < self.simulator.settings.averaging_steps:
                    self.begin_phase(round_nr, phase + 1, model)
                    return
                self.publish_model(round_nr, model)
                if self.simulator.settings.test_method == "individual" and self.simulator.settings.test_interval > 0 and round_nr % self.simulator.settings.test_interval == 0:
                    self.add_compute_task(Task(Task.generate_name("test"), "test", {
                        "model": model, "round": round_nr, "peer": self.index, "time": self.simulator.current_time}))
                if self.simulator.settings.rounds <= 0 or round_nr < self.simulator.settings.rounds:
                    self.simulator.schedule(Event(self.simulator.current_time, self.index, START_ROUND,
                        {"round": round_nr + 1, "model": model}))
