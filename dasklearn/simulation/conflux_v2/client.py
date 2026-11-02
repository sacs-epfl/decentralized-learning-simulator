"""Round-local, message-driven reduce and disseminate protocol.

A temporary coordinator freezes the first q ready contributors. It only handles
control messages; independent reducers aggregate disjoint model chunks. The same
contributor set is used for every chunk and every next-sample recipient.
With immediate uploads and full participation, that set is known beforehand;
uploads overlap other contributors' training without coordinator messages.
"""
from collections import defaultdict
from functools import lru_cache
from math import ceil

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.conflux.sample_manager import SampleManager
from dasklearn.simulation.events import Event, START_ROUND, START_TRAIN
from dasklearn.tasks.task import Task
from dasklearn.tasks.references import TaskRef


@lru_cache(maxsize=256)
def _sample(round_nr, peers, sample_size, seed):
    return tuple(SampleManager.get_sample(round_nr, list(peers), sample_size, seed=seed))


@lru_cache(maxsize=256)
def _layout(round_nr, count, limit, capacities):
    """Cache public placement math, with capacities/settings in the cache key."""
    bandwidths = dict(capacities)
    order = SampleManager.get_sample(round_nr, list(bandwidths), len(bandwidths))
    order = order[round_nr % len(order):] + order[:round_nr % len(order)]
    order.sort(key=lambda peer: bandwidths[peer], reverse=True)
    relays = order[:min(limit, len(order))]
    load = dict.fromkeys(relays, 0)
    owners = []
    for chunk in range(count):
        owner = min(relays, key=lambda peer: (load[peer] + 1) / bandwidths[peer])
        owners.append(owner)
        load[owner] += 1
    return order[0], tuple(owners)


class ConfluxV2Client(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(lambda: {
            "ready": {}, "plan": None, "uploads": defaultdict(dict),
            "reduced": set(), "received": {}, "started": False, "trained": False})

    def sample(self, round_nr):
        return list(_sample(round_nr, tuple(self.other_nodes_bws), self.simulator.settings.sample_size,
                            self.simulator.settings.seed))

    def layout(self, round_nr):
        count = self.simulator.settings.chunks_in_sample
        limit = self.simulator.settings.relay_count or count
        leader, owners = _layout(round_nr, count, limit, tuple(self.other_nodes_bws.items()))
        # Callers receive independent containers, including when plans are stored.
        return leader, list(owners)

    def init_client(self, event):
        if self.index in self.sample(1):
            self.simulator.schedule(Event(event.time, self.index, START_ROUND,
                                          {"round": 1, "model": None}))

    def start_round(self, event):
        round_nr = event.data["round"]
        state = self.rounds[round_nr]
        if state["started"]:
            return
        state["started"] = True
        self.simulator.schedule(Event(self.simulator.current_time, self.index, START_TRAIN, event.data))

    def install_known_plan(self, round_nr):
        """Install the same public full sample at senders and reducers.

        Partial-contribution rounds must freeze one common ready set, and keep
        the original coordinated path. Merely uploading first must never pick
        different contributors for different chunks.
        """
        if not (self.simulator.settings.immediate_uploads and
                self.simulator.settings.success_fraction == 1):
            return False
        state = self.rounds[round_nr]
        if not state.get("plan_received"):
            contributors = self.sample(round_nr)
            state["plan"] = contributors
            state["plan_received"] = True
            self.simulator.contributor_sets[round_nr] = contributors
        return True

    def finish_train(self, event):
        round_nr = event.data["round"]
        state = self.rounds[round_nr]
        if state["trained"]:
            return
        state["trained"] = True
        task = self.create_compute_task("chunk", {
            "model": event.data["model"], "n": self.simulator.settings.chunks_in_sample,
            "round": round_nr, "peer": self.index, "time": self.simulator.current_time}, name=Task.generate_name("chunk"))
        state["chunked"] = task.name
        if self.install_known_plan(round_nr):
            _, owners = self.layout(round_nr)
            for chunk_idx, owner in enumerate(owners):
                self.send_chunk(owner, TaskRef(state["chunked"], chunk_idx), round_nr, chunk_idx, "upload")
            return
        leader, _ = self.layout(round_nr)
        self.send_control(leader, "ready", {"round": round_nr})

    def on_message(self, event):
        message = event.data["message"]
        round_nr = message["round"]
        state = self.rounds[round_nr]
        leader, owners = self.layout(round_nr)
        if event.data["type"] == "ready":
            if self.index != leader or event.data["from"] not in self.sample(round_nr):
                return
            state["ready"][event.data["from"]] = True
            q = ceil(self.simulator.settings.sample_size * self.simulator.settings.success_fraction)
            if state["plan"] is None and len(state["ready"]) >= q:
                contributors = sorted(state["ready"])
                state["plan"] = contributors
                self.simulator.contributor_sets[round_nr] = contributors
                plan = {"round": round_nr, "contributors": contributors}
                for peer in sorted(set(contributors) | set(owners)):
                    self.send_control(peer, "plan", plan)
        elif event.data["type"] == "plan":
            if event.data["from"] != leader or state.get("plan_received"):
                return
            state["plan_received"] = True
            state["plan"] = message["contributors"]
            if self.index in state["plan"]:
                for chunk_idx, owner in enumerate(owners):
                    self.send_chunk(owner, TaskRef(state["chunked"], chunk_idx), round_nr, chunk_idx, "upload")
            self.try_reduce(round_nr)
        else:
            raise ValueError("Unknown Conflux v2 message: %s" % event.data["type"])

    def send_chunk(self, recipient, model, round_nr, chunk_idx, phase):
        meta = {"round": round_nr, "chunk": chunk_idx, "phase": phase}
        size = self.chunk_transfer_size(chunk_idx, self.simulator.settings.chunks_in_sample)
        self.send(recipient, model, size_bytes=size, metadata=meta)

    def on_incoming_model(self, event):
        meta = event.data["metadata"]
        round_nr, chunk_idx = meta["round"], meta["chunk"]
        state = self.rounds[round_nr]
        if meta["phase"] == "upload":
            self.install_known_plan(round_nr)
            state["uploads"][chunk_idx][event.data["from"]] = event.data["model"]
            self.try_reduce(round_nr)
        else:
            if chunk_idx in state["received"]:
                return
            state["received"][chunk_idx] = event.data["model"]
            if len(state["received"]) == self.simulator.settings.chunks_in_sample:
                task = self.create_compute_task("reconstruct_from_chunks", {
                    "chunks": [[state["received"][idx]] for idx in range(self.simulator.settings.chunks_in_sample)],
                    "round": round_nr, "peer": self.index, "time": self.simulator.current_time}, name=Task.generate_name("reconstruct_from_chunks"))
                model = TaskRef(task.name, 0)
                self.publish_model(round_nr, model)
                self.simulator.completed_rounds.append((round_nr, self.index, self.simulator.current_time))
                if self.simulator.settings.test_method == "individual" and self.simulator.settings.test_interval > 0 and round_nr % self.simulator.settings.test_interval == 0:
                    self.create_compute_task("test", {
                        "model": model, "round": round_nr, "peer": self.index, "time": self.simulator.current_time}, name=Task.generate_name("test"))
                if self.simulator.settings.rounds <= 0 or round_nr < self.simulator.settings.rounds:
                    self.simulator.schedule(Event(self.simulator.current_time, self.index, START_ROUND,
                                                  {"round": round_nr + 1, "model": model}))

    def try_reduce(self, round_nr):
        state = self.rounds[round_nr]
        if not state.get("plan_received"):
            return
        _, owners = self.layout(round_nr)
        for idx, owner in enumerate(owners):
            if owner != self.index or idx in state["reduced"]:
                continue
            if not set(state["plan"]).issubset(state["uploads"][idx]):
                continue
            state["reduced"].add(idx)
            task = self.create_compute_task("aggregate_chunk", {
                "chunks": [state["uploads"][idx][peer] for peer in state["plan"]],
                "round": round_nr, "peer": self.index, "time": self.simulator.current_time}, name=Task.generate_name("aggregate_chunk"))
            for peer in self.sample(round_nr + 1):
                self.send_chunk(peer, TaskRef(task.name, 0), round_nr, idx, "aggregate")
