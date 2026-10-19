from typing import Dict
from collections import defaultdict
import copy
from random import Random
from typing import List, Optional, Set, Tuple

from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.conflux import NodeMembershipChange
from dasklearn.simulation.conflux.client_manager import ClientManager
from dasklearn.simulation.conflux.round import Round
from dasklearn.simulation.conflux.sample_manager import SampleManager
from dasklearn.simulation.events import *
from dasklearn.simulation.bandwidth_scheduler import BWScheduler
from dasklearn.tasks.task import Task
from dasklearn.tasks.references import TaskRef
from dasklearn.util import time_to_sec


class ConfluxClient(AsynchronousClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.round_info: Dict[int, Round] = {}
        self.random = Random(simulator.settings.seed + index * 1234)
        self.advertise_index: int = 1
        self.client_manager: ClientManager = ClientManager(self.index, 100000)
        self.train_sample_estimate: int = 0
        self.last_round_completed: int = 0
        self.round_durations: Dict[int, float] = {}

        self.contributions_in_reconstructed_models: List[Tuple[int, int, float, int, int]] = []  # Format: (round, peer_id, coverage, network_speed, compute_speed)
        self.contributions_per_model: Dict[int, int] = {}

        self.available_chunks: Dict[int, List[Tuple[int, Set[str]]]] = defaultdict(list)  # Keep track of the chunks we have available, indexed by round

    def init_client(self, _: Event):
        active_clients: List[int] = self.client_manager.get_active_clients()
        sample: List[int] = SampleManager.get_sample(1, active_clients, self.simulator.settings.sample_size, seed=self.simulator.settings.seed)

        if self.index in sample:
            round = Round(1)
            start_round_event = Event(self.simulator.current_time, self.index, START_ROUND, data=round)
            self.simulator.schedule(start_round_event)

    def come_online(self, event):
        super().come_online(event)
        self.advertise_membership(NodeMembershipChange.JOIN)

    def go_offline(self, event):
        super().go_offline(event)
        self.advertise_membership(NodeMembershipChange.LEAVE)

    def get_round_estimate(self) -> int:
        """
        Get the highest round estimation, based on our local estimations and the estimations in the population view.
        """
        max_round_in_population_view = self.client_manager.get_highest_round_in_population_view()
        return max(self.train_sample_estimate, max_round_in_population_view)

    def advertise_membership(self, change: NodeMembershipChange):
        """
        Advertise your (new) membership to random (online) clients.
        """
        advertise_index: int = self.advertise_index
        self.advertise_index += 1

        self.client_log("Client %d advertising its membership change %s to active clients (idx %d)" %
                         (self.index, change, advertise_index))

        active_clients: List[int] = self.client_manager.get_active_clients()
        if self.index in active_clients:
            active_clients.remove(self.index)

        if change == NodeMembershipChange.LEAVE:
            # When going offline, we can simply query our current view of the network and select the last nodes offline
            random_clients = self.random.sample(active_clients, min(self.simulator.settings.sample_size * 10, len(active_clients)))
        else:
            # When coming online we probably don't have a fresh view on the network so we need to determine online nodes
            # We could use a bootstrap server for this in practical settings
            random_clients = self.context.oracle('bootstrap_peers')
            random_clients = self.random.sample(random_clients, min(self.simulator.settings.sample_size * 10, len(random_clients)))

        for client_index in random_clients:
            self.send_control(client_index, "status", {"round": self.get_round_estimate(), "index": advertise_index, "change": change.value})

        # Update your own population view
        info = self.client_manager.last_active[self.index]
        self.client_manager.last_active[self.index] = (info[0], (advertise_index, change))

    def start_round(self, event: Event):
        round_info: Round = event.data
        self.round_info[round_info.round_nr] = round_info
        round_nr: int = round_info.round_nr
        if self.simulator.settings.rounds > 0 and round_nr > self.simulator.settings.rounds:
            return
        if round_info.is_training or round_info.train_done:
            return
        if round_nr < 1:
            raise RuntimeError("Round number %d invalid!" % round_nr)

        # Should we test the model?
        if (round_nr > 1 and self.simulator.settings.test_method == "individual"
                and self.simulator.settings.test_interval > 0
                and (round_nr - 1) % self.simulator.settings.test_interval == 0):
            test_task_name = "test_%d_%d" % (self.index, round_nr)
            task = self.create_compute_task("test", {"model": round_info.model, "round": round_nr - 1, "time": self.simulator.current_time, "peer": self.index}, name=test_task_name)
            round_info.model = TaskRef(test_task_name, 0)

        self.logger.info(f"[t=%.3f] Client %d starts training in round %d", time_to_sec(self.simulator.current_time), self.index, round_nr)

        # 1. Train the model
        self.schedule_train(round_info)

    def finish_train(self, event: Event):
        """
        We finished training.
        """
        round_nr: int = event.data["round"]
        round_info: Round = self.round_info[round_nr]
        round_info.is_training = False
        round_info.train_done = True
        round_info.model = event.data["model"]

        self.client_log(f"Client {self.index} finished training in round {round_nr}")

        self.chunk_model(round_info)
        for pending in sorted(self.round_info.values(), key=lambda info: info.round_nr):
            if not pending.received_enough_chunks and pending.inventories:
                self.pull_chunks_for_round(pending)

    def advertise_new_inventory(self, participants: List[int], round_nr: int, chunks: List[Tuple[int, Set[str]]]):
        for participant in participants:
            self.send_control(participant, "has_chunks", {"round": round_nr, "chunks": chunks})

    def send_population_view(self, receiver: int) -> None:
        self.send_control(receiver, "population_view", self.client_manager.last_active)

    def chunk_model(self, round_info: Round):
        """
        Chunk your current local model
        """

        # Add compute tasks for the chunks
        task_name = Task.generate_name("chunk")
        task = self.create_compute_task("chunk", {
            "model": round_info.model, "round": round_info.round_nr,
            "time": self.simulator.current_time, "peer": self.index,
            "n": self.simulator.settings.chunks_in_sample,
        }, name=task_name)

        all_chunks: List[Tuple[int, Set[str]]] = [(i, frozenset({task_name})) for i in range(self.simulator.settings.chunks_in_sample)]
        next_round_nr: int = round_info.round_nr + 1

        for chunk in all_chunks:
            self.available_chunks[next_round_nr].append(chunk)

        # Let the nodes in the next sample know about the availability of these chunks
        participants_next_sample: List[int] = SampleManager.get_sample(next_round_nr, self.client_manager.get_active_clients(), self.simulator.settings.sample_size, seed=self.simulator.settings.seed)
        if self.index in participants_next_sample:
            # We are in the next sample. We can already inject the chunks for this next round
            if next_round_nr not in self.round_info:
                self.round_info[next_round_nr] = Round(next_round_nr)
                self.round_info[next_round_nr].init_received_chunks(self.simulator.settings)
                self.round_info[next_round_nr].sample = participants_next_sample

            next_round_info: Round = self.round_info[next_round_nr]
            for chunk_idx in range(self.simulator.settings.chunks_in_sample):
                next_round_info.received_chunks[chunk_idx].add(task_name)
            self.maybe_complete_round(next_round_info)

        self.advertise_new_inventory(participants_next_sample, next_round_nr, all_chunks)

        # Membership is already shared at initialization. Synchronize views only
        # when membership can change; tiny-model comparisons must not charge a
        # static-membership baseline for redundant full-population exchanges.
        if self.simulator.settings.churn != "none":
            for participant in participants_next_sample:
                if participant != self.index:
                    self.send_population_view(participant)

        if round_info.round_nr > 1:
            self.round_durations[round_info.round_nr] = time_to_sec(self.simulator.current_time - round_info.pull_start)
        self.last_round_completed = max(self.last_round_completed, round_info.round_nr)
        self.simulator.set_finished(round_info.round_nr, round_info.model)

    def start_outgoing_chunk_transfer(self, round_nr: int, to: int, chunk: Tuple[int, Set[str]]) -> None:
        metadata = {"chunk": chunk, "round": round_nr}
        if round_nr in self.round_info and self.simulator.settings.churn != "none":
            round_info = self.round_info[round_nr]
            if to not in round_info.has_sent_view:
                metadata["population_view"] = copy.deepcopy(self.client_manager.last_active)
                round_info.has_sent_view.add(to)
        self.send(to, None,
                  size_bytes=self.chunk_transfer_size(chunk[0], self.simulator.settings.chunks_in_sample),
                  metadata=metadata, immediate=True)

    def has_free_pull_slot(self):
        return sum(len(info.is_pulling) for info in self.round_info.values()
                   if not info.received_enough_chunks) < 5

    def pull_chunks_for_round(self, round_info: Round):
        if round_info.pull_start is None:
            round_info.pull_start = self.simulator.current_time

        if round_info.received_enough_chunks or not self.has_free_pull_slot():
            # No space for another pull
            return
        # Future-round pulls must not occupy every slot while an earlier round
        # still needs contributions from this same client to make progress.
        if any(info.round_nr < round_info.round_nr and not info.train_done
               for info in self.round_info.values()):
            return

        # Outdated round
        if round_info.round_nr < self.last_round_completed:
            return

        if self.random.random() < 0.5:
            # Pull chunks for underrepresented chunk indices
            self.pull_chunks_for_underrepresented_chunk_idxs(round_info)
        else:
            self.pull_rare_chunks(round_info)

    @staticmethod
    def contribution_received_or_pending(round_info, chunk_idx, model_name):
        return (round_info.has_received_chunk(chunk_idx, model_name) or
                any(idx == chunk_idx and model_name in names for idx, names in round_info.is_pulling))

    @staticmethod
    def coalesce_disjoint_chunks(chunks):
        # Averages of overlapping contributor sets cannot be deduplicated from
        # provenance alone. Only disjoint partial sums can be merged exactly.
        names = set()
        for _, contributors in sorted(chunks, key=lambda chunk: len(chunk[1]), reverse=True):
            if names.isdisjoint(contributors):
                names.update(contributors)
        return names

    def pull_chunks_for_underrepresented_chunk_idxs(self, round_info: Round):
        # Determine chunk indices that are underrepresented
        chunk_idxs = list(range(self.simulator.settings.chunks_in_sample))
        self.random.shuffle(chunk_idxs)  # Shuffle before sorting to ensure randomness within groups
        chunk_idxs.sort(key=lambda x: len(round_info.received_chunks[x]))

        # Now, determine per chunk index the chunk owners that can offer us the chunk we don't have yet
        # Essentially, we make a list of owners and what they can offer us
        chunks_per_sender: Dict[int, Dict[int, List[Tuple[str, int]]]] = defaultdict(list)
        for idx in chunk_idxs:
            chunks_per_sender[idx] = defaultdict(list)

        for chunk, owners in round_info.inventories.items():
            chunk_idx: int = chunk[0]
            for owner in owners:
                if owner not in chunks_per_sender[chunk_idx]:
                    chunks_per_sender[chunk_idx][owner] = []

                # Do we have all components of this chunk already?
                need_chunk: bool = True
                for model_name in chunk[1]:
                    if self.contribution_received_or_pending(round_info, chunk_idx, model_name):
                        need_chunk = False
                        break

                if need_chunk:
                    chunks_per_sender[chunk_idx][owner].append(chunk)

        # Now, we iterate over each chunk index and start to contact the owners with the most chunks we don't have yet
        # until we have filled up our incoming slots
        for chunk_idx in chunk_idxs:
            owners = chunks_per_sender[chunk_idx]
            count_per_owner: Dict[int, int] = {}
            for owner, chunks in owners.items():
                owner_count: int = 0
                for chunk in chunks:
                    owner_count += len(chunk[1])
                count_per_owner[owner] = owner_count

            # Get the owners with the most chunks we don't have yet
            sorted_owners = sorted(count_per_owner.items(), key=lambda x: x[1], reverse=True)
            for owner, _ in sorted_owners:
                # Coalesce all the chunks from this owner into a single one
                model_names = self.coalesce_disjoint_chunks(owners[owner])

                chunk_to_pull: Tuple[int, Set[str]] = (chunk_idx, frozenset(model_names))
                if len(chunk_to_pull[1]) == 0:
                    continue  # This owner has nothing to offer us

                if chunk_to_pull not in round_info.is_pulling:
                    round_info.is_pulling.add(chunk_to_pull)
                    self.send_control(owner, "pull_chunk", {"round": round_info.round_nr, "chunk": chunk_to_pull})
                    break  # move to the next chunk index

            if not self.has_free_pull_slot():
                break

    def pull_rare_chunks(self, round_info: Round):
        # For each chunk, determine how rare it is
        chunk_rarity: Dict[Tuple[int, Set[str]], int] = {}
        for chunk, owners in round_info.inventories.items():
            chunk_rarity[chunk] = len(owners)

        # Sort the chunks by rarity
        sorted_chunks = list(chunk_rarity.items())
        self.random.shuffle(sorted_chunks)
        sorted_chunks.sort(key=lambda x: x[1])

        # Iterate over the chunks and find ones that we can pull
        for chunk_to_pull, _ in sorted_chunks:
            chunk_idx: int = chunk_to_pull[0]

            # Do we have all components of this chunk already?
            need_chunk: bool = True
            for model_name in chunk_to_pull[1]:
                if self.contribution_received_or_pending(round_info, chunk_idx, model_name):
                    need_chunk = False
                    break

            if not need_chunk:
                continue

            for owner in round_info.inventories[chunk_to_pull]:
                if chunk_to_pull not in round_info.is_pulling:
                    round_info.is_pulling.add(chunk_to_pull)
                    self.send_control(owner, "pull_chunk", {"round": round_info.round_nr, "chunk": chunk_to_pull})
                    break  # move to the next chunk

            if not self.has_free_pull_slot():
                break

    def on_incoming_model(self, event: Event):
        """
        We received a model chunk.
        """
        if not self.online:
            return

        metadata: Dict = event.data["metadata"]
        from_client: int = event.data["from"]
        self.client_manager.update_client_activity(from_client, max(metadata["round"], self.get_round_estimate()))
        self.received_model_chunk(from_client, metadata["round"], metadata["chunk"], metadata.get("population_view", None))

    def received_model_chunk(self, from_client: int, round_nr: int, chunk: Tuple[int, Set[int]], population_view: Optional[Dict] = None) -> None:
        if population_view:
            self.client_manager.merge_population_views(population_view)

        round_info: Round = self.round_info[round_nr]
        if round_info.received_enough_chunks:
            round_info.is_pulling.discard(chunk)
            return
        self.client_log("Client %d received chunk %s from %d for round %d" % (self.index, chunk, from_client, round_nr))

        chunk_idx: int = chunk[0]
        for chunk_model_name in chunk[1]:
            round_info.received_chunks[chunk_idx].add(chunk_model_name)

        round_info.is_pulling.discard(chunk)

        # Update our inventory
        self.available_chunks[round_nr].append(chunk)

        if not round_info.sample:
            round_info.sample = SampleManager.get_sample(round_nr, self.client_manager.get_active_clients(), self.simulator.settings.sample_size, seed=self.simulator.settings.seed)

        self.advertise_new_inventory(round_info.sample, round_nr, [chunk])

        self.maybe_complete_round(round_info)

    def maybe_complete_round(self, round_info):
        round_nr = round_info.round_nr
        if round_info.received_enough_chunks:
            return
        if round_info.has_received_enough_chunks():
            self.client_log(f"Client {self.index} received enough chunks in round {round_nr}")
            round_info.received_enough_chunks = True

            # Kill any incoming transfer related to this round
            for transfer in list(self.bw_scheduler.incoming_transfers.values()) + list(self.bw_scheduler.incoming_requests.values()):
                if transfer.metadata.get("round") == round_nr:
                    self.bw_scheduler.kill_transfer(transfer)

            # Collect the received chunks per index
            count_per_model_name: Dict[str, int] = defaultdict(int)
            received_chunks: List[Set[str]] = [[] for _ in range(self.simulator.settings.chunks_in_sample)]
            for chunk_idx, chunk_models in enumerate(round_info.received_chunks):
                for model_name in chunk_models:
                    count_per_model_name[model_name] += 1
                    received_chunks[chunk_idx].append(TaskRef(model_name, chunk_idx))

            self.contributions_per_model[round_nr] = len(count_per_model_name)

            # Write statistics on the received chunks
            for model_name, count in count_per_model_name.items():
                # Get the peer behind this chunk
                corresponding_task: Task = self.simulator.workflow_dag.tasks[model_name]
                client_id = corresponding_task.data["peer"]
                coverage = count / self.simulator.settings.chunks_in_sample
                network_speed: int = self.simulator.clients[client_id].bw_scheduler.bw_limit
                compute_speed: int = self.simulator.clients[client_id].simulated_speed
                self.contributions_in_reconstructed_models.append((round_nr, client_id, coverage, network_speed, compute_speed))

            # Reconstruct new model
            task_name = Task.generate_name("reconstruct_from_chunks")
            task = self.create_compute_task("reconstruct_from_chunks", {
                "chunks": received_chunks, "round": round_nr,
                "time": self.simulator.current_time, "peer": self.index
            }, name=task_name)
            round_info.model = TaskRef(task_name, 0)
            self.publish_model(round_nr - 1, round_info.model)

            start_round_event = Event(self.simulator.current_time, self.index, START_ROUND, data=round_info)
            self.simulator.schedule(start_round_event)
        else:
            self.pull_chunks_for_round(round_info)

    def on_membership_advertisement(self, other_client: int, status: int, round: int, index: int):
        """
        We received a membership advertisement from a client.
        """
        if not self.online:
            return

        change: NodeMembershipChange = NodeMembershipChange(status)
        latest_round = self.get_round_estimate()
        if change == NodeMembershipChange.JOIN:
            self.logger.debug("Client %d updating membership of client %d to: JOIN (idx %d)", self.index, other_client, index)
            self.client_manager.last_active[other_client] = (max(round, latest_round), (index, NodeMembershipChange.JOIN))
        else:
            self.logger.debug("Client %d updating membership of client %d to: LEAVE (idx %d)", self.index, other_client, index)
            self.client_manager.last_active[other_client] = (max(round, latest_round), (index, NodeMembershipChange.LEAVE))

    def on_message(self, event: Event):
        if not self.online:
            return
        if event.data["type"] == "pull_chunk":
            message = event.data["message"]
            self.start_outgoing_chunk_transfer(message["round"], event.data["from"], message["chunk"])
        elif event.data["type"] == "status":
            self.on_membership_advertisement(event.data["from"], event.data["message"]["change"], event.data["message"]["round"], event.data["message"]["index"])
        elif event.data["type"] == "has_chunks":
            # A client in some previous sample has chunks available - start pulling if we're not doing so already

            # Are we supposed to pull chunks for this round?
            round_nr = event.data["message"]["round"]

            if round_nr not in self.round_info:
                self.round_info[round_nr] = Round(round_nr)
                self.round_info[round_nr].init_received_chunks(self.simulator.settings)

            if not self.round_info[round_nr].sample:
                self.round_info[round_nr].sample = SampleManager.get_sample(round_nr, self.client_manager.get_active_clients(), self.simulator.settings.sample_size, seed=self.simulator.settings.seed)

            if self.index not in self.round_info[round_nr].sample:
                return

            # Update inventories
            for chunk in event.data["message"]["chunks"]:
                if chunk not in self.round_info[round_nr].inventories:
                    self.round_info[round_nr].inventories[chunk] = []
                if event.data["from"] not in self.round_info[round_nr].inventories[chunk]:
                    self.round_info[round_nr].inventories[chunk].append(event.data["from"])

            self.pull_chunks_for_round(self.round_info[round_nr])
        elif event.data["type"] == "population_view":
            self.client_manager.merge_population_views(event.data["message"])
        else:
            raise ValueError("Unknown message type: %s" % event.data["type"])

    def schedule_train(self, round_info: Round):
        round_info.is_training = True
        start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN, data={
            "model": round_info.model, "round": round_info.round_nr})
        self.simulator.schedule(start_train_event)
