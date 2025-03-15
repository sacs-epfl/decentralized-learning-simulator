from collections import defaultdict
import copy
from random import Random
import random
from typing import List, Optional, Set, Tuple

from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.conflux import NodeMembershipChange
from dasklearn.simulation.conflux.client_manager import ClientManager
from dasklearn.simulation.conflux.round import Round
from dasklearn.simulation.conflux.sample_manager import SampleManager
from dasklearn.simulation.events import *
from dasklearn.simulation.slot_bandwidth_scheduler import SlotBWScheduler
from dasklearn.tasks.task import Task


class ConfluxClient(AsynchronousClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.round_info: Dict[int, Round] = {}
        self.random = Random(index + 1234)
        self.advertise_index: int = 1
        self.client_manager: ClientManager = ClientManager(self.index, 100000)
        self.train_sample_estimate: int = 0
        self.last_round_completed: int = 0

        self.available_chunks: Dict[int, List[Tuple[int, Set[str]]]] = defaultdict(list)  # Keep track of the chunks we have available, indexed by round

    def init_client(self, _: Event):
        # Replace the bandwidth scheduler with a slot-based one
        self.bw_scheduler = SlotBWScheduler(self, total_bw=self.bw_scheduler.bw_limit)

        active_clients: List[int] = self.client_manager.get_active_clients()
        sample: List[int] = SampleManager.get_sample(1, active_clients, self.simulator.settings.sample_size)

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
            random_clients = [client.index for client in self.simulator.clients if client.index != self.index and client.online]
            random_clients = self.random.sample(random_clients, min(self.simulator.settings.sample_size * 10, len(random_clients)))

        for client_index in random_clients:
            self.send_message_to_client(client_index, "status", {"round": self.get_round_estimate(), "index": advertise_index, "change": change.value})

        # Update your own population view
        info = self.client_manager.last_active[self.index]
        self.client_manager.last_active[self.index] = (info[0], (advertise_index, change))

    def start_round(self, event: Event):
        round_info: Round = event.data
        self.round_info[round_info.round_nr] = round_info
        round_nr: int = round_info.round_nr
        if round_nr < 1:
            raise RuntimeError("Round number %d invalid!" % round_nr)
        
        # Should we test the model?
        if round_nr > 1 and (round_nr - 1) % self.simulator.settings.test_interval == 0:
            test_task_name = "test_%d_%d" % (self.index, round_nr)
            task = Task(test_task_name, "test", data={"model": round_info.model, "round": round_nr - 1, "time": self.simulator.current_time, "peer": self.index})
            self.add_compute_task(task)
            round_info.model = (test_task_name, 0)

        self.client_log(f"Client {self.index} starts training in round {round_nr}")

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

    def advertise_new_inventory(self, round_nr: int, chunks: List[Tuple[int, Set[str]]]):
        participants = SampleManager.get_sample(round_nr, self.client_manager.get_active_clients(), self.simulator.settings.sample_size)
        for participant in participants:
            self.send_message_to_client(participant, "has_chunks", {"round": round_nr, "chunks": chunks})

    def chunk_model(self, round_info: Round):
        """
        Chunk your current local model
        """

        # Add compute tasks for the chunks
        task_name = Task.generate_name("chunk")
        task = Task(task_name, "chunk", data={
            "model": round_info.model, "round": round_info.round_nr,
            "time": self.simulator.current_time, "peer": self.index,
            "n": self.simulator.settings.chunks_in_sample,
        })
        self.add_compute_task(task)

        all_chunks: List[Tuple[int, Set[str]]] = [(i, frozenset({task_name})) for i in range(self.simulator.settings.chunks_in_sample)]
        next_round_nr: int = round_info.round_nr + 1

        for chunk in all_chunks:
            self.available_chunks[next_round_nr].append(chunk)

        # Let the nodes in the next sample know about the availability of these chunks
        self.advertise_new_inventory(next_round_nr, all_chunks)

        self.last_round_completed = max(self.last_round_completed, round_info.round_nr)

    def start_outgoing_chunk_transfer(self, round_nr: int, to: int, chunk: Tuple[int, Set[str]]) -> None:
        event_data = {"from": self.index, "to": to, "model": None, "metadata": {"chunk": chunk, "round": round_nr}}
        start_transfer_event = Event(self.simulator.current_time, self.index, START_TRANSFER, data=event_data)
        self.start_transfer(start_transfer_event)        

    def pull_chunks_for_round(self, round_info: Round):
        if round_info.received_enough_chunks or not self.bw_scheduler.has_free_incoming_slot():
            # No space for another pull
            return

        # Determine chunk indices that are underrepresented
        chunk_idxs = list(range(self.simulator.settings.chunks_in_sample))
        random.shuffle(chunk_idxs)  # Shuffle before sorting to ensure randomness within groups
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
                    if round_info.has_received_chunk(chunk_idx, model_name):
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
                other = self.simulator.clients[owner]
                
                # Coalesce all the chunks from this owner into a single one
                model_names: Set[str] = set()
                for chunk in owners[owner]:
                    model_names.update(chunk[1])
                
                chunk_to_pull: Tuple[int, Set[str]] = (chunk_idx, frozenset(model_names))

                if other.bw_scheduler.has_free_outgoing_slot() and chunk_to_pull not in round_info.is_pulling:
                    round_info.is_pulling.add(chunk_to_pull)
                    other.start_outgoing_chunk_transfer(round_info.round_nr, self.index, chunk_to_pull)
                    break  # move to the next chunk index

            if not self.bw_scheduler.has_free_incoming_slot():
                break

    def start_transfer(self, event: Event):
        """
        Override the start_transfer because we are sending chunks instead of models.
        """
        to: int = event.data["to"]

        # TODO send the population view

        # round_info: Round = self.round_info[event.data["metadata"]["round"]]
        transfer_size = self.simulator.model_size // self.simulator.settings.chunks_in_sample
        # if to not in round_info.has_sent_view:
        #     population_view = self.client_manager.last_active
        #     event.data["metadata"]["population_view"] = population_view
        #     # TODO add the length of the serialized data view
        #     round_info.has_sent_view.add(to)

        receiver_scheduler: SlotBWScheduler = self.simulator.clients[to].bw_scheduler
        self.client_log(f"Client {self.index} starts sending chunk {event.data['metadata']['chunk']} to {to} for round {event.data['metadata']['round']}")
        self.bw_scheduler.add_transfer(receiver_scheduler, transfer_size, event.data["model"], event.data["metadata"])

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

    def update_my_inventory_with_new_chunk(self, round_nr: int, chunk: Tuple[int, Set[str]]) -> List[Tuple[int, Set[str]]]:
        new_chunks: List[Tuple[int, Set[str]]] = []

        # Compare the new chunk with existing chunks - if it's a subset, we can create a new chunk
        for existing_chunk in self.available_chunks[round_nr]:
            if existing_chunk[0] != chunk[0]:
                continue

            if existing_chunk[1].issubset(chunk[1]):
                new_chunk = (existing_chunk[0], chunk[1] - existing_chunk[1])
                new_chunks.append(new_chunk)
            elif chunk[1].issubset(existing_chunk[1]):
                new_chunk = (chunk[0], existing_chunk[1] - chunk[1])
                new_chunks.append(new_chunk)

        return new_chunks

    def received_model_chunk(self, from_client: int, round_nr: int, chunk: Tuple[int, Set[int]], population_view: Optional[Dict] = None) -> None:
        if population_view:
            self.client_manager.merge_population_views(population_view)

        round_info: Round = self.round_info[round_nr]
        self.client_log("Client %d received chunk %s from %d for round %d" % (self.index, chunk, from_client, round_nr))

        chunk_idx: int = chunk[0]
        for chunk_model_name in chunk[1]:
            round_info.received_chunks[chunk_idx].add(chunk_model_name)

        round_info.is_pulling.remove(chunk)
        
        # Update our inventory
        derived_chunks: List[Tuple[int, Set[str]]] = self.update_my_inventory_with_new_chunk(round_nr, chunk)
        self.available_chunks[round_nr].append(chunk)
        for derived_chunk in derived_chunks:
            self.available_chunks[round_nr].append(derived_chunk)
        self.advertise_new_inventory(round_nr, [chunk] + derived_chunks)

        counts_at_chunk_idxs = [len(chunks) for chunks in round_info.received_chunks]
        print(counts_at_chunk_idxs)

        # Did we receive sufficient chunks?
        if round_info.has_received_enough_chunks():
            self.client_log(f"Client {self.index} received enough chunks in round {round_nr}")
            round_info.received_enough_chunks = True

            # Kill any incoming transfer related to this round
            for slot_idx, transfer in enumerate(self.bw_scheduler.incoming_slots):
                if transfer and transfer.metadata["round"] == round_nr:
                    self.bw_scheduler.kill_transfer(transfer)
            
            # Reconstruct new model
            task_name = Task.generate_name("reconstruct_from_chunks")
            task = Task(task_name, "reconstruct_from_chunks", data={
                "chunks": round_info.received_chunks, "round": round_nr,  # TODO bug here since we changed the structure of received_chunks
                "time": self.simulator.current_time, "peer": self.index
            })
            self.add_compute_task(task)
            round_info.model = (task_name, 0)

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
        if event.data["type"] == "status":
            self.on_membership_advertisement(event.data["from"], event.data["message"]["change"], event.data["message"]["round"], event.data["message"]["index"])
        elif event.data["type"] == "has_chunks":
            # A client in some previous sample has chunks available - start pulling if we're not doing so already

            # Are we supposed to pull chunks for this round?
            round_nr = event.data["message"]["round"]
            participants: List[int] = SampleManager.get_sample(round_nr, self.client_manager.get_active_clients(), self.simulator.settings.sample_size)
            if self.index not in participants:
                return

            if round_nr not in self.round_info:
                self.round_info[round_nr] = Round(round_nr)
                self.round_info[round_nr].init_received_chunks(self.simulator.settings)

            # Update inventories
            for chunk in event.data["message"]["chunks"]:
                if chunk not in self.round_info[round_nr].inventories:
                    self.round_info[round_nr].inventories[chunk] = []
                self.round_info[round_nr].inventories[chunk].append(event.data["from"])

            self.pull_chunks_for_round(self.round_info[round_nr])            
        else:
            raise ValueError("Unknown message type: %s" % event.data["type"])

    def schedule_train(self, round_info: Round):
        round_info.is_training = True
        start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN, data={
            "model": round_info.model, "round": round_info.round_nr})
        self.simulator.schedule(start_train_event)
