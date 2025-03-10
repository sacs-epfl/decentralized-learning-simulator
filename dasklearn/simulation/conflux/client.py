import copy
from random import Random
from typing import List, Optional, Tuple

from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.bandwidth_scheduler import BWScheduler, Transfer
from dasklearn.simulation.conflux import NodeMembershipChange
from dasklearn.simulation.conflux.client_manager import ClientManager
from dasklearn.simulation.conflux.round import Round
from dasklearn.simulation.conflux.sample_manager import SampleManager
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task
from dasklearn.util import time_to_sec


class ConfluxClient(AsynchronousClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.round_info: Dict[int, Round] = {}
        self.random = Random(index + 1234)
        self.advertise_index: int = 1
        self.client_manager: ClientManager = ClientManager(self.index, 100000)
        self.train_sample_estimate: int = 0
        self.last_round_completed: int = 0

    def init_client(self, _: Event):
        sample: List[int] = SampleManager.get_sample(1, self.client_manager.get_active_clients(), self.simulator.settings.sample_size)

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

        # 2. Start sharing the model chunks with clients in the next sample
        participants_next_sample = SampleManager.get_sample(round_nr + 1, self.client_manager.get_active_clients(), self.simulator.settings.sample_size)
        self.gossip_chunks(round_info, participants_next_sample)

    def gossip_chunks(self, round_info: Round, participants: List[int]) -> None:
        """
        Gossip chunks to the next sample of participants.
        """
        self.client_log(f"Participant {self.index} starts gossiping chunks in round {round_info.round_nr}")

        # Add compute tasks for the chunks
        task_name = Task.generate_name("chunk")
        task = Task(task_name, "chunk", data={
            "model": round_info.model, "round": round_info.round_nr,
            "time": self.simulator.current_time, "peer": self.index,
            "n": self.simulator.settings.chunks_in_sample,
        })
        self.add_compute_task(task)

        # Queue chunk sending
        send_queue: List[Tuple[int, int]] = []
        for participant in participants:
            for chunk_idx in range(self.simulator.settings.chunks_in_sample):
                send_queue.append((participant, chunk_idx))

        self.random.shuffle(send_queue)

        while send_queue:
            participant, chunk_idx = send_queue.pop(0)
            self.send_chunk(round_info, task_name, chunk_idx, participant)

        self.last_round_completed = max(self.last_round_completed, round_info.round_nr)

    def send_chunk(self, round_info: Round, model_name: str, chunk_idx: int, recipient_idx: int) -> None:
        event_data = {"from": self.index, "to": recipient_idx, "model": model_name, "metadata": {"chunk": chunk_idx, "round": round_info.round_nr + 1}}
        start_transfer_event = Event(self.simulator.current_time, self.index, START_TRANSFER, data=event_data)
        self.simulator.schedule(start_transfer_event)

    def start_transfer(self, event: Event):
        """
        Override the start_transfer because we are sending chunks instead of models.
        """
        to: int = event.data["to"]
        round_info: Round = self.round_info[event.data["metadata"]["round"] - 1]
        transfer_size = self.simulator.model_size // self.simulator.settings.chunks_in_sample
        if to not in round_info.has_sent_view:
            population_view = self.client_manager.last_active
            event.data["metadata"]["population_view"] = population_view
            # TODO add the length of the serialized data view
            round_info.has_sent_view.add(to)

        receiver_scheduler: BWScheduler = self.simulator.clients[to].bw_scheduler
        self.bw_scheduler.add_transfer(receiver_scheduler, transfer_size, event.data["model"], event.data["metadata"])
        #self.client_log(f"Client {self.index} starts sending chunk {event.data['metadata']['chunk']} to {to} in round {event.data['metadata']['round'] - 1}")

    def finish_outgoing_transfer(self, event):
        super().finish_outgoing_transfer(event)
        metadata: Dict = event.data["transfer"].metadata
        round_info = self.round_info[metadata["round"] - 1]

        to: int = event.data["transfer"].receiver_scheduler.my_id
        transfer_duration = time_to_sec(event.data["transfer"].duration)
        #self.client_log(f"Client {self.index} finished sending chunk {metadata['chunk']} to {to} in round {metadata['round'] - 1} (duration: {transfer_duration}s)")

    def on_incoming_model(self, event: Event):
        """
        We received a model chunk.
        """
        if not self.online:
            return

        metadata: Dict = event.data["metadata"]
        from_client: int = event.data["from"]
        self.client_manager.update_client_activity(from_client, max(metadata["round"], self.get_round_estimate()))
        self.received_model_chunk(from_client, metadata["round"], event.data["model"], metadata["chunk"], metadata.get("population_view", None))

    def received_model_chunk(self, from_client: int, round_nr: int, model_name: str, chunk_idx: int, population_view: Optional[Dict] = None) -> None:
        if population_view:
            self.client_manager.merge_population_views(population_view)

        if round_nr == self.last_round_completed:
            # We received a chunk from a round we already completed
            self.send_message_to_client(from_client, "has_enough_chunks", {"round": round_nr - 1})
            return
        elif self.last_round_completed > round_nr:
            self.logger.warning("Client %d received a chunk from client %d for a round we already completed (%d < %d)" % (self.index, from_client, round_nr, self.last_round_completed))
            self.send_message_to_client(from_client, "stale", {"last_round_completed": self.last_round_completed})
            return

        self.train_sample_estimate = max(self.train_sample_estimate, round_nr)
        if round_nr not in self.round_info:
            # We received a chunk but haven't started this round yet - store it.
            new_round = Round(round_nr)
            self.round_info[round_nr] = new_round
            new_round.init_received_chunks(self.simulator.settings)
            new_round.received_chunks[chunk_idx].append((model_name, chunk_idx))
        else:
            if self.round_info[round_nr].received_enough_chunks:
                # We have already received enough chunks - ignore this chunk.
                return

            # Otherwise, process it right away!
            self.round_info[round_nr].received_chunks[chunk_idx].append((model_name, chunk_idx))

        # Did we receive sufficient chunks?
        if self.round_info[round_nr].has_received_enough_chunks():
            self.client_log(f"Client {self.index} received enough chunks in round {round_nr}")
            self.round_info[round_nr].received_enough_chunks = True
            self.inform_nodes_in_previous_sample(self.round_info[round_nr])
            
            # Reconstruct new model
            task_name = Task.generate_name("reconstruct_from_chunks")
            task = Task(task_name, "reconstruct_from_chunks", data={
                "chunks": self.round_info[round_nr].received_chunks, "round": round_nr,
                "time": self.simulator.current_time, "peer": self.index
            })
            self.add_compute_task(task)
            self.round_info[round_nr].model = (task_name, 0)

            start_round_event = Event(self.simulator.current_time, self.index, START_ROUND, data=self.round_info[round_nr])
            self.simulator.schedule(start_round_event)

    def inform_nodes_in_previous_sample(self, round_info: Round):
        participants_previous_sample = SampleManager.get_sample(round_info.round_nr - 1, self.client_manager.get_active_clients(), self.simulator.settings.sample_size)
        for participant in participants_previous_sample:
            # TODO assumed to be instant for now
            self.send_message_to_client(participant, "has_enough_chunks", {"round": round_info.round_nr - 1})

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
        if event.data["type"] == "has_enough_chunks":
            if event.data["message"]["round"] not in self.round_info:
                return  # It could be that the round is already completed, which is fine.
            
            from_client: int = event.data["from"]

            # Remove the scheduled transfers that are not relevant anymore
            for transfer in list(self.bw_scheduler.outgoing_requests.values()):
                if transfer.metadata["round"] == (event.data["message"]["round"] + 1) and transfer.receiver_scheduler.my_id == from_client:
                    self.bw_scheduler.kill_transfer(transfer)          
        
        elif event.data["type"] == "stale":
            self.logger.warning("Client %d received a stale message from client %d (t=%d)" % (self.index, event.data["from"], self.simulator.cur_time_in_sec()))
            last_round: int = event.data["message"]["last_round_completed"]

            # Kill all the transfers that are not relevant anymore
            for transfer in list(self.bw_scheduler.outgoing_requests.values()):
                if transfer.metadata["round"] < last_round:
                    self.bw_scheduler.kill_transfer(transfer)

        elif event.data["type"] == "status":
            self.on_membership_advertisement(event.data["from"], event.data["message"]["change"], event.data["message"]["round"], event.data["message"]["index"])
        else:
            raise ValueError("Unknown message type: %s" % event.data["type"])

    def schedule_train(self, round_info: Round):
        round_info.is_training = True
        start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN, data={
            "model": round_info.model, "round": round_info.round_nr})
        self.simulator.schedule(start_train_event)
