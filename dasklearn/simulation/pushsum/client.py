import random
from typing import Dict, List, Optional, Tuple, Set

from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.conflux.client_manager import ClientManager
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.simulation.conflux.sample_manager import SampleManager
from dasklearn.simulation.events import *
from dasklearn.simulation.pushsum import NodeMembershipChange
from dasklearn.simulation.pushsum.round import Round
from dasklearn.simulation.slot_bandwidth_scheduler import SlotBWScheduler
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS, time_to_sec


class PushSumClient(AsynchronousClient):
    """
    Client implementation for the PushSum algorithm.
    """

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.round_info: Dict[int, Round] = {}
        self.client_manager: ClientManager = ClientManager(self.index, 100000)
        self.train_sample_estimate: int = 0

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
        if round_nr > 1 and self.simulator.settings.test_method == "individual" and (round_nr - 1) % self.simulator.settings.test_interval == 0:
            test_task_name = "test_%d_%d" % (self.index, round_nr)
            task = Task(test_task_name, "test", data={"model": round_info.model, "round": round_nr - 1, "time": self.simulator.current_time, "peer": self.index})
            self.add_compute_task(task)
            round_info.model = (test_task_name, 0)

        self.logger.info(f"[t=%.3f] Client %d starts training in round %d", time_to_sec(self.simulator.current_time), self.index, round_nr)

        # 1. Train the model
        self.schedule_train(round_info)

    def schedule_train(self, round_info: Round):
        round_info.is_training = True
        start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN, data={
            "model": round_info.model, "round": round_info.round_nr})
        self.simulator.schedule(start_train_event)

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
        
        # Store the chunked model
        round_info.init_received_chunks(task_name, self.simulator.settings.chunks_in_sample)
        
        # Initialize weights for each chunk to 1.0
        for i in range(self.simulator.settings.chunks_in_sample):
            round_info.weights[i] = 1.0

        # Inform all nodes in the current sample that this node is ready
        sample: List[int] = SampleManager.get_sample(
            round_info.round_nr,
            self.client_manager.get_active_clients(),
            self.simulator.settings.sample_size
        )
        for client_index in sample:
            if client_index != self.index:
                self.send_message_to_client(client_index, "ready_for_push_sum", {"round": round_info.round_nr})

        # Start the push-sum process
        self.start_push_sum(round_info)

    def on_message(self, event: Event):
        if event.data["type"] == "ready_for_push_sum":
            round_info: Round = self.round_info[event.data["message"]["round"]]
            round_info.clients_ready.append(event.data["from"])
            self.fill_all_available_slots(round_info.round_nr)
        else:
            raise ValueError("Unknown message type: %s" % event.data["type"])
        
    def start_push_sum(self, round_info: Round):
        """
        Start the push-sum process for model synchronization
        """
        self.client_log(f"Client {self.index} starting push-sum process in round {round_info.round_nr}")
        
        # Set the start time for the push-sum process
        round_info.push_sum_start_time = self.simulator.current_time
        
        # Fill all available slots immediately
        self.fill_all_available_slots(round_info.round_nr)
        
        # Schedule the end of push-sum after the specified duration
        end_time = self.simulator.current_time + self.simulator.settings.push_sum_duration * MICROSECONDS
        end_event = Event(end_time, self.index, FINISH_PUSH_SUM, 
                          data={"round": round_info.round_nr})
        self.simulator.schedule(end_event)
    
    def fill_all_available_slots(self, round_nr: int):
        """
        Fill all available slots with chunk transfers
        """
        if round_nr not in self.round_info:
            return
            
        round_info = self.round_info[round_nr]
        if not round_info.train_done:
            return
        
        # Keep sending chunks until we can't send more
        while self.bw_scheduler.has_free_outgoing_slot():
            res = self.send_chunk(round_nr)
            if not res:
                break

    def send_chunk(self, round_nr: int) -> bool:
        round_info = self.round_info[round_nr]
        
        # Check if pushsum period has ended
        if round_info.push_sum_ended:
            return False

        # Check if we have slots available for sending
        if not self.bw_scheduler.has_free_outgoing_slot():
            return False

        # Find eligible recipients (online clients in the sample with free incoming slots)
        eligible_recipients = []
        for idx in round_info.clients_ready:
            if idx != self.index and self.simulator.clients[idx].online:
                receiver = self.simulator.clients[idx]
                if receiver.bw_scheduler.has_free_incoming_slot():
                    eligible_recipients.append(idx)
        
        if not eligible_recipients:
            return False
        
        # Create eligible recipient-chunk index pairs
        eligible_send: List[Tuple[int, int]] = []
        for idx in eligible_recipients:
            for chunk_idx in range(self.simulator.settings.chunks_in_sample):
                if (idx, chunk_idx) not in round_info.sending:
                    eligible_send.append((idx, chunk_idx))
        
        if not eligible_send:
            return False
        
        target_idx, chunk_idx = random.choice(eligible_send)
        round_info.sending.add((target_idx, chunk_idx))
        
        # Split our weight
        task_name = Task.generate_name("split_chunk")
        task = Task(task_name, "split_chunk", data={
            "chunk": round_info.pushsum_chunks[chunk_idx], "round": round_info.round_nr,
            "time": self.simulator.current_time, "peer": self.index
        })
        self.add_compute_task(task)
        round_info.pushsum_chunks[chunk_idx] = (task_name, 0)

        weight = round_info.weights[chunk_idx]
        send_weight = weight / 2
        round_info.weights[chunk_idx] = weight / 2
        
        # Send the chunk with its weight
        self.send_chunk_with_weight(round_info, round_info.pushsum_chunks[chunk_idx], chunk_idx, target_idx, send_weight)
        return True
        
    def send_chunk_with_weight(self, round_info: Round, model_name: str, chunk_idx: int, recipient_idx: int, weight: float):
        """
        Send a chunk with its weight to another node
        """
        self.client_log(f"Client {self.index} sending chunk {chunk_idx} with weight {weight:.4f} to {recipient_idx}")
        
        event_data = {
            "from": self.index,
            "to": recipient_idx,
            "model": model_name,
            "metadata": {
                "to": recipient_idx,
                "chunk": chunk_idx,
                "round": round_info.round_nr,
                "weight": weight
            }
        }
        start_transfer_event = Event(self.simulator.current_time, self.index, START_TRANSFER, data=event_data)
        self.start_transfer(start_transfer_event)
        
    def start_transfer(self, event: Event):
        """
        Start a bandwidth-slot based transfer
        """
        to = event.data["to"]
        transfer_size = self.simulator.model_size // self.simulator.settings.chunks_in_sample
        receiver_scheduler = self.simulator.clients[to].bw_scheduler
        self.bw_scheduler.add_transfer(receiver_scheduler, transfer_size, event.data["model"], event.data["metadata"])
    
    def finish_outgoing_transfer(self, event: Event):
        super().finish_outgoing_transfer(event)
        
        metadata = event.data["transfer"].metadata
        if "chunk" in metadata:
            round_info: Round = self.round_info[metadata["round"]]
            round_info.sending.remove((metadata["to"], metadata["chunk"]))
            self.fill_all_available_slots(metadata["round"])
        
    def on_incoming_model(self, event: Event):
        """
        Process incoming model chunks during push-sum
        """
        metadata = event.data["metadata"]
        
        # Check if this is a chunk or a complete model
        if "chunk" in metadata:
            # This is a chunk with a weight
            chunk_idx = metadata["chunk"]
            round_nr = metadata["round"]
            weight = metadata["weight"]
            model_name = event.data["model"]
            
            self.client_log(f"Client {self.index} received chunk {chunk_idx} from {event.data['from']} "
                           f"with weight {weight:.4f} in round {round_nr}")
            
            self.received_model_chunk(round_nr, model_name, chunk_idx, weight)
        else:
            # This is a complete model for the next round
            self.process_incoming_complete_model(event)
            
    def received_model_chunk(self, round_nr: int, model_name: str, chunk_idx: int, weight: float):
        """
        Process a received model chunk with its weight
        """
        if round_nr not in self.round_info:
            new_round = Round(round_nr)
            self.round_info[round_nr] = new_round
        round_info = self.round_info[round_nr]
        
        # Update weight for this chunk
        if chunk_idx in round_info.weights:
            round_info.weights[chunk_idx] += weight
        else:
            round_info.weights[chunk_idx] = weight
            
        # Add to received chunks
        task_name = Task.generate_name("add_chunks")
        task = Task(task_name, "add_chunks", data={
            "chunks": [(model_name, chunk_idx), round_info.pushsum_chunks[chunk_idx]], "round": round_info.round_nr,
            "time": self.simulator.current_time, "peer": self.index
        })
        self.add_compute_task(task)
        round_info.pushsum_chunks[chunk_idx] = (task_name, 0)

        self.fill_all_available_slots(round_nr)

    def finish_push_sum(self, event: Event):
        """
        End the push-sum process and reconstruct the model
        """
        round_nr = event.data["round"]
        if round_nr not in self.round_info:
            return

        round_info = self.round_info[round_nr]
        round_info.push_sum_ended = True

        self.client_log(f"Client {self.index} finished push-sum in round {round_nr}, reconstructing model")

        # Reconstruct the model from weighted chunks
        task_name = Task.generate_name("weighted_reconstruct")
        task = Task(task_name, "weighted_reconstruct_from_chunks", data={
            "chunks": round_info.pushsum_chunks,
            "weights": round_info.weights,
            "round": round_nr,
            "time": self.simulator.current_time,
            "peer": self.index
        })
        self.add_compute_task(task)
        round_info.model = (task_name, 0)

        # Kill all the outgoing transfers related to this round
        for transfer in self.bw_scheduler.outgoing_slots:
            if transfer and transfer.metadata["round"] == round_nr:
                self.bw_scheduler.kill_transfer(transfer)
        round_info.sending.clear()

        # Send to next sample
        self.send_to_next_sample(round_info)
    
    def send_to_next_sample(self, round_info: Round):
        """
        Send the reconstructed model to a client in the next sample
        """
        round_nr = round_info.round_nr
        next_round_nr = round_nr + 1

        current_sample = SampleManager.get_sample(
            round_nr, 
            self.client_manager.get_active_clients(), 
            self.simulator.settings.sample_size
        )
        if self.index not in current_sample:
            self.logger.warning("Client %d not in current sample %s", self.index, current_sample)
            return

        index_in_current = current_sample.index(self.index)

        # Get next sample
        next_sample = SampleManager.get_sample(
            next_round_nr,
            self.client_manager.get_active_clients(),
            self.simulator.settings.sample_size
        )
                
        if self.index in current_sample and next_sample:
            target_idx = next_sample[index_in_current]
            self.client_log(f"Client {self.index} sending model to {target_idx} for round {next_round_nr}")
            self.send_model(target_idx, round_info.model, metadata={"round": next_round_nr})
            # TODO we assume all nodes are online here!
    
    def process_incoming_complete_model(self, event: Event):
        """
        Process a complete model for the next round
        """
        round_nr = event.data["metadata"]["round"]
        model = event.data["model"]
        
        self.client_log(f"Client {self.index} received complete model from {event.data['from']} for round {round_nr}")
        
        # Create new round
        new_round = Round(round_nr)
        new_round.model = model
        
        # Start the round
        start_round_event = Event(self.simulator.current_time, self.index, START_ROUND, data=new_round)
        self.simulator.schedule(start_round_event)
