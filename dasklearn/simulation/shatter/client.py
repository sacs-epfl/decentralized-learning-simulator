from asyncio import Event
from typing import Dict
from dasklearn.simulation.bandwidth_scheduler import BWScheduler
from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import *
from dasklearn.simulation.shatter.round import Round
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS

class ShatterClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.round_info: Dict[int, Round] = {}

    def init_client(self, event: Event):
        # Start with round 1; local model comes from previous round—None for round 1.
        self.schedule_next_round({"round": 1, "model": (None, 0)})

    def schedule_next_round(self, round_data: Dict):
        if self.simulator.settings.rounds > 0 and round_data["round"] > self.simulator.settings.rounds:
            return
        round_nr = round_data["round"]
        # Create new Round; note that we initialize the state and chunks_sent.
        new_round = Round(round_nr)
        new_round.model = round_data["model"]
        new_round.should_ignore = not self.online
        new_round.state = "WAIT_FOR_MODEL"  # Initially waiting to start local training.
        self.round_info[round_nr] = new_round
        start_round_event = Event(self.simulator.current_time, self.index, START_ROUND, data=round_data)
        self.simulator.schedule(start_round_event)

    def start_round(self, event: Event):
        round_nr = event.data["round"]
        round_info = self.round_info[round_nr]
        self.client_log("Client %d starting round %d" % (self.index, round_nr))
        self.progress_round(round_info)

    def progress_round(self, round_info: Round):
        """
        Central state machine for a round.
        States:
          - "TRAINING": local training is running; remote chunks may arrive concurrently.
          - "WAITING": training finished; waiting for all expected chunks.
          - "DONE": round finished.
        """
        cur_round = round_info.round_nr

        if round_info.state == "WAIT_FOR_MODEL":
            if round_info.model is not None:
                self.client_log("Client %d: Round %d transitioning from WAIT_FOR_MODEL to TRAINING" % (self.index, cur_round))
                round_info.state = "TRAINING"
                self._start_training(round_info)
            return
        
        # In TRAINING state, if training has finished and we haven't sent chunks yet, do so.
        if round_info.state == "TRAINING":
            if round_info.train_done:
                self.client_log("Client %d: Round %d training finished, sending chunks" % (self.index, cur_round))

                event = Event(self.simulator.current_time, self.index, SEND_CHUNKS, data={"round": cur_round})
                self.simulator.schedule(event)
                round_info.state = "SENDING_CHUNKS"
            return
        
        if round_info.state == "SENDING_CHUNKS":
            if round_info.chunks_sent:
                self.client_log("Client %d: Round %d transitioning from SENDING to WAIT_FOR_CHUNKS" % (self.index, cur_round))
                round_info.state = "WAIT_FOR_CHUNKS"
            # No return here as we may have received all chunks already.

        # In WAITING state, check if we have received all expected chunks.
        if round_info.state == "WAIT_FOR_CHUNKS":
            if len(self.simulator.topologies) < cur_round:
                self.simulator.add_topology()

            my_vns = [self.index * self.simulator.settings.k + i for i in range(self.simulator.settings.k)]
            total_nbs = sum(len(list(self.simulator.topologies[cur_round - 1].predecessors(vn))) for vn in my_vns)
            #self.client_log("Client %d: Round %d waiting for %d chunks (received %d)" % (self.index, cur_round, total_nbs, round_info.num_received_chunks))
            assert round_info.num_received_chunks <= total_nbs, "Received more chunks than expected (%d > %d)" % (round_info.num_received_chunks, total_nbs)
            if round_info.num_received_chunks == total_nbs:
                self._aggregate(round_info)
                self.client_log("Client %d finished round %d" % (self.index, round_info.round_nr))
                self.simulator.set_finished(round_info.round_nr)
                self.round_info.pop(cur_round)
                next_round_nr = cur_round + 1

                if next_round_nr not in self.round_info:
                    self.schedule_next_round({"round": next_round_nr, "model": round_info.model})
                else:
                    next_round: Round = self.round_info[next_round_nr]
                    next_round.model = round_info.model
                    self.progress_round(next_round)
    
    def _start_training(self, round_info: Round):
        cur_round = round_info.round_nr
        self.client_log("Client %d starting training for round %d" % (self.index, cur_round))
        if not round_info.should_ignore:
            start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN,
                                      data={"model": round_info.model, "round": cur_round})
            self.simulator.schedule(start_train_event)
        else:
            finish_train_event = Event(self.simulator.current_time, self.index, FINISH_TRAIN,
                                       data={"round": cur_round, "model": round_info.model, "train_time": 0})
            self.simulator.schedule(finish_train_event)

    def finish_train(self, event: Event):
        cur_round = event.data["round"]
        self.compute_time += event.data["train_time"]
        self.client_log("Client %d finished model training in round %d (online? %s)" %
                         (self.index, cur_round, self.online))
        if cur_round not in self.round_info:
            raise RuntimeError("Client %d does not know about round %d after training finished!" % (self.index, cur_round))
        round_info = self.round_info[cur_round]
        round_info.model = event.data["model"]
        round_info.train_done = True

        # Chunk the model
        if not round_info.should_ignore:
            task_name = Task.generate_name("chunk")
            task = Task(task_name, "chunk", data={
                "model": round_info.model, "round": cur_round,
                "time": self.simulator.current_time, "peer": self.index,
                "n": self.simulator.settings.k,
            })
            self.add_compute_task(task)
            round_info.my_chunked_model = task_name

        self.progress_round(round_info)

    def send_chunks(self, event: Event):
        round_info = self.round_info[event.data["round"]]
        cur_round = round_info.round_nr
        self.client_log("Participant %d starts gossiping chunks in round %d" % (self.index, cur_round))

        if len(self.simulator.topologies) < cur_round:
            self.simulator.add_topology()

        topology = self.simulator.topologies[cur_round - 1]
        my_vns = [self.index * self.simulator.settings.k + i for i in range(self.simulator.settings.k)]
        for chunk_idx, vn in enumerate(my_vns):
            for neighbour in topology.successors(vn):
                rn = neighbour // self.simulator.settings.k
                self.client_log("Client %d sending chunk %d to %d in round %d" % (self.index, chunk_idx, rn, cur_round))
                if not round_info.should_ignore and self.simulator.clients[rn].online:
                    self.send_chunk(round_info, round_info.my_chunked_model, chunk_idx, rn)
                else:
                    self.send_message_to_client(rn, "offline_chunk", {"round": cur_round, "chunk": chunk_idx})

        round_info.chunks_sent = True
        self.progress_round(round_info)

    def send_chunk(self, round_info: Round, model_name: str, chunk_idx: int, recipient_idx: int):
        event_data = {
            "from": self.index,
            "to": recipient_idx,
            "model": model_name,
            "metadata": {"chunk": chunk_idx, "round": round_info.round_nr}
        }
        start_transfer_event = Event(self.simulator.current_time, self.index, START_TRANSFER, data=event_data)
        self.simulator.schedule(start_transfer_event)

    def start_transfer(self, event: Event):
        to = event.data["to"]
        transfer_size = self.simulator.model_size // self.simulator.settings.k
        receiver_scheduler: BWScheduler = self.simulator.clients[to].bw_scheduler
        self.bw_scheduler.add_transfer(receiver_scheduler, transfer_size, event.data["model"], event.data["metadata"])

    def on_message(self, event: Event):
        if event.data["type"] == "offline_chunk":
            self.on_offline_sentinel(event.data["from"], event.data["message"]["round"], event.data["message"]["chunk"])

    def on_offline_sentinel(self, from_client: int, round_nr: int, chunk_idx: int):
        event = {
            "from": from_client,
            "model": None,
            "metadata": {"round": round_nr, "chunk": chunk_idx},
        }
        self.on_incoming_model(Event(self.simulator.current_time, self.index, INCOMING_MODEL, data=event))

    def on_incoming_model(self, event: Event):
        metadata = event.data["metadata"]
        self.received_model_chunk(event.data["from"], metadata["round"], event.data["model"], metadata["chunk"])

    def received_model_chunk(self, from_client: int, round_nr: int, model_name: str, chunk_idx: int):
        self.client_log("Client %d received from %d model chunk %s/%d in round %d" %
                         (self.index, from_client, model_name, chunk_idx, round_nr))
        if round_nr not in self.round_info:
            self.schedule_next_round({"round": round_nr, "model": None})
        round_info = self.round_info[round_nr]
        if not round_info.received_chunks:
            round_info.init_received_chunks(self.simulator.settings)
        round_info.num_received_chunks += 1
        if model_name:
            round_info.received_chunks[chunk_idx].append((model_name, chunk_idx))
        self.progress_round(round_info)

    def _aggregate(self, round_info: Round):
        round_nr = round_info.round_nr
        self.client_log("Client %d will aggregate in round %d" % (self.index, round_nr))
        if not round_info.should_ignore:
            for chunk_idx in range(self.simulator.settings.k):
                round_info.received_chunks[chunk_idx].append((round_info.my_chunked_model, chunk_idx))

            task_name = Task.generate_name("reconstruct_from_chunks")
            task = Task(task_name, "reconstruct_from_chunks", data={
                "chunks": round_info.received_chunks, "round": round_nr,
                "time": self.simulator.current_time, "peer": self.index
            })
            self.add_compute_task(task)
            round_info.model = (task_name, 0)
            if (self.simulator.settings.stop == "rounds" and
                self.simulator.settings.test_interval > 0 and round_nr % self.simulator.settings.test_interval == 0):
                test_task_name = "test_%d_%d" % (self.index, round_nr)
                task = Task(test_task_name, "test", data={
                    "model": round_info.model, "round": round_nr,
                    "time": self.simulator.current_time, "peer": self.index
                })
                self.add_compute_task(task)
                round_info.model = (test_task_name, 0)
