from asyncio import Future
from random import Random
from typing import List

from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.bandwidth_scheduler import BWScheduler
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.simulation.conflux.round import Round
from dasklearn.simulation.conflux.sample_manager import SampleManager
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task


class ConfluxClient(AsynchronousClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.round_info: Dict[int, Round] = {}
        self.random = Random(index + 1234)

    def init_client(self, _: Event):
        sample: List[int] = SampleManager.get_sample(1, len(self.simulator.clients), self.simulator.settings.sample_size)
        
        if self.index in sample:
            round = Round(1)
            start_round_event = Event(self.simulator.current_time, self.index, START_ROUND, data=round)
            self.simulator.schedule(start_round_event)

    async def start_round(self, event: Event):
        round_info: Round = event.data
        self.round_info[round_info.round_nr] = round_info
        round_nr: int = round_info.round_nr
        if round_nr < 1:
            raise RuntimeError("Round number %d invalid!" % round_nr)
        
        # Should we test the model?
        if round_nr > 1 and (round_nr - 1) % self.simulator.settings.test_interval == 0:
            test_task_name = "test_%d_%d" % (self.index, round_nr)
            task = Task(test_task_name, "test", data={"model": round_info.model, "round": round_nr, "time": self.simulator.current_time, "peer": self.index})
            self.add_compute_task(task)
            round_info.model = (test_task_name, 0)

        self.client_log(f"Node {self.index} starts training in round {round_nr}")

        # 1. Train the model
        round_info.is_training = True
        self.schedule_train(round_info)
        round_info.model = await round_info.train_future
        round_info.is_training = False
        round_info.train_done = True

        # 2. Start sharing the model chunks
        participants_next_sample = SampleManager.get_sample(round_nr + 1, len(self.simulator.clients), self.simulator.settings.sample_size)
        participants_next_sample = sorted(participants_next_sample)
        await self.gossip_chunks(round_info, participants_next_sample)

        self.logger.info("Participant %d completed round %d", self.index, round_nr)
        self.round_info.pop(round_nr)

    async def gossip_chunks(self, round_info: Round, participants: List[int]) -> None:
        """
        Gossip chunks to the next sample of participants.
        """
        self.logger.info("Participant %d starts gossiping chunks in round %d", self.index, round_info.round_nr)

        # Add compute tasks for the chunks
        task_name = Task.generate_name("chunk")
        task = Task(task_name, "chunk", data={
            "model": round_info.model, "round": round_info.round_nr,
            "time": self.simulator.current_time, "peer": self.index
        })
        self.add_compute_task(task)

        for participant in participants:
            for chunk_idx in range(self.simulator.settings.chunks_in_sample):
                round_info.send_queue.append((participant, chunk_idx))

        self.random.shuffle(round_info.send_queue)

        for recipient_idx, chunk_idx in round_info.send_queue:
            # TODO we should probably start several transfers at the same time to better utilize outgoing bandwidth!
            await self.send_chunk(round_info, task_name, chunk_idx, recipient_idx)

    async def send_chunk(self, round_info: Round, model_name: str, chunk_idx: int, recipient_idx: int) -> None:
        event_data = {"from": self.index, "to": recipient_idx, "model": model_name, "metadata": {"chunk": chunk_idx, "round": round_info.round_nr + 1}}
        start_transfer_event = Event(self.simulator.current_time, self.index, START_TRANSFER, data=event_data)
        self.simulator.schedule(start_transfer_event)

        round_info.send_chunk_future = Future()
        await round_info.send_chunk_future

    def start_transfer(self, event: Event):
        """
        Override the start_transfer because we are sending chunks instead of models.
        """
        receiver_scheduler: BWScheduler = self.simulator.clients[event.data["to"]].bw_scheduler
        self.bw_scheduler.add_transfer(receiver_scheduler, self.simulator.model_size // self.simulator.settings.chunks_in_sample, event.data["model"], event.data["metadata"])

    def finish_outgoing_transfer(self, event):
        super().finish_outgoing_transfer(event)
        metadata: Dict = event.data["transfer"].metadata
        round_info = self.round_info[metadata["round"] - 1]
        round_info.send_chunk_future.set_result(None)

    def on_incoming_model(self, event: Event):
        """
        We received a model chunk.
        """
        metadata: Dict = event.data["metadata"]
        self.received_model_chunk(metadata["round"], event.data["model"], metadata["chunk"])

    def received_model_chunk(self, round_nr: int, model_name: str, chunk_idx: int) -> None:
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
        participants_previous_sample = SampleManager.get_sample(round_info.round_nr - 1, len(self.simulator.clients), self.simulator.settings.sample_size)
        for participant in participants_previous_sample:
            # TODO assumed to be instant for now
            self.send_message_to_client(participant, "has_enough_chunks", {"round": round_info.round_nr})

    def on_message(self, event: Event):
        if event.data["type"] == "has_enough_chunks":
            if event.data["message"]["round"] not in self.round_info:
                return  # It could be that the round is already completed, which is fine.
            round_info = self.round_info[event.data["message"]["round"]]
            round_info.send_queue = [(participant, chunk_idx) for participant, chunk_idx in round_info.send_queue if participant != event.data["from"]]
        else:
            raise ValueError("Unknown message type: %s" % event.data["type"])

    def schedule_train(self, round_info: Round):
        round_info.is_training = True
        start_train_event = Event(self.simulator.current_time, self.index, START_TRAIN, data={
            "model": round_info.model, "round": round_info.round_nr})
        self.simulator.schedule(start_train_event)

    def finish_train(self, event: Event):
        """
        We finished training.
        """
        cur_round: int = event.data["round"]
        round_info: Round = self.round_info[cur_round]
        round_info.is_training = False
        round_info.train_done = True
        round_info.train_future.set_result(event.data["model"])
