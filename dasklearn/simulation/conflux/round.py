from typing import List, Optional, Tuple

from dasklearn.simulation.conflux.chunk_manager import ChunkManager

from torch import nn

from dasklearn.simulation.conflux.settings import ConfluxSettings


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.model: Tuple[Optional[str], int] = (None, 0)

        self.sample_size: int = 0
        self.success_fraction: float = 0.0

        # State
        self.is_training: bool = False
        self.train_done: bool = False
        self.chunk_gossip_done: bool = False
        self.send_queue: List[Tuple[bytes, int]] = []
        self.received_enough_chunks: bool = False
        self.received_chunks: List[List[Tuple[str, int]]] = []  # Keep track of received chunks, indexed by chunk index

    def init_received_chunks(self, settings: ConfluxSettings):
        self.sample_size = settings.sample_size
        self.success_fraction = settings.success_fraction
        self.received_chunks = [[] for _ in range(settings.chunks_in_sample)]

    def has_received_enough_chunks(self):
        return all([(len(chunks) / self.sample_size) >= self.success_fraction for chunks in self.received_chunks])
