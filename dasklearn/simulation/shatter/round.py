from typing import List, Optional, Tuple

from dasklearn.simulation.shatter.settings import ShatterSettings


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.model: Tuple[Optional[str], int] = None
        self.my_chunked_model: Optional[str] = None
        self.num_received_chunks: int = 0
        self.received_chunks: List[List[Tuple[str, int]]] = []  # Keep track of received chunks, indexed by chunk index

        self.state = "WAITING"
        self.chunks_sent = False
        self.train_done: bool = False
        self.should_ignore: bool = False  # The node should ignore the round if it was offline when the round started

    def init_received_chunks(self, settings: ShatterSettings):
        self.received_chunks = [[] for _ in range(settings.k)]
