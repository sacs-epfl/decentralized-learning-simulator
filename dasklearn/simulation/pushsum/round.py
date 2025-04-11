from typing import Dict, List, Optional, Tuple, Set


class Round:
    """
    Stores the state of a client in a particular round.
    """

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.model: Tuple[Optional[str], int] = (None, 0)

        # State
        self.is_training: bool = False
        self.train_done: bool = False
        self.push_sum_start_time: int = 0
        self.push_sum_ended: bool = False
        self.clients_ready: List[int] = []
        self.pushsum_chunks: List[Tuple[str, int]] = []
        self.weights: List[float] = []  # Weights for each chunk
        self.sending: Set[int, int] = set()  # Set of clients - chunk index we are sending

    def init_received_chunks(self, chunked_model: str, chunks_in_sample: int):
        self.pushsum_chunks = [(chunked_model, i) for i in range(chunks_in_sample)]
        for _ in range(chunks_in_sample):
            self.weights.append(1.0)
