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
        self.scheduled_push_sum_end: bool = False
        self.clients_ready: List[int] = []
        self.pushsum_chunks: List[Tuple[str, int]] = []
        self.sending: Set[int, int] = set()  # Set of clients - chunk index we are sending

    def split_chunk(self, chunk: Dict[Tuple[str, int], float]) -> Dict[Tuple[str, int], float]:
        """
        Splits the chunk into two halves.
        """
        split_chunk = {}
        for key, value in chunk.items():
            split_chunk[key] = value / 2
        return split_chunk
    
    def merge_chunk(self, other_chunk: Dict[Tuple[str, int], float], chunk_idx: int) -> None:
        """
        Merge the incoming chunk with the existing chunk at the particular chunk index
        """
        local_chunk = self.pushsum_chunks[chunk_idx]
        for key, value in other_chunk.items():
            if key in local_chunk:
                local_chunk[key] += value
            else:
                local_chunk[key] = value

    def init_received_chunks(self, chunked_model: str, chunks_in_sample: int):
        self.pushsum_chunks = [{(chunked_model, i): 1} for i in range(chunks_in_sample)]
