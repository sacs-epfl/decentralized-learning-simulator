from random import Random
from typing import List, Optional, Set, Tuple

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
        self.received_enough_chunks: bool = False
        self.received_chunks: List[List[Tuple[str, int, int]]] = []  # Keep track of received chunks, indexed by chunk index
        self.has_sent_view: Set[int] = set()  # Keep track of which clients have received our view
        self.clients_ready_prev_round: Set[int] = set() # Keep track of clients that are ready to send us chunks in the previous round
        self.pulling_chunks: List[Optional[Tuple[int, int]]] = [None] * 5  # Keep track of which chunks we are currently pulling

    def init_received_chunks(self, settings: ConfluxSettings):
        self.sample_size = settings.sample_size
        self.success_fraction = settings.success_fraction
        self.received_chunks = [[] for _ in range(settings.chunks_in_sample)]

    def has_received_enough_chunks(self):
        return all([(len(chunks) / self.sample_size) >= self.success_fraction for chunks in self.received_chunks])
    
    def has_free_slot(self):
        return any(slot is None for slot in self.pulling_chunks)
    
    def fill_slot(self, slot_info: Tuple[int, int]) -> None:
        for slot_idx, slot in enumerate(self.pulling_chunks):
            if slot is None:
                self.pulling_chunks[slot_idx] = slot_info
                return
        
        raise ValueError("No empty slot available to fill")
    
    def empty_slot(self, slot_info: Tuple[int, int]) -> None:
        for slot_idx, slot in enumerate(self.pulling_chunks):
            if slot == slot_info:
                self.pulling_chunks[slot_idx] = None
                return
        
        raise ValueError("Slot not found to empty")
    
    def get_next_chunk_suggestion(self, available_clients: Set[int]) -> Optional[Tuple[int, int]]:
        """
        Suggest the next (sender, chunk_index) pair to pull, making sure that:
          - We haven't already received that (sender, chunk_index) combination.
          - We are not already pulling that same pair.
          - We prefer a chunk index that is entirely missing, and among those,
            a sender from which we have received few chunks overall.
        
        Args:
            available_clients (Set[int]): The set of client IDs that are available to send chunks.
        
        Returns:
            Optional[Tuple[int, int]]: A tuple (sender, chunk_index) for the next pull,
            or None if no candidate is available.
        """
        num_chunks = len(self.received_chunks)
        candidates = []
        rand = Random(42)

        # Iterate over each available sender and each possible chunk index.
        for sender in available_clients:
            for chunk_idx in range(num_chunks):
                # Skip if this exact (sender, chunk_idx) is currently being pulled.
                if (sender, chunk_idx) in self.pulling_chunks:
                    continue

                # Check if we already have a chunk from this sender for this chunk index.
                already_received = any(entry[2] == sender for entry in self.received_chunks[chunk_idx])
                if not already_received:
                    candidates.append((sender, chunk_idx))

        if not candidates:
            return None

        # First, prefer candidates for which the entire chunk index is missing.
        missing_chunk_candidates = [
            (sender, chunk_idx) for (sender, chunk_idx) in candidates
            if len(self.received_chunks[chunk_idx]) == 0
        ]

        # Compute overall counts: how many chunks have been received from each sender.
        sender_counts = {sender: 0 for sender in available_clients}
        for chunk_list in self.received_chunks:
            for entry in chunk_list:
                s = entry[2]
                if s in sender_counts:
                    sender_counts[s] += 1

        if missing_chunk_candidates:
            # Among candidates with a missing chunk index, pick the sender with the fewest chunks.
            min_count = min(sender_counts[sender] for sender, _ in missing_chunk_candidates)
            best_candidates = [cand for cand in missing_chunk_candidates if sender_counts[cand[0]] == min_count]
            return rand.choice(best_candidates)
        else:
            # Otherwise, consider candidates for the chunk index with the fewest total chunks.
            chunk_counts = [len(self.received_chunks[i]) for i in range(num_chunks)]
            min_chunk_count = min(chunk_counts)
            filtered_candidates = [
                (sender, chunk_idx) for (sender, chunk_idx) in candidates
                if len(self.received_chunks[chunk_idx]) == min_chunk_count
            ]
            # Again, choose the candidate from a sender with few chunks overall.
            min_sender_count = min(sender_counts[sender] for sender, _ in filtered_candidates)
            best_candidates = [cand for cand in filtered_candidates if sender_counts[cand[0]] == min_sender_count]
            return rand.choice(best_candidates)
