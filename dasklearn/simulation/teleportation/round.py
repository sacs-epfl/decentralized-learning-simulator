from asyncio import Future
from typing import Dict, List, Optional, Tuple


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.model: Tuple[Optional[str], int] = (None, 0)
        self.incoming_models: Dict[int, Tuple[str, List[float]]] = {}

        # State
        self.is_training: bool = False
        self.train_done: bool = False
        self.train_future: Future = Future()
