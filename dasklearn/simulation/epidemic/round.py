from typing import Dict, Optional


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.model: Optional[str] = None
        self.incoming_models: Dict[int, str] = {}

        # State
        self.is_training: bool = False
        self.train_done: bool = False
