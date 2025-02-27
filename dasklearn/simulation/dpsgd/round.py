from typing import Dict, Optional, Tuple


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.model: Tuple[Optional[str], int] = (None, 0)
        self.incoming_models: Dict[int, Tuple[str, int]] = {}

        # State
        self.is_training: bool = False
        self.train_done: bool = False
        self.should_ignore: bool = False  # The node should ignore the round if it was offline when the round started
