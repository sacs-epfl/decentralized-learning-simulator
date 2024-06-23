from typing import Dict, Optional, Tuple, List


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.model: Optional[str] = None
        self.incoming_models: Dict[int, Tuple[str, List[float]]] = {}

        # State
        self.is_training: bool = False
        self.train_done: bool = False
