"""Teleportation, ICLR 2025, Algorithm 1, with local-SGD blocks.

Membership/token assignment is a deterministic oracle, as is sampling in V2.
Every model transfer uses the shared bandwidth/latency scheduler.
"""
import random
from dasklearn.simulation.simulation import Simulation
from dasklearn.simulation.conflux.sample_manager import SampleManager
from dasklearn.simulation.teleportation.client import TeleportationClient
from dasklearn.simulation.events import START_ROUND, FINISH_TRAIN


class TeleportationSimulation(Simulation):
    CLIENT_CLASS = TeleportationClient

    def __init__(self, settings):
        if settings.churn != "none":
            raise ValueError("Teleportation currently requires stable membership")
        if not 1 <= settings.sample_size <= settings.participants:
            raise ValueError("Teleportation sample_size must be in [1, participants]")
        if settings.topology not in {"ring", "hypercube", "complete"}:
            raise ValueError("Unknown Teleportation topology")
        if settings.topology == "hypercube" and settings.sample_size & (settings.sample_size - 1):
            raise ValueError("Hypercube token count must be a power of two")
        if not settings.shuffle_tokens and settings.sample_size != settings.participants:
            raise ValueError("Fixed-token D-Local-SGD requires all participants")
        if settings.learning.momentum:
            raise ValueError("Teleportation momentum transport is not implemented; use momentum=0")
        if settings.test_method == "global":
            raise ValueError("Use snapshot or individual evaluation")
        super().__init__(settings)
        self.register_event_callback(START_ROUND, "start_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")

    def sample(self, round_nr):
        if not self.settings.shuffle_tokens:
            return list(range(self.settings.participants))
        peers = SampleManager.get_sample(round_nr, list(range(self.settings.participants)),
                                         self.settings.sample_size, seed=self.settings.seed)
        # Alg. 1 randomly assigns tokens; sorted peer IDs are biased.
        random.Random(self.settings.seed + 2147483647 * round_nr).shuffle(peers)
        return peers

    def neighbors(self, token):
        size = self.settings.sample_size
        if self.settings.topology == "ring":
            return sorted({(token - 1) % size, (token + 1) % size} - {token})
        if self.settings.topology == "complete":
            return [i for i in range(size) if i != token]
        return sorted(token ^ (1 << bit) for bit in range(size.bit_length() - 1))
