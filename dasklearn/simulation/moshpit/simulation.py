"""Stable-membership specialization of Moshpit SGD, NeurIPS 2021.

Algorithm 1's rolling group keys become rotating coordinate axes of a seeded
grid. No global group barrier: readiness, reduce and gather are network events.
Default T=1 matches the paper's ImageNet averaging configuration; T=d gives
exact consensus on a full grid. DHT discovery/failure recovery are not modeled.
"""
import random

from dasklearn.simulation.simulation import Simulation
from dasklearn.simulation.events import START_ROUND, FINISH_TRAIN
from dasklearn.simulation.moshpit.client import MoshpitClient


def balanced_weights(bandwidths):
    """Solve Appendix G's minimax LP by monotone capacity search.

For m>2, feasibility at time T requires sum((T*b_i-1)/(m-2)) >= 1
and T >= max(1/b_i). Normalizing feasible capacities gives optimal weights.
"""
    size = len(bandwidths)
    if size < 1 or min(bandwidths) <= 0:
        raise ValueError("Positive bandwidths required")
    if size <= 2:
        return [1 / size] * size
    low, high = 1 / min(bandwidths), 2 / min(bandwidths)
    for _ in range(64):
        middle = (low + high) / 2
        if sum(max(0, (middle * b - 1) / (size - 2)) for b in bandwidths) >= 1:
            high = middle
        else:
            low = middle
    capacity = [max(0, (high * b - 1) / (size - 2)) for b in bandwidths]
    return [x / sum(capacity) for x in capacity]


class MoshpitSimulation(Simulation):
    CLIENT_CLASS = MoshpitClient

    def __init__(self, settings):
        if settings.churn != "none":
            raise ValueError("Moshpit baseline currently requires stable membership")
        if settings.group_size < 2 or settings.averaging_steps < 1:
            raise ValueError("group_size >= 2 and averaging_steps >= 1 required")
        if settings.learning.momentum:
            raise ValueError("Moshpit baseline uses plain local SGD; use momentum=0")
        if settings.test_method == "global":
            raise ValueError("Use snapshot or individual evaluation")
        super().__init__(settings)
        self.peers = list(range(settings.participants))
        random.Random(settings.seed).shuffle(self.peers)
        self.ranks = {peer: rank for rank, peer in enumerate(self.peers)}
        self.dimensions = 1
        while settings.group_size ** self.dimensions < settings.participants:
            self.dimensions += 1
        self.layouts = {}
        self.register_event_callback(START_ROUND, "start_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")

    def layout(self, peer, round_nr, phase):
        step = (round_nr - 1) * self.settings.averaging_steps + phase
        axis = step % self.dimensions
        rank = self.ranks[peer]
        stride = self.settings.group_size ** axis
        base = rank - (rank // stride % self.settings.group_size) * stride
        key = (axis, base)
        if key not in self.layouts:
            members = [self.peers[base + digit * stride] for digit in range(self.settings.group_size)
                       if base + digit * stride < len(self.peers)]
            weights = balanced_weights([self.clients[i].bw_scheduler.bw_limit for i in members]) if self.settings.load_balancing else [1 / len(members)] * len(members)
            self.layouts[key] = members, weights
        return self.layouts[key]
