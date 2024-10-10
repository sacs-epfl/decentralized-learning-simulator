import hashlib
from typing import List

import networkx as nx


class SampleManager:
    """
    The SampleManager class is responsible for deriving samples.
    """

    @staticmethod
    def get_sample(round: int, total_peers: int, sample_size: int) -> List[int]:
        peers = list(range(total_peers))
        hashes = []
        for peer_id in peers:
            h = hashlib.md5(b"%d-%d" % (peer_id, round))
            hashes.append((peer_id, h.digest()))
        hashes = sorted(hashes, key=lambda t: t[1])
        return [t[0] for t in hashes[:sample_size]]
    
    @staticmethod
    def get_neighbours_in_round(round: int, sample_size: int, k_in_sample: int, my_index: int, sample: List[int]) -> List[int]:
        G = nx.random_regular_graph(k_in_sample, sample_size, seed=round)
        return [sample[i] for i in G.neighbors(my_index)]
