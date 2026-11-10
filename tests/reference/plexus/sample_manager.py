import hashlib
from typing import List, Dict

from dlsim.core.peer_manager import PeerManager


class SampleManager:
    """
    The SampleManager class is responsible for deriving samples and determine who the aggregators in a particular
    sample are.
    """

    def __init__(self, peer_manager: PeerManager, sample_size: int, num_aggregators: int):
        self.peer_manager: PeerManager = peer_manager
        self.sample_size = sample_size
        self.num_aggregators = num_aggregators
        self.sample_cache: Dict[int, List[bytes]] = {}

    def get_ordered_sample_list(self, round: int, peers: List[bytes]) -> List[bytes]:
        peers = sorted(peers)
        hashes = []
        for peer_id in peers:
            h = hashlib.md5(b"%s-%d" % (peer_id, round))
            hashes.append((peer_id, h.digest()))
        hashes = sorted(hashes, key=lambda t: t[1])
        return [t[0] for t in hashes]
