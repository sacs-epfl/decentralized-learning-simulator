import hashlib
from typing import List


class SampleManager:
    """
    The SampleManager class is responsible for deriving samples.
    """

    @staticmethod
    def get_sample(round: int, eligible_peers: List[int], sample_size: int) -> List[int]:
        hashes = []
        for peer_id in eligible_peers:
            h = hashlib.md5(b"%d-%d" % (peer_id, round))
            hashes.append((peer_id, h.digest()))
        hashes = sorted(hashes, key=lambda t: t[1])
        return sorted([t[0] for t in hashes[:sample_size]])
