"""Algorithm 1; hash ordering matches the authors' released peer sampler.

The simulator assigns reproducible byte identities instead of IPv8 public keys.
Bandwidth ties retain the hash order, as in the authors' stable bandwidth sort.
"""
import hashlib
from functools import lru_cache


def ordered_peers(round_nr, peer_ids):
    return sorted(sorted(peer_ids),
                  key=lambda peer: hashlib.md5(b"%s-%d" % (peer, round_nr)).digest())


@lru_cache(maxsize=256)
def sample(round_nr, peer_ids, sample_size):
    indices = {identity: index for index, identity in enumerate(peer_ids)}
    return tuple(indices[peer] for peer in ordered_peers(round_nr, peer_ids)[:sample_size])
