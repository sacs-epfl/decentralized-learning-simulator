"""
Code related to the PushSum learning algorithm, which combines sample-based training with
pushsum-based model synchronization and slot-based bandwidth allocation.
"""
from enum import Enum


class NodeMembershipChange(Enum):
    JOIN = 0
    LEAVE = 1
