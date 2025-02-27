"""
Contains the code related to the Conflux code.
"""
from enum import Enum


class NodeMembershipChange(Enum):
    JOIN = 0
    LEAVE = 1
