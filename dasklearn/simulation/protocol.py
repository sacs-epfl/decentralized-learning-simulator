"""Small opt-in helpers for event-driven participant implementations."""
import copy
import hashlib
import random
from dataclasses import dataclass
from types import MappingProxyType


class RoundInbox:
    """First message per expected sender; no implicit timeout or global barrier."""
    def __init__(self, expected):
        self.expected = frozenset(expected)
        self.messages = {}
        self.completed = False

    def add(self, sender, payload):
        if sender not in self.expected:
            raise ValueError(f"Unexpected sender {sender}")
        if self.completed or sender in self.messages:
            return False
        self.messages[sender] = payload
        return True

    def finish(self, local_ready=True):
        if self.completed or not local_ready or not self.expected.issubset(self.messages):
            return False
        self.completed = True
        return True


@dataclass
class Timer:
    cancelled: bool = False

    def cancel(self):
        self.cancelled = True


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze(item) for item in value)
    return value


class ParticipantContext:
    """Public local observations; not a security sandbox for Python protocols.

    Global inspection remains available through explicitly requested oracle
    services. Protocol-specific topology and coordination still live on the
    simulator; this context distinguishes local observations from those services.
    """
    def __init__(self, client):
        self._client = client
        self.index = client.index
        self.config = freeze(copy.deepcopy(client.simulator.settings.to_dict()))
        self._random = {}

    @property
    def now(self):
        return self._client.simulator.current_time

    @property
    def online(self):
        return self._client.online

    def rng(self, stream="protocol"):
        if not isinstance(stream, str):
            raise TypeError("Random stream names must be strings")
        if stream not in self._random:
            seed = f"{self.config['seed']}:{self.index}:{stream}".encode()
            self._random[stream] = random.Random(int.from_bytes(hashlib.sha256(seed).digest(), "big"))
        return self._random[stream]

    def oracle(self, name):
        try:
            service = self._client.simulator.oracle_services[name]
        except KeyError as exc:
            raise ValueError(f"Oracle service {name!r} was not configured") from exc
        return service(self.index)

    def timer(self, delay, callback, data=None):
        return self._client.set_timer(delay, callback, data)
