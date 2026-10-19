"""Explicit DAG references; ordinary tuples remain application data."""
from typing import NamedTuple


class TaskRef(NamedTuple):
    task: str
    output: int = 0


def as_ref(value):
    """Validate an explicit reference or optional absent state."""
    if value is None:
        return None
    if (isinstance(value, TaskRef) and
            isinstance(value[0], str) and type(value[1]) is int and value[1] >= 0):
        return value
    raise ValueError(f"Invalid task reference: {value!r}")


def is_reference(value):
    return isinstance(value, TaskRef)


def references(value):
    if is_reference(value):
        yield as_ref(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from references(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from references(item)


def map_references(value, resolve):
    if is_reference(value):
        return resolve(as_ref(value))
    if isinstance(value, dict):
        return {key: map_references(item, resolve) for key, item in value.items()}
    if isinstance(value, list):
        return [map_references(item, resolve) for item in value]
    if isinstance(value, tuple):
        return tuple(map_references(item, resolve) for item in value)
    return value
