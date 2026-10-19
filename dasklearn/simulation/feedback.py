"""Feedback observations and provenance for deterministic protocol replay.

Like saved workflow DAGs, journals use pickle and must come from trusted runs.
Replay supplies observations; it does not replace numerical task checkpoints.
"""
import copy
import hashlib
import inspect
import pickle
from importlib.util import find_spec
from pathlib import Path

from dasklearn.tasks.references import map_references


def provenance(simulation):
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted(root.rglob("*.py"))
    for cls in (type(simulation), simulation.CLIENT_CLASS):
        path = inspect.getsourcefile(cls)
        if path and Path(path) not in paths:
            paths.append(Path(path))
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    settings = simulation.settings.to_dict()
    for key in ("work_dir", "data_dir", "port", "brokers", "dry_run", "unit_testing",
                "from_dir", "profile", "log_level", "torch_device_name", "torch_threads",
                "gpu_residency", "training_batching", "training_batch_size", "gpu_state_cache_mb",
                "feedback_execution", "feedback_replay"):
        settings.pop(key, None)
    return {"source_hash": digest.hexdigest(), "settings": settings}


class FeedbackJournal:
    def __init__(self, simulation, replay=None):
        self.simulation = simulation
        self.provenance = provenance(simulation)
        self.records = []
        self.cursor = 0
        self.replay = replay is not None
        self.expected_trace = None
        self.implementation_hashes = {}
        if self.replay:
            with open(replay, "rb") as stream:
                saved = pickle.load(stream)
            if saved.get("version") != 1 or saved.get("provenance") != self.provenance:
                raise ValueError("Feedback replay source or settings do not match this run")
            self.records = saved["records"]
            self.expected_trace = saved["trace_hash"]

    def key(self, event):
        tasks = self.simulation.workflow_dag.tasks
        indices = {name: i for i, name in enumerate(tasks)}
        task = tasks[event.data["task"]]
        def normalize(ref):
            return (indices[ref.task], ref.output) if ref.task in indices else tuple(ref)
        payload = map_references(task.data, normalize)
        for candidate in tasks.values():
            if candidate.operation:
                module = candidate.operation['function'].split(':', 1)[0]
                if module in self.implementation_hashes:
                    continue
                spec = find_spec(module)
                path = Path(spec.origin) if spec and spec.origin else None
                if path and path.is_file():
                    self.implementation_hashes[module] = hashlib.sha256(path.read_bytes()).hexdigest()
        signature = hashlib.sha256(pickle.dumps((task.func, task.operation, payload,
                                                  sorted(self.implementation_hashes.items())))).hexdigest()
        return (event.time, event.client_id, event.data["callback"], indices[task.name], signature)

    def observation(self, event, values=None):
        key = self.key(event)
        if self.replay:
            if self.cursor >= len(self.records) or self.records[self.cursor]["key"] != key:
                raise ValueError("Feedback replay diverged from the recorded decision point")
            record = self.records[self.cursor]
        else:
            record = {"key": key, "values": copy.deepcopy(values)}
            self.records.append(record)
        self.cursor += 1
        return copy.deepcopy(record["values"])

    def save(self, path, trace_hash):
        if self.replay and self.cursor != len(self.records):
            raise ValueError("Feedback replay ended before consuming all observations")
        if self.replay and trace_hash != self.expected_trace:
            raise ValueError("Feedback replay produced a different event trace")
        with open(path, "wb") as stream:
            pickle.dump({"version": 1, "provenance": self.provenance,
                         "records": self.records, "trace_hash": trace_hash}, stream)
