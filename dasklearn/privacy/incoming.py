"""Passive capture and loss attacks on completed, participant-visible deliveries."""
import copy
from dataclasses import dataclass, field

import torch

from dasklearn.models import create_model_structure
from dasklearn.privacy.membership import add_model_mia_task
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.tasks.operations import operation_contract
from dasklearn.tasks.references import TaskRef
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS


@dataclass
class IncomingObservation:
    attacker: int
    sender: int
    round_nr: int
    first_delivery_us: int
    last_delivery_us: int
    baseline: TaskRef | None
    full_model: TaskRef | None = None
    chunks: dict[int, TaskRef] = field(default_factory=dict)
    deliveries: int = 0


def complete_partial_model(baseline, chunks, chunk_count):
    """Replace observed slices in an attacker-owned model, preserving other slices.

    Missing target values are never inputs to this function. Chunks must use the
    public equal-sized layout of ChunkManager.chunk_model (last has remainder).
    Neither baseline nor payload tensors are mutated.
    """
    if not chunks or any(type(index) is not int or not 0 <= index < chunk_count for index in chunks):
        raise ValueError("Received chunk indices must be nonempty and within the public layout")
    result = copy.deepcopy(baseline)
    pieces = [piece.clone() for piece in ChunkManager.chunk_model(result, chunk_count)]
    for index, value in chunks.items():
        if not isinstance(value, torch.Tensor) or value.shape != pieces[index].shape:
            raise ValueError("Received chunk shape does not match the public layout")
        pieces[index] = value.detach().to(pieces[index]).clone()
    return ChunkManager.weighted_reconstruct_model(pieces, result)


def complete_incoming_operation(settings, params):
    return [complete_partial_model(params["baseline"], params["chunks"], params["chunk_count"])]


class IncomingModelObserver:
    """Attach before simulation; audit only deliveries actually seen by recipients.

    Source IDs, rounds and chunk indices are visible in this simulator's messages.
    The receiver's latest completed local training/published state is frozen at
    the first fragment of each (receiver, sender, round). No fragments from other
    rounds, other recipients, or future deliveries fill gaps.
    """
    def __init__(self, *, attackers, rounds=None, chunk_count=None):
        if chunk_count is not None and (type(chunk_count) is not int or chunk_count < 1):
            raise ValueError("chunk_count must be a positive integer or None for full models")
        self.attackers = frozenset(attackers)
        self.rounds = None if rounds is None else frozenset(rounds)
        self.chunk_count = chunk_count
        self.local_models = {}
        self.observations = {}

    def on_compute_complete(self, client, event):
        # Explicitly model-valued outputs; gradient-only/compressed outputs must
        # not accidentally become a complete-model background.
        task = client.simulator.workflow_dag.tasks[event.data["task"]]
        if task.func in {"train", "qgm_train"} and client.index in self.attackers:
            self.local_models[client.index] = event.data["outputs"][0]

    def on_model_published(self, client, round_nr, model):
        if client.index in self.attackers:
            self.local_models[client.index] = model

    def on_payload(self, client, event):
        if client.index not in self.attackers:
            return
        sender = event.data["from"]
        if sender == client.index:
            return  # A local self-delivery provides no new other-client model.
        metadata = event.data["metadata"]
        round_nr = metadata.get("round")
        if round_nr is None or (self.rounds is not None and round_nr not in self.rounds):
            return
        payload = event.data["payload"]
        if not isinstance(payload, TaskRef):
            raise ValueError("Incoming MIA requires explicit full-model or tensor-chunk references")
        if self.chunk_count is not None:
            index = metadata.get("chunk")
            if type(index) is not int or not 0 <= index < self.chunk_count:
                raise ValueError("Chunked incoming MIA needs a visible valid chunk index")
            if payload.output != index:
                raise ValueError("Chunk metadata and payload reference disagree")
        key = (client.index, sender, round_nr)
        if key not in self.observations:
            baseline = self.local_models.get(client.index)
            if baseline is None:
                own_model = client.simulator.latest_models.get(client.index)
                baseline = own_model[1] if own_model else None
            self.observations[key] = IncomingObservation(client.index, sender, round_nr,
                event.time, event.time, baseline)
        observation = self.observations[key]
        observation.last_delivery_us = event.time
        observation.deliveries += 1
        if self.chunk_count is None:
            if observation.full_model not in (None, payload):
                raise ValueError("Multiple distinct full models share the same sender/round")
            observation.full_model = payload
        else:
            if observation.chunks and next(iter(observation.chunks.values())).task != payload.task:
                raise ValueError("Cannot combine chunks from distinct source model versions")
            observation.chunks[index] = payload

    def attach_audits(self, simulation, *, max_examples=128, target_fpr=0.1,
                      cohort_policy="class_matched"):
        """After simulation, add audit dependencies without perturbing the protocol."""
        initial = None
        total = sum(value.numel() for value in create_model_structure(
            simulation.settings.dataset, simulation.settings.model, settings=simulation.settings).state_dict().values())
        audits = []
        for key, observation in sorted(self.observations.items()):
            attacker, sender, round_nr = key
            time_us = observation.last_delivery_us
            fields = {"first_delivery_time": observation.first_delivery_us / MICROSECONDS,
                      "last_delivery_time": time_us / MICROSECONDS,
                      "delivery_count": observation.deliveries,
                      "total_state_elements": total}
            if self.chunk_count is None:
                model = observation.full_model
                fields.update(received_chunks=1, chunk_count=1, visible_state_elements=total,
                              visible_state_fraction=1.0, completion_strategy="none")
                view = "incoming_full"
            else:
                baseline = observation.baseline
                if baseline is None:
                    # Shared initialization is public and exactly reproducible.
                    # Create it only after simulation, outside the protocol graph.
                    if initial is None:
                        name = Task.generate_name("mia_public_initialization")
                        initial = simulation.workflow_dag.add(Task(name, "initialize",
                            {"peer": 0, "round": 0, "time": 0},
                            operation=operation_contract("initialize"), available_at=0))
                    baseline = TaskRef(initial.name, 0)
                name = Task.generate_name("mia_complete_incoming")
                data = {"baseline": baseline, "chunks": dict(observation.chunks),
                        "chunk_count": self.chunk_count, "peer": attacker, "round": round_nr,
                        "time": time_us}
                task = simulation.workflow_dag.add(Task(name, "complete_incoming", data,
                    operation=operation_contract("complete_incoming"), available_at=time_us))
                model = TaskRef(task.name, 0)
                size = total // self.chunk_count
                visible = sum(size + (total % self.chunk_count if index == self.chunk_count - 1 else 0)
                              for index in observation.chunks)
                fields.update(received_chunks=len(observation.chunks), chunk_count=self.chunk_count,
                              visible_state_elements=visible, visible_state_fraction=visible / total,
                              completion_strategy="receiver_model_at_first_delivery")
                view = "incoming_partial"
            audits.append(add_model_mia_task(simulation, attacker, model, round_nr=round_nr,
                time_us=time_us, targets=[sender], max_examples=max_examples, target_fpr=target_fpr,
                attacker_view=view, observation=fields, cohort_policy=cohort_policy))
        return audits
