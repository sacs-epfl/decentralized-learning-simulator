"""Importable operation contracts shared by coordinators and worker processes."""
from dataclasses import dataclass, asdict
from importlib import import_module
import copy


@dataclass(frozen=True)
class Operation:
    function: str
    outputs: int
    readonly_inputs: bool = False

    def __post_init__(self):
        if ":" not in self.function or type(self.outputs) is not int or self.outputs < 0:
            raise ValueError("An operation needs a module:function path and nonnegative output count")


READONLY = {"initialize", "train", "compute_gradient", "gradient_update", "aggregate", "chunk",
            "qgm_train", "qgm_finish", "choco_prepare", "choco_receive", "aggregate_chunk",
            "reconstruct_from_chunks", "weighted_reconstruct_from_chunks"}

OPERATIONS = {name: Operation(f"dasklearn.functions:{name}", outputs, name in READONLY) for name, outputs in {
    "initialize": 1, "train": 1, "compute_gradient": 1, "gradient_update": 1,
    "aggregate": 1, "test": 1, "evaluate": 1, "qgm_train": 3, "qgm_finish": 2,
    "choco_prepare": 2, "choco_receive": 1, "aggregate_chunk": 1,
    "demo_prepare": 2, "demo_update": 1,
    "reconstruct_from_chunks": 1, "weighted_reconstruct_from_chunks": 1,
}.items()}

OPERATIONS["membership_inference"] = Operation(
    "dasklearn.privacy.membership:evaluate_membership", 1, True)
OPERATIONS["complete_incoming"] = Operation(
    "dasklearn.privacy.incoming:complete_incoming_operation", 1, True)


def register_operation(name, function, outputs, *, readonly_inputs=False):
    """Register an importable implementation; its contract travels with each task."""
    operation = Operation(function, outputs, readonly_inputs)
    if name in OPERATIONS and OPERATIONS[name] != operation:
        raise ValueError(f"Operation {name!r} already has a different contract")
    OPERATIONS[name] = operation
    return operation


def operation_contract(name, data=None):
    if name == "chunk":
        count = (data or {}).get("n")
        return asdict(Operation("dasklearn.functions:chunk", count, True))
    try:
        return asdict(OPERATIONS[name])
    except KeyError as exc:
        raise ValueError(f"Register operation {name!r} before declaring it") from exc


def execute_operation(name, settings, data, contract=None, *, copy_inputs=False):
    path = contract["function"] if contract else f"dasklearn.functions:{name}"
    module, attribute = path.split(":", 1)
    function = getattr(import_module(module), attribute)
    if copy_inputs and not (contract and contract['readonly_inputs']):
        data = copy.deepcopy(data)
    result = function(settings, data)
    if not isinstance(result, (list, tuple)):
        raise ValueError(f"Operation {name!r} must return a list or tuple of outputs")
    if contract and len(result) != contract["outputs"]:
        raise ValueError(f"Operation {name!r} returned {len(result)} outputs; expected {contract['outputs']}")
    return result
