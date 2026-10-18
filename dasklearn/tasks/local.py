"""Serial execution of the real simulator DAG for small reproducible experiments."""
import copy
from collections import deque
import torch
from dasklearn import functions


def execute_local(simulation):
    functions.dataset = None
    functions.model_trainers = None
    functions.evaluator = None
    torch.set_num_threads(simulation.settings.torch_threads)
    tasks = simulation.workflow_dag.tasks
    pending = {name: len(task.inputs) for name, task in tasks.items()}
    ready = deque(task for task in tasks.values() if not task.inputs)
    results = {}

    def resolve(value):
        if isinstance(value, tuple) and len(value) == 2 and value[0] in tasks:
            # Training may mutate its input model. Each DAG edge has value
            # semantics, independent of execution order or number of workers.
            return copy.deepcopy(results[value[0]][value[1]])
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    while ready:
        task = ready.popleft()
        results[task.name] = getattr(functions, task.func)(simulation.settings, resolve(task.data))
        for successor, _ in task.outputs:
            pending[successor.name] -= 1
            if pending[successor.name] == 0:
                ready.append(successor)
    if len(results) != len(tasks):
        raise RuntimeError("DAG contains a cycle or unresolved inputs")
    return results
