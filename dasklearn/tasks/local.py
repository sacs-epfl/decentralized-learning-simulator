"""Serial execution of the real simulator DAG for small reproducible experiments."""
import copy
import heapq
import time
import torch
from dasklearn import functions


def execute_local(simulation, retain_results=True, progress=None):
    functions.dataset = None
    functions.model_trainers = None
    functions.evaluator = None
    torch.set_num_threads(simulation.settings.torch_threads)
    tasks = simulation.workflow_dag.tasks
    pending = {name: len(task.inputs) for name, task in tasks.items()}
    # Execute in simulated-time order to keep the live model frontier small.
    order = {name: idx for idx, name in enumerate(tasks)}
    ready = [(task.data.get("time", 0), order[task.name], task.name)
             for task in tasks.values() if not task.inputs]
    heapq.heapify(ready)
    results = {}
    remaining = {name: len(task.outputs) for name, task in tasks.items()}
    completed = 0
    evaluation_cache = {}
    timings = {}

    def model_key(ref):
        # V2 reconstructions of a round have identical chunk inputs. These
        # references identify exact DAG values, never approximate model hashes.
        source = tasks.get(ref[0]) if isinstance(ref, tuple) else None
        if source and source.func == "reconstruct_from_chunks":
            return (source.func, repr(source.data["chunks"]))
        return ref

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
        _, _, name = heapq.heappop(ready)
        task = tasks[name]
        data = resolve(task.data)
        started = time.perf_counter()
        if task.func == "test" and not retain_results:
            key = model_key(task.data["model"])
            data["cached_metrics"] = evaluation_cache.get(key)
            results[name] = functions.test(simulation.settings, data)
            evaluation_cache[key] = functions.last_test_metrics
        else:
            results[name] = getattr(functions, task.func)(simulation.settings, data)
        timings[task.func] = timings.get(task.func, 0) + time.perf_counter() - started
        completed += 1
        del data
        for predecessor, _ in task.inputs:
            remaining[predecessor.name] -= 1
            if not retain_results and remaining[predecessor.name] == 0:
                results.pop(predecessor.name, None)
        if not retain_results and remaining[name] == 0:
            results.pop(name, None)
        for successor, _ in task.outputs:
            pending[successor.name] -= 1
            if pending[successor.name] == 0:
                heapq.heappush(ready, (successor.data.get("time", 0), order[successor.name], successor.name))
        if progress and (completed % 500 == 0 or completed == len(tasks)):
            progress({"completed": completed, "total": len(tasks), "live_results": len(results), "timings_s": timings})
    if completed != len(tasks):
        raise RuntimeError("DAG contains a cycle or unresolved inputs")
    return results
