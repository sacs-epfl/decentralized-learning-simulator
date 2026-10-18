"""Serial execution of the real simulator DAG for small reproducible experiments."""
import copy
from collections import Counter, OrderedDict
import heapq
import time
import torch
from dasklearn import functions


def execute_local(simulation, retain_results=True, progress=None, share_readonly_inputs=True,
                  reconstruction_cache_bytes=64 * 1024 * 1024):
    if reconstruction_cache_bytes < 0:
        raise ValueError("reconstruction_cache_bytes must be nonnegative")
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

    def reconstruction_key(task):
        return (task.func, repr(task.data["chunks"]))

    # Only exact DAG references are interchangeable. Retained results must stay
    # independently mutable by callers, so memoization is streaming-only.
    duplicate_counts = Counter(reconstruction_key(task) for task in tasks.values()
        if task.func == "reconstruct_from_chunks") if not retain_results and reconstruction_cache_bytes else {}
    duplicate_counts = {key: count for key, count in duplicate_counts.items() if count > 1}
    reconstruction_cache = OrderedDict()
    cached_bytes = cache_peak_bytes = cache_hits = 0

    def drop_cached(key):
        nonlocal cached_bytes
        _, size = reconstruction_cache.pop(key)
        cached_bytes -= size

    def model_key(ref):
        # V2 reconstructions of a round have identical chunk inputs. These
        # references identify exact DAG values, never approximate model hashes.
        source = tasks.get(ref[0]) if isinstance(ref, tuple) else None
        if source and source.func == "reconstruct_from_chunks":
            return reconstruction_key(source)
        return ref

    # These built-ins read their inputs or make their own private model copy.
    # Unknown functions retain defensive edge copies. Test/evaluation moves the
    # input model between devices and changes its mode, so it also needs a copy.
    readonly = {"train", "compute_gradient", "gradient_update", "aggregate", "chunk",
                "aggregate_chunk", "reconstruct_from_chunks", "weighted_reconstruct_from_chunks"}

    def resolve(value, share=False):
        if isinstance(value, tuple) and len(value) == 2 and value[0] in tasks:
            # Training may mutate its input model. Each DAG edge has value
            # semantics, independent of execution order or number of workers.
            item = results[value[0]][value[1]]
            if share:
                return item
            # deepcopy(view) copies the whole backing storage, potentially a
            # full ResNet for every individual chunk. Clone only the slice.
            return item.clone() if isinstance(item, torch.Tensor) else copy.deepcopy(item)
        if isinstance(value, dict):
            return {key: resolve(item, share) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item, share) for item in value]
        return value

    while ready:
        _, _, name = heapq.heappop(ready)
        task = tasks[name]
        started = time.perf_counter()
        reconstruction = reconstruction_key(task) if task.func == "reconstruct_from_chunks" else None
        if reconstruction in reconstruction_cache:
            results[name] = reconstruction_cache[reconstruction][0]
            reconstruction_cache.move_to_end(reconstruction)
            cache_hits += 1
        else:
            evaluation_key = None
            metrics = None
            if task.func == "test" and not retain_results:
                evaluation_key = (model_key(task.data["model"]), task.data.get("evaluation_examples", simulation.settings.evaluation_examples))
                metrics = evaluation_cache.get(evaluation_key)
            discard_test_output = task.func == "test" and not retain_results and not task.outputs
            # Cached sink tests only write metrics; they never touch the model.
            share = (share_readonly_inputs and task.func in readonly) or (discard_test_output and metrics is not None)
            data = resolve(task.data, share)
            started = time.perf_counter()
            if evaluation_key is not None:
                data["cached_metrics"] = metrics
                data["return_model"] = not discard_test_output
            results[name] = getattr(functions, task.func)(simulation.settings, data)
            if evaluation_key is not None:
                evaluation_cache[evaluation_key] = functions.last_test_metrics
            del data
            if duplicate_counts.get(reconstruction, 0) > 1:
                size = sum(value.numel() * value.element_size() for value in results[name][0].state_dict().values())
                if size <= reconstruction_cache_bytes:
                    while reconstruction_cache and (cached_bytes + size > reconstruction_cache_bytes or len(reconstruction_cache) >= 4):
                        drop_cached(next(iter(reconstruction_cache)))
                    reconstruction_cache[reconstruction] = (results[name], size)
                    cached_bytes += size
                    cache_peak_bytes = max(cache_peak_bytes, cached_bytes)
        if reconstruction in duplicate_counts:
            duplicate_counts[reconstruction] -= 1
            if duplicate_counts[reconstruction] == 0:
                del duplicate_counts[reconstruction]
                if reconstruction in reconstruction_cache:
                    drop_cached(reconstruction)
        timings[task.func] = timings.get(task.func, 0) + time.perf_counter() - started
        completed += 1
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
            progress({"completed": completed, "total": len(tasks), "live_results": len(results), "timings_s": timings,
                      "reconstruction_cache_hits": cache_hits, "reconstruction_cache_bytes": cached_bytes,
                      "reconstruction_cache_peak_bytes": cache_peak_bytes})
    if completed != len(tasks):
        raise RuntimeError("DAG contains a cycle or unresolved inputs")
    return results
