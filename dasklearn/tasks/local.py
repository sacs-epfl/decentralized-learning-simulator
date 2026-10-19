"""Serial execution of the real simulator DAG for small reproducible experiments."""
import copy
from collections import Counter, OrderedDict
from contextlib import ExitStack
import heapq
import time
import torch
from dasklearn import functions
from dasklearn.tasks.references import map_references
from dasklearn.tasks.operations import execute_operation


def execute_local(simulation, retain_results=True, progress=None, share_readonly_inputs=True,
                  reconstruction_cache_bytes=64 * 1024 * 1024, training_backend=None):
    if getattr(simulation, "execution_session", None) is not None:
        session = simulation.execution_session
        session.execute_all()
        return dict(session.results) if retain_results else {}
    with ExitStack() as resources:
        return _execute_local(simulation,retain_results,progress,share_readonly_inputs,
                              reconstruction_cache_bytes,training_backend,resources)


def _execute_local(simulation,retain_results,progress,share_readonly_inputs,
                   reconstruction_cache_bytes,training_backend,resources):
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
    readonly = {"train", "compute_gradient", "gradient_update", "aggregate", "chunk", "choco_prepare", "choco_receive", "qgm_train", "qgm_finish",
                "aggregate_chunk", "reconstruct_from_chunks", "weighted_reconstruct_from_chunks"}

    def resolve(value, share=False):
        def output(ref):
            # Training may mutate its input model. Each DAG edge has value
            # semantics, independent of execution order or number of workers.
            item = results[ref.task][ref.output]
            if share:
                return item
            # deepcopy(view) copies the whole backing storage, potentially a
            # full ResNet for every individual chunk. Clone only the slice.
            return item.clone() if isinstance(item, torch.Tensor) else copy.deepcopy(item)
        return map_references(value, output)

    settings = simulation.settings
    if training_backend is None and torch.device(getattr(settings,"torch_device_name","cpu")).type == "cuda" and (
            settings.gpu_residency or settings.training_batching):
        from dasklearn.tasks.training import TrainingExecutor
        training_backend = TrainingExecutor(settings)
    if training_backend:resources.callback(training_backend.close)
    prefetched, prefetch_times = {}, {}
    prefetch_peak = 0
    input_counts = Counter(model_key(t.data["model"]) for t in tasks.values() if t.func == "train") if training_backend else {}

    while ready:
        _, _, name = heapq.heappop(ready)
        task = tasks[name]
        if training_backend and task.func == "train" and name not in prefetched and training_backend.can_train(task.data):
            selected = [task]
            # Prefetch only already-ready independent tasks. Their outputs are
            # published at the original heap position, preserving bookkeeping,
            # snapshot references and simulated protocol timestamps exactly.
            if training_backend.can_batch(task.data):
                scanned = []
                slots = max(1,training_backend.width-len(prefetched))
                while ready and len(scanned) < 128 and len(selected) < slots:
                    entry = heapq.heappop(ready);scanned.append(entry)
                    candidate = tasks[entry[2]]
                    if candidate.func == "train" and candidate.name not in prefetched and training_backend.can_batch(candidate.data):
                        selected.append(candidate)
                for entry in scanned:heapq.heappush(ready,entry)
            keys = [model_key(t.data["model"]) for t in selected]
            resolved = [resolve(t.data,share=True) for t in selected]
            begin = time.perf_counter()
            values = training_backend.train_many(resolved,[k if input_counts[k]>1 else None for k in keys])
            elapsed = (time.perf_counter()-begin)/len(selected)
            for selected_task,key,value in zip(selected,keys,values):
                prefetched[selected_task.name]=[value];prefetch_times[selected_task.name]=elapsed
                input_counts[key]-=1
                if input_counts[key]==0:training_backend.drop_input(key)
            prefetch_peak=max(prefetch_peak,len(prefetched))
            del resolved,values
        started = time.perf_counter()
        reconstruction = reconstruction_key(task) if task.func == "reconstruct_from_chunks" else None
        if name in prefetched:
            results[name] = prefetched.pop(name)
        elif reconstruction in reconstruction_cache:
            results[name] = reconstruction_cache[reconstruction][0]
            reconstruction_cache.move_to_end(reconstruction)
            cache_hits += 1
        else:
            evaluation_key = None
            metrics = None
            if task.func == "test" and not retain_results:
                evaluation_key = (model_key(task.data["model"]), task.data.get("evaluation_examples", simulation.settings.evaluation_examples),
                                  task.data.get("evaluation_split", simulation.settings.evaluation_split))
                metrics = evaluation_cache.get(evaluation_key)
            discard_test_output = task.func == "test" and not retain_results and not task.outputs
            # Cached sink tests only write metrics; they never touch the model.
            share = (share_readonly_inputs and task.func in readonly) or (discard_test_output and metrics is not None)
            data = resolve(task.data, share)
            started = time.perf_counter()
            if evaluation_key is not None:
                data["cached_metrics"] = metrics
                data["return_model"] = not discard_test_output
            contract = task.operation
            if discard_test_output and contract:
                contract = dict(contract, outputs=0)
            results[name] = execute_operation(task.func, simulation.settings, data, contract)
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
        task_elapsed = prefetch_times.pop(name) if name in prefetch_times else time.perf_counter()-started
        timings[task.func] = timings.get(task.func, 0) + task_elapsed
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
                      "reconstruction_cache_peak_bytes": cache_peak_bytes,
                      "training_backend":training_backend.status() if training_backend else {"active":False}})
    simulation.execution_training_stats = training_backend.status() if training_backend else {
        "active":False,"reason":"Reference executor: non-CUDA device or both optimizations disabled"}
    simulation.execution_training_stats["prefetch_peak_models"]=prefetch_peak
    if completed != len(tasks):
        raise RuntimeError("DAG contains a cycle or unresolved inputs")
    return results
