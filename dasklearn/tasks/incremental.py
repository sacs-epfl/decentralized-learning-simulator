"""Conservative, resumable execution for protocols that consume real results.

Values stay pinned until the session is closed: a graph under construction does
not yet know its final consumers. The existing bounded executor remains the fast
path for static DAGs.
"""
import copy
import asyncio
from contextlib import contextmanager

import torch

from dasklearn import functions
from dasklearn.tasks.operations import execute_operation
from dasklearn.tasks.references import map_references, references


def execution_order(dag, targets, completed):
    """Topological dependency closure, without recursive Python stack growth."""
    visiting, visited, order = set(), set(completed), []
    for target in targets:
        stack = [(target, False)]
        while stack:
            name, expanded = stack.pop()
            if name in visited:
                continue
            if name not in dag.tasks:
                raise ValueError(f"Unknown task {name!r}")
            if expanded:
                visiting.remove(name)
                visited.add(name)
                order.append(name)
                continue
            if name in visiting:
                raise ValueError("Cycle in computation dependencies")
            visiting.add(name)
            stack.append((name, True))
            stack.extend((task.name, False) for task, _ in reversed(dag.tasks[name].inputs))
    return order


class LocalExecutionSession:
    def __init__(self, simulation):
        self.simulation = simulation
        self.results = {}
        self.state = dict(dataset=None, model_trainers=None, evaluator=None, last_test_metrics=None)
        self.closed = False

    @contextmanager
    def runtime(self):
        previous = {key: getattr(functions, key) for key in self.state}
        threads = torch.get_num_threads()
        torch.set_num_threads(self.simulation.settings.torch_threads)
        for key, value in self.state.items():
            setattr(functions, key, value)
        try:
            yield
        finally:
            self.state = {key: getattr(functions, key) for key in self.state}
            for key, value in previous.items():
                setattr(functions, key, value)
            torch.set_num_threads(threads)

    def _data(self, task):
        def output(ref):
            return copy.deepcopy(self.results[ref.task][ref.output])
        return map_references(task.data, output)

    def resolve(self, value, *, tasks=()):
        if self.closed:
            raise RuntimeError("Execution session is closed")
        dag = self.simulation.workflow_dag
        targets = list(tasks) + [ref.task for ref in references(value)]
        with self.runtime():
            for name in execution_order(dag, targets, self.results):
                task = dag.tasks[name]
                self.results[name] = execute_operation(task.func, self.simulation.settings,
                                                      self._data(task), task.operation)
        return map_references(value, lambda ref: copy.deepcopy(self.results[ref.task][ref.output]))

    def execute_all(self):
        if self.closed:
            raise RuntimeError("Execution session is closed")
        # Include operations with zero outputs without inventing an output reference.
        dag = self.simulation.workflow_dag
        with self.runtime():
            for name in execution_order(dag, list(dag.tasks), self.results):
                task = dag.tasks[name]
                self.results[name] = execute_operation(task.func, self.simulation.settings,
                                                      self._data(task), task.operation)

    def close(self):
        self.results.clear()
        self.state = dict.fromkeys(self.state)
        self.closed = True


class RemoteExecutionSession(LocalExecutionSession):
    """Coordinator-owned values, persistent broker workers, one ready task per RPC.

    Every RPC is acknowledged before the next task is dispatched. Consequently
    a temporary sink never terminates a run, and graph extension needs no global
    broker graph replacement. This deliberately favors correctness over throughput.
    """
    def __init__(self, simulation):
        super().__init__(simulation)
        self.pending = {}
        self.configured = False

    async def configure(self):
        if self.configured:
            return
        import pickle
        sim = self.simulation
        await sim.brokers_available_future
        sim.distribute_brokers()
        sim.communication.send_message_to_all_brokers(pickle.dumps({
            "type": "config", "incremental": True, "brokers": sim.broker_addresses,
            "brokers_to_clients": sim.brokers_to_clients,
            "settings": sim.settings.to_dict(), "settings_class": type(sim.settings).__name__,
            "dag": [],
        }))
        self.configured = True

    async def _execute(self, targets):
        import pickle
        await self.configure()
        sim = self.simulation
        for name in execution_order(sim.workflow_dag, targets, self.results):
            task = sim.workflow_dag.tasks[name]
            future = asyncio.get_running_loop().create_future()
            self.pending[name] = future
            peer = task.data.get("peer", 0)
            # Client assignment is stable even when new participant tasks appear.
            brokers = list(sim.broker_addresses)
            broker = sim.clients_to_brokers.get(peer, brokers[peer % len(brokers)])
            sim.communication.send_message_to_broker(broker, pickle.dumps({
                "type": "execute", "task": name, "func": task.func,
                "data": self._data(task), "operation": task.operation,
            }))
            try:
                self.results[name] = await future
            finally:
                self.pending.pop(name, None)

    async def resolve(self, value, *, tasks=()):
        if self.closed:
            raise RuntimeError("Execution session is closed")
        await self._execute(list(tasks) + [ref.task for ref in references(value)])
        return map_references(value, lambda ref: copy.deepcopy(self.results[ref.task][ref.output]))

    async def execute_all(self):
        if self.closed:
            raise RuntimeError("Execution session is closed")
        await self._execute(list(self.simulation.workflow_dag.tasks))

    def on_result(self, message):
        future = self.pending.get(message["task"])
        if future is None or future.done():
            if message.get("error"):
                self.fail(message["error"])
            return
        if message.get("error"):
            future.set_exception(RuntimeError(message["error"]))
        else:
            future.set_result(message["result"])

    def fail(self, message):
        for future in self.pending.values():
            if not future.done():
                future.set_exception(RuntimeError(message))
