import asyncio
import pickle
import random

from typing import Optional, Set

import zmq
import zmq.asyncio

from dasklearn.communication import Communication
from dasklearn.tasks.dag import WorkflowDAG
from dasklearn.tasks.task import Task
from dasklearn.functions import *
from dasklearn.util.logging import setup_logging


class Worker:

    def __init__(self, args):
        self.args = args
        self.workers_to_clients: Dict = {}
        self.clients_to_workers: Dict = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings: Optional[SessionSettings] = None
        self.dag: Optional[WorkflowDAG] = None
        self.task_queue = asyncio.Queue()
        self.queue_thread_task = None
        self.coordinator_socket = None
        self.data_dir: Optional[str] = None

        self.communication: Optional[Communication] = None

        self.worker_addresses: Dict[str, str] = {}

        self.identity: str = "worker_%s" % ''.join(random.choice('0123456789abcdef') for _ in range(6))
        self.logger.info("Worker %s initialized", self.identity)

    def handle_task_result(self, task: Task, res):
        """
        We received a task result - either locally or from another worker. Handle it.
        """
        for next_task in task.outputs:
            if next_task.data["peer"] in self.workers_to_clients[self.identity]:
                next_task.set_data(task.name, res)
                if next_task.has_all_inputs():
                    self.task_queue.put_nowait(next_task)

    def shutdown_everyone(self):
        self.logger.error("Will send shutdown signal to all nodes")
        msg = pickle.dumps({"type": "shutdown"})
        self.communication.send_message_to_all_workers(msg)
        self.communication.send_message_to_coordinator(msg)
        asyncio.get_event_loop().call_later(2, asyncio.get_event_loop().stop)

    async def queue_thread(self, queue):
        self.logger.info("Starting queue thread")
        while True:
            # Get a "work item" out of the queue.
            task = await queue.get()
            try:
                res = self.execute_task(task)

                self.logger.info("Task %s completed", task.name)

                # If this is a sink task, inform the coordinator about the result
                if not task.outputs:
                    # This is a sink task with no further outputs - send the result back to the coordinator
                    msg = pickle.dumps({"type": "result", "task": task.name, "result": res})
                    self.communication.send_message_to_coordinator(msg)
                else:
                    # Some worker needs this result - all workers we need to inform about this result
                    workers_to_inform: Set[str] = set()
                    for next_task in task.outputs:
                        peer_next_task = next_task.data["peer"]
                        workers_to_inform.add(self.clients_to_workers[peer_next_task])

                    for worker_to_inform in workers_to_inform:
                        if worker_to_inform == self.identity:
                            self.handle_task_result(task, res)
                        else:
                            msg = pickle.dumps({"type": "result", "task": task.name, "result": res})
                            self.communication.send_message_to_worker(worker_to_inform, msg)

            except Exception as exc:
                self.logger.exception(exc)
                self.shutdown_everyone()
                break

            # Clean up the memory of the completed task.
            task.clear_data()

            queue.task_done()

        self.logger.warning("Queue thread terminated")

    def start(self):
        self.communication = Communication(self.identity, self.args.port, self.on_message, is_worker=True)
        self.communication.start()
        self.communication.connect_to_coordinator(self.args.coordinator)

    def execute_task(self, task: Task):
        self.logger.info("Worker executing task %s", task.name)
        if task.func not in globals():
            raise RuntimeError("Task %s not found!" % task.func)
        f = globals()[task.func]

        res = f(self.settings, task.data)
        return res

    def connect_to_workers(self):
        self.logger.info("Connecting to workers: %s", self.worker_addresses)
        for worker_name, worker_address in self.worker_addresses.items():
            if worker_name == self.identity:
                continue

            self.communication.connect_to(worker_name, worker_address)

    def on_message(self, identity: str, msg: Dict):
        if identity == "coordinator" and msg["type"] == "config":  # Configuration received from the coordinator
            self.logger.info("Received configuration from the coordinator")
            self.worker_addresses = msg["workers"]
            self.workers_to_clients = msg["workers_to_clients"]

            # Build the reverse map
            for worker, clients in self.workers_to_clients.items():
                for client in clients:
                    self.clients_to_workers[client] = worker

            self.settings = SessionSettings.from_dict(msg["settings"])
            setup_logging(self.settings.data_dir, "%s.log" % self.identity)
            self.dag = WorkflowDAG.unserialize(msg["dag"])
            self.connect_to_workers()
            self.queue_thread_task = asyncio.create_task(self.queue_thread(self.task_queue))
        elif identity == "coordinator" and msg["type"] == "tasks":
            for task_name in msg["tasks"]:
                # Enqueue tasks
                task = self.dag.tasks[task_name]
                self.task_queue.put_nowait(task)
        elif msg["type"] == "shutdown":
            self.logger.info("Received shutdown signal - stopping")
            asyncio.get_event_loop().stop()
        elif msg["type"] == "result":
            self.logger.info("Received task %s result from %s", msg["task"], identity)
            task = self.dag.tasks[msg["task"]]
            self.handle_task_result(task, msg["result"])
        else:
            raise RuntimeError("Unknown message with type %s" % msg["type"])
