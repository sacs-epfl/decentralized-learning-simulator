import asyncio
import threading
from asyncio import ensure_future

import psutil
import torch.multiprocessing as multiprocessing
import pickle
import random
from asyncio.subprocess import Process
from typing import List, Optional, Set, Tuple

from dasklearn.communication import Communication
from dasklearn.functions import *
from dasklearn.tasks.dag import WorkflowDAG
from dasklearn.tasks.task import Task
from dasklearn.util.logging import setup_logging
from dasklearn.worker import Worker

torch.multiprocessing.set_sharing_strategy('file_system')


def worker_proc(shared_queue, result_queue, index, settings):
    torch.set_num_threads(4)
    worker = Worker(shared_queue, result_queue, index, settings)
    worker.start()


class Broker:

    def __init__(self, args):
        self.args = args
        self.brokers_to_clients: Dict = {}
        self.clients_to_brokers: Dict = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings: Optional[SessionSettings] = None
        self.dag: Optional[WorkflowDAG] = None
        self.coordinator_socket = None
        self.communication: Optional[Communication] = None

        self.workers: List[Process] = []
        self.worker_result_queues: List = []
        self.worker_result_queue = asyncio.Queue()
        self.read_worker_results_task = None
        self.monitor_task = None
        self.worker_queue = None
        self.items_in_worker_queue: int = 0
        self.workers_ready: bool = False
        self.pending_tasks: List[Task] = []
        self.identity: str = "broker_%s" % ''.join(random.choice('0123456789abcdef') for _ in range(6))

        # For statistics
        self.start_time = time.time()
        self.task_start_times: Dict[str, float] = {}
        self.task_statistics: List[Tuple] = []

        self.logger.info("Broker %s initialized", self.identity)

    def write_statistics(self):
        broker_id = self.identity.split("_")[1]
        with open(os.path.join(self.settings.data_dir, "tasks_%s.csv" % self.identity), "w") as tasks_file:
            tasks_file.write("broker,task_name,function,worker,transfer_to_time,execute_time,transfer_from_time,total_time\n")
            for task_stats in self.task_statistics:
                task_stats_str = "%s,%s,%d,%f,%f,%f,%f" % task_stats
                tasks_file.write("%s,%s\n" % (broker_id, task_stats_str))

    async def monitor(self):
        self.logger.info("Started monitor")
        broker_id = self.identity.split("_")[1]
        process = psutil.Process(os.getpid())

        file_path = os.path.join(self.settings.data_dir, "resources_%s.csv" % self.identity)
        with open(file_path, "w") as resources_file:
            resources_file.write("broker,time,queue_items,cpu_percent,phys_mem_usage,virt_mem_usage,shared_mem_usage\n")

        while True:
            try:
                with open(file_path, "a") as resources_file:
                    cpu_usage = psutil.cpu_percent()
                    mem_info = process.memory_info()
                    phys_mem = mem_info.rss
                    virt_mem = mem_info.vms
                    shared_mem = 0 if not hasattr(mem_info, "shared") else mem_info.shared

                    resources_file.write("%s,%f,%d,%f,%d,%d,%d\n" % (broker_id, time.time() - self.start_time,
                                                                 self.items_in_worker_queue, cpu_usage,
                                                                 phys_mem, virt_mem, shared_mem))
                await asyncio.sleep(1)
            except Exception as exc:
                self.logger.exception(exc)
                self.shutdown_everyone()
                break

    async def start_workers(self):
        self.logger.info("Starting %d workers...", self.args.workers)
        self.worker_queue = multiprocessing.Queue()
        self.read_worker_results_task = asyncio.create_task(self.worker_result_queue_task())
        self.monitor_task = asyncio.create_task(self.monitor())
        for worker_ind in range(self.args.workers):
            await self.start_worker(worker_ind)

        # Workers ready - clear any pending tasks
        self.workers_ready = True
        for pending_task in self.pending_tasks:
            self.put_in_worker_queue(pending_task)
        self.pending_tasks = []

    def worker_result_queue_thread(self, mp_queue, loop):
        while True:
            item = mp_queue.get()
            if item is None:  # Sentinel value to end loop
                break

            asyncio.run_coroutine_threadsafe(self.worker_result_queue.put(item), loop)

    async def worker_result_queue_task(self):
        self.logger.info("Starting result queue task")
        while True:
            try:
                task_name, res, info = await self.worker_result_queue.get()
                if task_name == "error":
                    # One of the workers encountered an exception - stop everything
                    self.logger.info("Worker encountered an exception - shutting down everything")
                    self.shutdown_everyone()
                    break

                task = self.dag.tasks[task_name]
                self.items_in_worker_queue -= 1

                received_by_broker_time = time.time()
                received_by_worker_time = info["received"]
                finished_time = info["finished"]
                task_start_time = self.task_start_times[task_name]
                task_stats = (task.name, task.func, info["worker"],
                              received_by_worker_time - task_start_time,
                              finished_time - received_by_worker_time,
                              received_by_broker_time - finished_time,
                              received_by_broker_time - task_start_time)
                self.task_statistics.append(task_stats)

                self.logger.info("Task %s completed", task.name)

                # If this is a sink task, inform the coordinator about the result
                if not task.outputs:
                    # This is a sink task with no further outputs - send the result back to the coordinator
                    msg = pickle.dumps({"type": "result", "task": task.name, "result": res})
                    self.communication.send_message_to_coordinator(msg)
                else:
                    # Some broker needs this result - get all brokers we need to inform about this result
                    brokers_to_inform: Set[str] = set()
                    for next_task in task.outputs:
                        peer_next_task = next_task.data["peer"]
                        brokers_to_inform.add(self.clients_to_brokers[peer_next_task])

                    for broker_to_inform in brokers_to_inform:
                        if broker_to_inform == self.identity:
                            self.handle_task_result(task, res)
                        else:
                            msg = pickle.dumps({"type": "result", "task": task.name, "result": res})
                            self.communication.send_message_to_broker(broker_to_inform, msg)
            except Exception as exc:
                self.logger.exception(exc)
                self.shutdown_everyone()

    async def start_worker(self, index: int):
        worker_result_queue = multiprocessing.Queue()
        self.worker_result_queues.append(worker_result_queue)
        threading.Thread(target=self.worker_result_queue_thread, args=(worker_result_queue, asyncio.get_event_loop()), daemon=True).start()

        proc = multiprocessing.Process(target=worker_proc, args=(self.worker_queue, worker_result_queue, index, self.settings))
        proc.start()
        self.workers.append(proc)
        self.logger.info("Worker %d started: %s", index, proc)

    def connect_to_brokers(self):
        self.logger.info("Connecting to other brokers: %s", self.broker_addresses)
        for broker_name, broker_address in self.broker_addresses.items():
            if broker_name == self.identity:
                continue

            self.communication.connect_to(broker_name, broker_address)

    def shutdown(self):
        self.write_statistics()

        for worker_proc in self.workers:
            worker_proc.terminate()

        asyncio.get_event_loop().call_later(2, asyncio.get_event_loop().stop)

    def shutdown_everyone(self):
        self.logger.error("Will send shutdown signal to all nodes")
        msg = pickle.dumps({"type": "shutdown"})
        self.communication.send_message_to_all_brokers(msg)
        self.communication.send_message_to_coordinator(msg)
        self.shutdown()

    def put_in_worker_queue(self, task: Task):
        self.items_in_worker_queue += 1
        self.worker_queue.put((task.name, task.func, task.data))

    def schedule_task(self, task: Task):
        """
        Schedule the task on one of the available workers.
        """
        self.task_start_times[task.name] = time.time()
        if not self.workers_ready:
            self.logger.info("Broker enqueueing task %s since workers are not ready yet", task.name)
            self.pending_tasks.append(task)
        else:
            self.logger.info("Broker dispatching task %s to workers", task.name)
            self.put_in_worker_queue(task)

    def handle_task_result(self, task: Task, res):
        """
        We received a task result - either locally or from another worker. Handle it.
        """
        self.logger.info("Handling result of task %s", task)
        for next_task in task.outputs:
            if next_task.data["peer"] in self.brokers_to_clients[self.identity]:
                next_task.set_data(task.name, res)
                if next_task.has_all_inputs():
                    self.schedule_task(next_task)

    def on_message(self, identity: str, msg: Dict):
        if identity == "coordinator" and msg["type"] == "config":  # Configuration received from the coordinator
            self.logger.info("Received configuration from the coordinator")
            self.broker_addresses = msg["brokers"]
            self.brokers_to_clients = msg["brokers_to_clients"]

            # Build the reverse map
            for broker, clients in self.brokers_to_clients.items():
                for client in clients:
                    self.clients_to_brokers[client] = broker

            self.settings = SessionSettings.from_dict(msg["settings"])
            setup_logging(self.settings.data_dir, "%s.log" % self.identity)
            self.dag = WorkflowDAG.unserialize(msg["dag"])
            self.connect_to_brokers()
            ensure_future(self.start_workers())
        elif identity == "coordinator" and msg["type"] == "tasks":
            for task_name in msg["tasks"]:
                # Enqueue tasks
                task = self.dag.tasks[task_name]
                self.schedule_task(task)
        elif msg["type"] == "shutdown":
            self.logger.info("Received shutdown signal - stopping")
            self.shutdown()
        elif msg["type"] == "result":
            self.logger.info("Received task %s result from %s", msg["task"], identity)
            task = self.dag.tasks[msg["task"]]
            self.handle_task_result(task, msg["result"])
        else:
            raise RuntimeError("Unknown message with type %s" % msg["type"])

    async def start(self):
        self.communication = Communication(self.identity, self.args.port, self.on_message, is_broker=True)
        self.communication.start()
        self.communication.connect_to_coordinator(self.args.coordinator)
