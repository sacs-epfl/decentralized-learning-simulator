import asyncio
import bisect
import os
import pickle
import shutil
import resource
import psutil
import random
import time
from asyncio import Future
from random import Random
from typing import List, Optional, Callable, Tuple
from datetime import datetime
from heapq import nlargest

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
try:
    import seaborn as sns
    seaborn_available = True
except ImportError:
    seaborn_available = False

from dasklearn.communication import Communication
from dasklearn.models import create_model, serialize_model
from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.events import *

from dasklearn.simulation.client import BaseClient
from dasklearn.tasks.dag import WorkflowDAG
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from dasklearn.util.logging import setup_logging


class Simulation:
    """
    Contains general control logic related to the simulator.
    """
    CLIENT_CLASS = BaseClient

    def __init__(self, settings: SessionSettings):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.data_dir = ""
        self.setup_data_dir(settings)

        self.settings = settings
        self.events: List[Tuple[int, int, Event]] = []
        self.event_callbacks: Dict[str, str] = {}
        self.workflow_dag = WorkflowDAG()
        self.model_size: int = 0
        self.current_time: int = 0
        self.brokers_available_future: Future = Future()
        self.simulation_start_time: float = 0

        self.clients: List[BaseClient] = []
        self.broker_addresses: Dict[str, str] = {}
        self.brokers_to_clients: Dict = {}
        self.clients_to_brokers: Dict = {}

        self.communication: Optional[Communication] = None
        self.n_sink_tasks: int = 0
        self.sink_tasks_counter: int = 0

        self.memory_log: List[Tuple[int, psutil.pmem]] = []  # time, memory info

        self.register_event_callback(INIT_CLIENT, "init_client")
        self.register_event_callback(START_TRAIN, "start_train")
        self.register_event_callback(START_TRANSFER, "start_transfer")
        self.register_event_callback(FINISH_OUTGOING_TRANSFER, "finish_outgoing_transfer")

    def setup_data_dir(self, settings: SessionSettings) -> None:
        self.data_dir = os.path.join(settings.work_dir, "data", "%s_%s_n%d_b%d_s%d_%s" %
                                     (settings.algorithm, settings.dataset, settings.participants,
                                      settings.brokers, settings.seed, datetime.now().strftime("%Y%m%d%H%M")))
        settings.data_dir = self.data_dir

    def setup_directories(self):
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

    def distribute_brokers(self):
        """
        Distribute the clients over the brokers using a very simple modulo-based assignment mapping.
        """
        self.brokers_to_clients = {}
        self.clients_to_brokers = {}

        for client in range(len(self.clients)):
            self.clients_to_brokers[client] = list(self.broker_addresses.keys())[client % len(self.broker_addresses)]

        # Build the reverse map
        for client, broker in self.clients_to_brokers.items():
            if broker not in self.brokers_to_clients:
                self.brokers_to_clients[broker] = []
            self.brokers_to_clients[broker].append(client)

    def on_message(self, identity: str, msg: Dict):
        if msg["type"] == "hello":  # Register a new broker
            self.logger.info("Registering new broker %s", msg["address"])
            self.communication.connect_to(identity, msg["address"])
            self.broker_addresses[identity] = msg["address"]
            if len(self.broker_addresses) >= self.settings.brokers:
                self.brokers_available_future.set_result(True)
        elif msg["type"] == "result":  # Received a sink task result
            self.logger.info("Received result for sink task %s" % msg["task"])
            completed_task_name: str = msg["task"]
            if completed_task_name not in self.workflow_dag.tasks:
                self.logger.error("Task %s not in the DAG!", completed_task_name)
                return

            self.workflow_dag.tasks[completed_task_name].done = True
            self.sink_tasks_counter += 1

            if self.sink_tasks_counter == self.n_sink_tasks:
                self.logger.info("All sink tasks completed - shutting down brokers")
                out_msg = pickle.dumps({"type": "shutdown"})
                self.communication.send_message_to_all_brokers(out_msg)
                self.logger.info("Plotting accuracies")
                self.merge_accuracies_files()
                self.plot_loss()
                self.logger.info("Simulation took %.2f s.", time.time() - self.simulation_start_time)
                asyncio.get_event_loop().call_later(2, asyncio.get_event_loop().stop)
        elif msg["type"] == "shutdown":
            self.logger.info("Received shutdown signal - stopping")
            asyncio.get_event_loop().stop()

    def initialize_clients(self):
        for client_id in range(self.settings.participants):
            self.clients.append(self.CLIENT_CLASS(self, client_id))
            init_client_event = Event(0, client_id, INIT_CLIENT)
            self.schedule(init_client_event)

    def apply_fedscale_traces(self):
        self.logger.info("Applying FedScale capability trace file")
        with open(os.path.join("data", "fedscale_traces"), "rb") as traces_file:
            data = pickle.load(traces_file)

        # Filter and convert all bandwidth values to bytes/s.
        data = {
            key: {
                **value,
                "communication": int(value["communication"]) * 1000 // 8  # Convert to bytes/s
            }
            for key, value in data.items()
            if int(value["communication"]) * 1000 // 8 >= self.settings.min_bandwidth  # Filter based on minimum bandwidth
        }

        rand = Random(self.settings.seed)
        device_ids = rand.sample(list(data.keys()), len(self.clients))
        nodes_bws: Dict[int, int] = {}
        for ind, client in enumerate(self.clients):
            client.simulated_speed = data[device_ids[ind]]["computation"]
            # Also apply the network latencies
            bw_limit: int = int(data[device_ids[ind]]["communication"])
            client.bw_scheduler.bw_limit = bw_limit
            nodes_bws[ind] = bw_limit

        for client in self.clients:
            client.other_nodes_bws = nodes_bws

    def apply_diablo_traces(self):
        # Read and process the latency matrix
        bw_means = []
        with open(os.path.join("data", "diablo.txt"), "r") as diablo_file:
            rows = diablo_file.readlines()
            for row in rows:
                values = list(map(float, row.strip().split(',')))
                mean_value = np.mean(values) * 1000 * 1000 // 8
                bw_means.append(mean_value)

        nodes_bws: Dict[bytes, int] = {}
        for ind, node in enumerate(self.nodes):
            # TODO this is rather arbitrary for now
            node.overlays[0].model_manager.model_trainer.simulated_speed = 100
            bw_limit: int = bw_means[ind % len(bw_means)]
            node.overlays[0].bw_scheduler.bw_limit = bw_limit
            nodes_bws[node.overlays[0].my_peer.public_key.key_to_bin()] = bw_limit

        for node in self.nodes:
            node.overlays[0].other_nodes_bws = nodes_bws

    def apply_traces(self):
        if self.settings.traces == "none":
            return
        elif self.settings.traces == "fedscale":
            self.apply_fedscale_traces()
        elif self.settings.traces == "diablo":
            self.apply_diablo_traces()
        else:
            raise RuntimeError("Unknown traces %s" % self.settings.traces)

    async def run(self):
        self.simulation_start_time: float = time.time()
        self.setup_directories()
        if not self.settings.unit_testing:
            setup_logging(self.data_dir, "coordinator.log")
            self.communication = Communication("coordinator", self.settings.port, self.on_message)
            self.communication.start()

        # Initialize the clients
        self.initialize_clients()

        # Apply traces
        self.apply_traces()

        # Apply strugglers
        n_strugglers: int = int(self.settings.participants * self.settings.stragglers_proportion + 0.0000001)
        strugglers = nlargest(n_strugglers, self.clients, key=lambda x: x.simulated_speed)
        for client in strugglers:
            client.struggler = True
            if self.settings.stragglers_ratio == 0.0:
                # arbitrary large int
                client.simulated_speed = 1000000000000
            else:
                client.simulated_speed /= self.settings.stragglers_ratio

        # Determine the size of the model, which will be used to determine the duration of model transfers
        self.model_size = len(serialize_model(create_model(self.settings.dataset, architecture=self.settings.model)))
        self.logger.info("Determine model size: %d bytes", self.model_size)

        process = psutil.Process()
        self.memory_log.append((self.current_time, process.memory_info()))

        while self.events:
            _, _, event = self.events.pop(0)
            assert event.time >= self.current_time, "New event %s cannot be executed in the past! (current time: %d)" % (str(event), self.current_time)
            self.current_time = event.time
            self.process_event(event)
            # No need to track memory at every event
            if random.random() < 0.1 / self.settings.participants:
                self.memory_log.append((self.current_time, process.memory_info()))

        self.memory_log.append((self.current_time, process.memory_info()))
        self.workflow_dag.save_to_file(os.path.join(self.data_dir, "workflow_graph.txt"))
        self.save_measurements()

        # Sanity check the DAG
        self.workflow_dag.check_validity()
        self.plot_compute_graph()
        self.n_sink_tasks = len(self.workflow_dag.get_sink_tasks())

        if not self.settings.dry_run:
            await self.solve_workflow_graph()

        # Done! Sanity checks
        for client in self.clients:
            assert len(client.bw_scheduler.incoming_requests) == 0
            assert len(client.bw_scheduler.outgoing_requests) == 0
            assert len(client.bw_scheduler.incoming_transfers) == 0
            assert len(client.bw_scheduler.outgoing_transfers) == 0

        if self.settings.dry_run:
            self.logger.info("Dry run - shutdown")
            asyncio.get_event_loop().stop()

    def cur_time_in_sec(self) -> float:
        return self.current_time / MICROSECONDS

    def register_event_callback(self, name: str, callback: str):
        self.event_callbacks[name] = callback

    def process_event(self, event: Event):
        if event.action not in self.event_callbacks:
            raise RuntimeError("Action %s has no callback!" % event.action)

        callback: Callable = getattr(self.clients[event.client_id], self.event_callbacks[event.action])
        callback(event)

    def schedule(self, event: Event):
        assert event.time >= self.current_time, "Cannot schedule event %s in the past!" % event
        bisect.insort(self.events, (event.time, event.index, event))

    def schedule_tasks_on_broker(self, tasks: List[Task], broker: str):
        msg = pickle.dumps({"type": "tasks", "tasks": [task.name for task in tasks]})
        self.communication.send_message_to_broker(broker, msg)

    async def solve_workflow_graph(self):
        self.logger.info("Will start solving workflow DAG with %d tasks, waiting for %d broker(s)...", len(self.workflow_dag.tasks), self.settings.brokers)

        await self.brokers_available_future

        self.logger.info("%d brokers available - starting to solve workload", len(self.broker_addresses))

        # Send all brokers the right configuration
        self.distribute_brokers()
        data = {
            "type": "config",
            "brokers": self.broker_addresses,
            "brokers_to_clients": self.brokers_to_clients,
            "settings": self.settings.to_dict(),
            "dag": self.workflow_dag.serialize()
        }
        msg = pickle.dumps(data)
        self.communication.send_message_to_all_brokers(msg)

        source_tasks = self.workflow_dag.get_source_tasks()
        brokers_to_tasks: Dict[str, List[Task]] = {}
        for source_task in source_tasks:
            broker = self.clients_to_brokers[source_task.data["peer"]]
            if broker not in brokers_to_tasks:
                brokers_to_tasks[broker] = []
            brokers_to_tasks[broker].append(source_task)

        for broker, tasks in brokers_to_tasks.items():
            self.logger.info("Scheduling %d task(s) on broker %s", len(tasks), broker)
            self.schedule_tasks_on_broker(tasks, broker)

    def plot_loss(self):
        if not seaborn_available:
            self.logger.warning("Seaborn not available - skipping plotting.")
            return
        # Check if the accuracies file exists
        path = os.path.join(self.settings.data_dir, "accuracies.csv")
        if not os.path.exists(path):
            self.logger.warning("accuracies.csv not found - skipping plotting.")
            return
        # Read the data
        data = pd.read_csv(path)
        # Create all combinations of time and peer and fill the missing values from previous measurements
        # This ensures the plot correctly shows std for algorithms which test at not exactly the same time for all peers
        all_combinations = data['peer'].drop_duplicates().to_frame().merge(data['time'].drop_duplicates(), how='cross')
        all_combinations = pd.merge(data, all_combinations, on=['peer', 'time'], how='outer')
        all_combinations.sort_values('time', inplace=True)
        all_combinations = all_combinations.groupby('peer').ffill()
        # Convert to hours
        all_combinations['hours'] = all_combinations['time'] / MICROSECONDS / 3600
        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.lineplot(all_combinations, x='hours', y='loss', ax=ax[0])
        sns.lineplot(all_combinations, x='hours', y='accuracy', ax=ax[1])
        plt.savefig(os.path.join(self.settings.data_dir, "accuracies.png"))

    def plot_compute_graph(self):
        self.logger.info("Plotting compute graph")
        graph, position, colors, color_key = self.workflow_dag.to_nx(self.settings.compute_graph_plot_size)
        nx.draw(graph, position, node_color=colors, node_size=50, arrows=True)
        # Dummy points for legend
        dummy_points = [plt.Line2D([0], [-1], marker='o', color='w', markerfacecolor=color, markersize=10,
                                   label=f'{node}') for node, color in color_key.items()]
        plt.legend(handles=dummy_points)
        plt.savefig(os.path.join(self.settings.data_dir, "compute_graph.png"))

    def merge_accuracies_files(self):
        paths = [os.path.join(self.settings.data_dir, "accuracies_" + str(i) + ".csv")
                 for i in range(self.settings.participants)]

        with open(os.path.join(self.settings.data_dir, "accuracies.csv"), "w") as output_file:
            output_file.write("peer,round,time,accuracy,loss\n")
            for path in paths:
                if os.path.exists(path):
                    with open(path, "r") as input_file:
                        next(input_file)  # Ignore the header
                        output_file.write(input_file.read())
                    os.remove(path)

    def save_measurements(self) -> None:
        # Write time utilization
        with open(os.path.join(self.data_dir, "time_utilisation.csv"), "w") as file:
            file.write("client,compute_time,total_time\n")
            for client in self.clients:
                # Supports only duration based algorithms
                file.write("%d,%d,%d\n" % (client.index, client.compute_time, self.settings.duration))
        # Write aggregation log
        with open(os.path.join(self.data_dir, "aggregations.csv"), "w") as file:
            file.write("client;clients;ages\n")
            for client in self.clients:
                for aggregation in client.aggregations:
                    file.write("%d;%s;%s\n" % (client.index, list(map(lambda x: x[0], aggregation)),
                                               list(map(lambda x: x[2], aggregation))))
        # Write incoming counter log
        with open(os.path.join(self.data_dir, "incoming.csv"), "w") as file:
            file.write("client,from,count\n")
            for client in self.clients:
                for sender, count in client.incoming_counter.items():
                    file.write("%d,%d,%d\n" % (client.index, sender, count))
        # Write opportunity log
        with open(os.path.join(self.data_dir, "opportunities.csv"), "w") as file:
            file.write("client,contributing_client,value\n")
            for client in self.clients:
                total_opportunity: int = sum(client.opportunity.values())
                for contributor, value in client.opportunity.items():
                    file.write("%d,%d,%f\n" % (client.index, contributor, value / total_opportunity))
        # Write client speed log
        with open(os.path.join(self.data_dir, "speeds.csv"), "w") as file:
            file.write("client,training_time\n")
            for client in self.clients:
                file.write("%d,%d\n" % (client.index, client.simulated_speed))
        # Write max memory log
        with open(os.path.join(self.data_dir, "max_memory.csv"), "w") as file:
            file.write("max_memory_usage_kb\n")
            file.write("%d\n" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Write memory log
        with open(os.path.join(self.data_dir, "memory.csv"), "w") as file:
            # In bytes
            file.write("time,physical,virtual,shared\n")
            for time, mem_info in self.memory_log:
                shared_mem = 0 if not hasattr(mem_info, "shared") else mem_info.shared
                file.write("%d,%d,%d,%d\n" % (time, mem_info.rss, mem_info.vms, shared_mem))
        # Write strugglers log
        if self.settings.stragglers_ratio > 0.0:
            with open(os.path.join(self.data_dir, "strugglers.csv"), "w") as file:
                file.write("client,struggler\n")
                for client in self.clients:
                    file.write("%d,%s\n" % (client.index, client.struggler))
