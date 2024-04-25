import asyncio
import bisect
import os
import pickle
import shutil
from asyncio import Future
from random import Random
from typing import List, Optional, Callable, Tuple
from datetime import datetime

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

        self.clients: List[BaseClient] = []
        self.broker_addresses: Dict[str, str] = {}
        self.brokers_to_clients: Dict = {}
        self.clients_to_brokers: Dict = {}

        self.communication: Optional[Communication] = None

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

            sink_tasks: List[Task] = self.workflow_dag.get_sink_tasks()
            for task in sink_tasks:
                if task.name == completed_task_name:
                    task.done = True

            if all([task.done for task in sink_tasks]):
                self.logger.info("All sink tasks completed - shutting down brokers")
                out_msg = pickle.dumps({"type": "shutdown"})
                self.communication.send_message_to_all_brokers(out_msg)
                self.logger.info("Plotting accuracies")
                self.plot_loss()
                asyncio.get_event_loop().call_later(2, asyncio.get_event_loop().stop)
        elif msg["type"] == "shutdown":
            self.logger.info("Received shutdown signal - stopping")
            asyncio.get_event_loop().stop()

    def initialize_clients(self):
        for client_id in range(self.settings.participants):
            self.clients.append(self.CLIENT_CLASS(self, client_id))
            init_client_event = Event(0, client_id, INIT_CLIENT)
            self.schedule(init_client_event)

    async def run(self):
        self.setup_directories()
        if not self.settings.unit_testing:
            setup_logging(self.data_dir, "coordinator.log")
            self.communication = Communication("coordinator", self.settings.port, self.on_message)
            self.communication.start()

        # Initialize the clients
        self.initialize_clients()

        # Apply traces if applicable
        if self.settings.capability_traces:
            self.logger.info("Applying capability trace file %s", self.settings.capability_traces)
            with open(self.settings.capability_traces, "rb") as traces_file:
                data = pickle.load(traces_file)

            rand = Random(self.settings.seed)
            device_ids = rand.sample(list(data.keys()), len(self.clients))
            nodes_bws: Dict[int, int] = {}
            for ind, client in enumerate(self.clients):
                client.simulated_speed = data[device_ids[ind]]["computation"]
                # Also apply the network latencies
                bw_limit: int = int(data[ind + 1]["communication"]) * 1024 // 8
                client.bw_scheduler.bw_limit = bw_limit
                nodes_bws[ind] = bw_limit

            for client in self.clients:
                client.other_nodes_bws = nodes_bws
        else:
            for client in self.clients:
                client.bw_scheduler.bw_limit = 100000000000

        # Determine the size of the model, which will be used to determine the duration of model transfers
        self.model_size = len(serialize_model(create_model(self.settings.dataset, architecture=self.settings.model)))
        self.logger.info("Determine model size: %d bytes", self.model_size)

        while self.events:
            _, _, event = self.events.pop(0)
            assert event.time >= self.current_time, "New event %s cannot be executed in the past! (current time: %d)" % (str(event), self.current_time)
            self.current_time = event.time
            self.process_event(event)

        self.workflow_dag.save_to_file(os.path.join(self.data_dir, "workflow_graph.txt"))
        self.save_measurements()

        # Sanity check the DAG
        self.workflow_dag.check_validity()
        self.plot_compute_graph()

        if not self.settings.dry_run:
            await self.solve_workflow_graph()

        # Done! Sanity checks
        for client in self.clients:
            assert len(client.bw_scheduler.incoming_requests) == 0
            assert len(client.bw_scheduler.outgoing_requests) == 0
            assert len(client.bw_scheduler.incoming_transfers) == 0
            assert len(client.bw_scheduler.outgoing_transfers) == 0

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
        # Check if the accuracies file exists
        path = os.path.join(self.settings.data_dir, "accuracies.csv")
        if not os.path.exists(path):
            return
        # Read the data
        data = pd.read_csv(path, header=None, names=['peer', 'round', 'time', 'accuracy', 'loss'])
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
