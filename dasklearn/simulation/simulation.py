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
from typing import List, Optional, Callable, Set, Tuple
from datetime import datetime
from heapq import nlargest

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from dasklearn.simulation.churn_manager import ChurnManager
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
from dasklearn.util import MICROSECONDS, time_to_sec
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
        self.pending_tasks: Set[asyncio.Task] = set()
        self.workflow_dag = WorkflowDAG()
        self.model_size: int = 0
        self.current_time: int = 0
        self.brokers_available_future: Future = Future()
        self.simulation_start_time: float = 0
        self.churn_manager: ChurnManager = ChurnManager(settings)
        self.num_active_clients: int = 0
        self.active_counts: List[Tuple[int, int]] = []

        self.clients: List[BaseClient] = []
        self.broker_addresses: Dict[str, str] = {}
        self.brokers_to_clients: Dict = {}
        self.clients_to_brokers: Dict = {}

        self.communication: Optional[Communication] = None
        self.n_sink_tasks: int = 0
        self.sink_tasks_counter: int = 0

        # Logging
        self.memory_log: List[Tuple[int, psutil.pmem]] = []  # time, memory info
        self.bw_utilization: List[Tuple[int, float, float]] = []  # time, utilization in, utilization out

        self.register_event_callback(INIT_CLIENT, "init_client")
        self.register_event_callback(START_TRAIN, "start_train")
        self.register_event_callback(START_TRANSFER, "start_transfer")
        self.register_event_callback(FINISH_OUTGOING_TRANSFER, "finish_outgoing_transfer")
        self.register_event_callback(SEND_MESSAGE, "on_message")
        self.register_event_callback(ONLINE, "come_online")
        self.register_event_callback(OFFLINE, "go_offline")
        self.register_event_callback(MONITOR_BANDWIDTH_UTILIZATION, "monitor_bandwidth_utilization")

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
        
    def apply_churn(self):
        if self.settings.churn == "none":
            return
        elif self.settings.churn == "synthetic":
            min_online = int(self.settings.participants * 0.1)
            max_online = int(self.settings.participants * 0.8)
            self.churn_manager.simulate_node_traces(self.settings.participants, min_online, max_online, 3600 * MICROSECONDS, self.settings.duration, time_step=MICROSECONDS)

            # Set nodes offline that should be offline
            for index in range(self.settings.participants):
                if index in self.churn_manager.traces and self.churn_manager.traces[index] and self.churn_manager.traces[index][0][0] != 0:
                    self.clients[index].online = False

            # Apply churn
            for index in range(self.settings.participants):
                traces: List[Tuple[int, int]] = self.churn_manager.traces.get(index, [])
                for start, end in traces:
                    if start == 0:
                        continue  # Ignore the very first event where the node starts online

                    start_event = Event(int(start), index, ONLINE)
                    self.schedule(start_event)
                    end_event = Event(int(end), index, OFFLINE)
                    self.schedule(end_event)

        elif self.settings.churn == "fedscale":
            self.logger.info("Applying FedScale availability trace file")
            with open(os.path.join("data", "fedscale_churn"), "rb") as traces_file:
                data = pickle.load(traces_file)

            rand = Random(self.settings.seed)
            device_ids = rand.sample(list(data.keys()), self.settings.participants)
            for ind in range(self.settings.participants):
                self.set_fedscale_trace_for_client(ind, data[device_ids[ind]])
        else:
            raise RuntimeError("Unknown churn model %s" % self.settings.churn)
        
        self.num_active_clients = sum([1 for client in self.clients if client.online])
        self.logger.info("Number of active clients at the start of the simulation: %d", self.num_active_clients)
        self.active_counts.append((int(self.cur_time_in_sec()), self.num_active_clients))

    def update_active_clients(self, comes_online: bool):
        if comes_online:
            self.num_active_clients += 1
        else:
            self.num_active_clients -= 1

        self.active_counts.append((self.current_time, self.num_active_clients))

    def set_fedscale_trace_for_client(self, client_id: int, data: Dict):
        events: int = 0

        # Figure out if the node starts online or not
        self.clients[client_id].online = True if (data["inactive"][0] < data["active"][0] or data["active"][0] == 0) else False

        # Combine the traces of this client into a list of tuples with the timestamp and the event type
        traces: List[Tuple[int, str]] = [(x, ONLINE) for x in data["active"] if x > 0] + [(x, OFFLINE) for x in data["inactive"]]
        traces.sort(key=lambda x: x[0])

        iterations: int = 0
        done: bool = False
        while True:
            for timestamp, event_type in traces:
                actual_timestamp = (timestamp + iterations * data["finish_time"]) * MICROSECONDS
                if actual_timestamp > self.settings.duration:
                    done = True
                    break

                event = Event(actual_timestamp, client_id, event_type)
                self.schedule(event)
                events += 1

            iterations += 1

            if done:
                break

        # # Clients that have no events scheduled should be online
        # if events == 0:
        #     self.clients[client_id].online = True

        self.logger.info("Scheduled %d join/leave events for client %d (trace length in sec: %d)", events, client_id, data["finish_time"])

    def start_profile(self):
        # Check if the Yappi library has been installed
        try:
            import yappi
        except ImportError:
            self.logger.error("Yappi library not installed - cannot profile")
            return
        yappi.start(builtins=True)

    def stop_profile(self):
        import yappi
        yappi.stop()
        yappi_stats = yappi.get_func_stats()
        yappi_stats.sort("tsub")
        yappi_stats.save(os.path.join(self.data_dir, "yappi.stats"), type='callgrind')

    def monitor_bandwidth_utilization(self, event: Event) -> None:
        """
        Monitor the bandwidth utilization of all clients that are sending or receiving data.
        """
        total_in, total_out, total_in_used, total_out_used = 0, 0, 0, 0
        for client in self.clients:
            in_used = client.bw_scheduler.get_allocated_incoming_bw()
            out_used = client.bw_scheduler.get_allocated_outgoing_bw()
            if in_used > 0 or out_used > 0:
                total_in += client.bw_scheduler.bw_limit
                total_out += client.bw_scheduler.bw_limit
                total_in_used += client.bw_scheduler.get_allocated_incoming_bw()
                total_out_used += client.bw_scheduler.get_allocated_outgoing_bw()

        self.bw_utilization.append((time_to_sec(self.current_time), total_in_used / total_in, total_out_used / total_out))

        # Schedule the next event
        next_event = Event(self.current_time + MICROSECONDS, 0, MONITOR_BANDWIDTH_UTILIZATION, is_global=True)
        self.schedule(next_event)

    async def run(self):
        if self.settings.profile:
            self.start_profile()

        self.simulation_start_time: float = time.time()
        self.setup_directories()
        if not self.settings.unit_testing:
            setup_logging(self.data_dir, "coordinator.log")
            self.communication = Communication("coordinator", self.settings.port, self.on_message)
            self.communication.start()

        # Create the clients
        for client_id in range(self.settings.participants):
            self.clients.append(self.CLIENT_CLASS(self, client_id))

        # Apply traces
        self.apply_traces()

        # Apply churn
        self.apply_churn()

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

        # Initialize the clients
        self.initialize_clients()

        # Initialize the bandwidth utilization monitor if needed
        if self.settings.log_bandwidth_utilization:
            self.logger.info("Initializing bandwidth utilization monitor")
            monitor_bw_event: Event = Event(MICROSECONDS, 0, MONITOR_BANDWIDTH_UTILIZATION, is_global=True)
            self.schedule(monitor_bw_event)

        # Determine the size of the model, which will be used to determine the duration of model transfers
        self.model_size = len(serialize_model(create_model(self.settings.dataset, architecture=self.settings.model)))
        self.logger.info("Determine model size: %d bytes", self.model_size)

        process = psutil.Process()
        self.memory_log.append((self.current_time, process.memory_info()))

        while self.events and self.current_time < self.settings.duration:
            # Process synchronous events
            _, _, event = self.events.pop(0)
            assert event.time >= self.current_time, "New event %s cannot be executed in the past! (current time: %d)" % (str(event), self.current_time)
            self.current_time = event.time
            self.process_event(event)
            # No need to track memory at every event
            if random.random() < 0.1 / self.settings.participants:
                self.memory_log.append((self.current_time, process.memory_info()))
        
        self.events = []

        self.memory_log.append((self.current_time, process.memory_info()))
        self.workflow_dag.save_to_file(os.path.join(self.data_dir, "workflow_graph.txt"))
        self.save_measurements()

        if self.settings.profile:
            self.stop_profile()

        # Sanity check the DAG
        self.workflow_dag.check_validity()
        self.plot_compute_graph()
        self.n_sink_tasks = len(self.workflow_dag.get_sink_tasks())

        if not self.settings.dry_run:
            await self.solve_workflow_graph()

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

        if event.is_global:
            callback: Callable = getattr(self, self.event_callbacks[event.action])
        else:
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
            "settings_class": self.settings.__class__.__name__,
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
            output_file.write("algorithm,dataset,partitioner,alpha,peer,round,time,accuracy,loss\n")
            for path in paths:
                if os.path.exists(path):
                    with open(path, "r") as input_file:
                        next(input_file)  # Ignore the header
                        output_file.write(input_file.read())
                    os.remove(path)

    def save_measurements(self) -> None:
        # Write client statistics
        with open(os.path.join(self.data_dir, "client_statistics.csv"), "w") as file:
            file.write("client,bytes_sent,bytes_received,compute_time,training_speed,bandwidth\n")
            for client in self.clients:
                file.write("%d,%d,%d,%d,%d,%d\n" % (
                    client.index, client.bw_scheduler.total_bytes_sent, client.bw_scheduler.total_bytes_received,
                    client.compute_time, client.simulated_speed, client.bw_scheduler.bw_limit))
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
        # Write active client log
        with open(os.path.join(self.data_dir, "churn.csv"), "w") as file:
            file.write("time,active_clients\n")
            for time, active_clients in self.active_counts:
                file.write("%d,%d\n" % (time, active_clients))
        # Write bandwidth utilizations
        if self.settings.log_bandwidth_utilization:
            with open(os.path.join(self.data_dir, "bw_utilization.csv"), "w") as file:
                file.write("time,incoming,outgoing\n")
                for time, in_util, out_util in self.bw_utilization:
                    file.write("%d,%f,%f\n" % (time, in_util, out_util))
