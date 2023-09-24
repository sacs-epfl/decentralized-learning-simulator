import heapq
import logging
import math
import os
import pickle
import shutil
import time
from random import Random
from typing import List

import networkx as nx

from dasklearn.models import create_model, serialize_model
from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.events import *


from dask.distributed import Client as DaskClient, LocalCluster

from dasklearn.simulation.client import Client


class Simulation:
    """
    Contains general control logic related to the simulator.
    """

    def __init__(self, settings: SessionSettings):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.data_dir = os.path.join("data", "%s_%s_n%d_s%d" % (settings.algorithm, settings.dataset, settings.participants, settings.seed))
        settings.data_dir = self.data_dir

        self.settings = settings
        self.events: List[Event] = []
        heapq.heapify(self.events)
        self.tasks = {}  # Workflow DAG
        self.model_size: int = 0
        self.current_time: float = 0

        # TODO assume D-PSGD for now
        k = math.floor(math.log2(self.settings.participants))
        self.topology = nx.random_regular_graph(k, self.settings.participants, seed=self.settings.seed)

        self.clients: List[Client] = []

    def setup_directories(self):
        if not os.path.exists("data"):
            os.mkdir("data")

        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

        # Create accuracies file
        with open(os.path.join(self.data_dir, "accuracies.csv"), "w") as accuracies_file:
            accuracies_file.write("peer,round,time,accuracy,loss\n")

    def run(self):
        self.setup_directories()

        # Initialize the clients
        for client_id in range(self.settings.participants):
            self.clients.append(Client(self, client_id))
            model_init_event = Event(0, client_id, MODEL_INIT)
            heapq.heappush(self.events, model_init_event)

        # Apply traces if applicable
        if self.settings.capability_traces:
            self.logger.info("Applying capability trace file %s", self.settings.capability_traces)
            with open(self.settings.capability_traces, "rb") as traces_file:
                data = pickle.load(traces_file)

            rand = Random(self.settings.seed)
            device_ids = rand.sample(list(data.keys()), self.settings.participants)
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
            event = heapq.heappop(self.events)
            self.current_time = event.time
            self.process_event(event)

        self.evaluate_workflow_graph()

        # Done! Sanity checks
        for client in self.clients:
            assert len(client.bw_scheduler.incoming_requests) == 0
            assert len(client.bw_scheduler.outgoing_requests) == 0
            assert len(client.bw_scheduler.incoming_transfers) == 0
            assert len(client.bw_scheduler.outgoing_transfers) == 0

    def process_event(self, event: Event):
        if event.action == MODEL_INIT:
            self.clients[event.client_id].init_model(event)
        elif event.action == START_TRAIN:
            self.clients[event.client_id].start_train(event)
        elif event.action == FINISH_TRAIN:
            self.clients[event.client_id].finish_train(event)
        elif event.action == START_TRANSFER:
            self.clients[event.client_id].start_transfer(event)
        elif event.action == FINISH_OUTGOING_TRANSFER:
            self.clients[event.client_id].finish_outgoing_transfer(event)
        elif event.action == AGGREGATE:
            self.clients[event.client_id].aggregate(event)
        else:
            raise RuntimeError("Unknown event %s!", event.action)

    def schedule(self, event: Event):
        heapq.heappush(self.events, event)

    def evaluate_workflow_graph(self):
        if self.settings.scheduler:
            client = DaskClient(self.settings.scheduler)
        else:
            # Start a local Dask cluster and connect to it
            cluster = LocalCluster(n_workers=self.settings.workers)
            client = DaskClient(cluster)
            print("Client dashboard URL: %s" % client.dashboard_link)

        # Submit the tasks
        self.logger.info("Evaluating workflow graph...")
        start_time = time.time()
        frontier_tasks = [c.latest_task for c in self.clients if c.latest_task is not None]
        self.logger.info("Frontier tasks: %s" % frontier_tasks)
        result = client.get(self.tasks, frontier_tasks)
        elapsed_time = time.time() - start_time

        print("Final result: %s (took %d s.)" % (result, elapsed_time))
