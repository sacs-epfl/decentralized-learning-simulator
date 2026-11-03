"""RelaySGD Algorithm 1: one trained parcel and one relay exchange per round."""
import csv
import json
import math
import os
from collections import defaultdict

import networkx as nx

from dasklearn.models import create_model
from dasklearn.simulation.events import Event, START_ROUND
from dasklearn.simulation.relaysgd.client import RelaySGDClient
from dasklearn.simulation.simulation import Simulation
from dasklearn.tasks.references import TaskRef


class RelaySGDSimulation(Simulation):
    CLIENT_CLASS = RelaySGDClient

    def __init__(self, settings):
        if settings.participants < 1 or settings.learning.local_steps != 1:
            raise ValueError('RelaySGD requires positive participants and local_steps=1')
        if settings.learning.momentum:
            raise ValueError('RelaySGD currently implements plain SGD; set momentum=0')
        if not math.isfinite(settings.learning.learning_rate) or settings.learning.learning_rate <= 0:
            raise ValueError('RelaySGD requires a positive finite learning rate')
        if not math.isfinite(settings.learning.weight_decay) or settings.learning.weight_decay < 0:
            raise ValueError('RelaySGD requires nonnegative finite weight decay')
        if settings.churn != 'none' or settings.lora:
            raise ValueError('RelaySGD requires stable membership and fully trainable models')
        if settings.test_method not in {'individual', 'snapshot'}:
            raise ValueError('RelaySGD supports individual or snapshot evaluation')
        if settings.normalization not in {'counts', 'initial'}:
            raise ValueError('Unknown RelaySGD startup normalization')
        n = settings.participants
        if settings.topology == 'balanced':
            graph = nx.Graph()
            graph.add_nodes_from(range(n))
            graph.add_edges_from(((i - 1) // 2, i) for i in range(1, n))
        elif settings.topology == 'path':
            graph = nx.path_graph(n)
        elif settings.topology == 'star':
            graph = nx.star_graph(n - 1)
        else:
            raise ValueError('RelaySGD requires a balanced, path or star tree')
        self.graph = graph
        model = create_model(settings.dataset, architecture=settings.model, settings=settings)
        parameters = list(model.parameters())
        if (not parameters or any(not p.requires_grad or not p.is_floating_point() for p in parameters)
                or len({p.dtype for p in parameters}) != 1):
            raise ValueError('RelaySGD requires unfrozen floating parameters with a common dtype')
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
        self.packet_size = self.parameter_bytes + 8  # Dense sum plus an int64 contribution count.
        self.ready = defaultdict(dict)
        super().__init__(settings)
        self.register_event_callback(START_ROUND, 'start_round')

    def neighbors(self, peer):
        return sorted(self.graph.neighbors(peer))

    def initialize_clients(self):
        if any(c.get_train_time() <= 0 or c.bw_scheduler.bw_limit <= 0 for c in self.clients):
            raise ValueError('RelaySGD requires positive training durations and bandwidths')
        task = self.clients[0].create_compute_task('initialize', {'peer': 0, 'round': 0, 'time': 0})
        self.initial_model = TaskRef(task.name, 0)
        super().initialize_clients()

    def schedule_round(self, peer, r, model, previous):
        data = {'round': r, 'model': model, 'previous': previous}
        if not self.settings.synchronous:
            self.schedule(Event(self.current_time, peer, START_ROUND, data))
            return
        self.ready[r][peer] = data
        if len(self.ready[r]) == self.settings.participants:
            for p, values in sorted(self.ready.pop(r).items()):
                self.schedule(Event(self.current_time, p, START_ROUND, values))

    def save_measurements(self):
        super().save_measurements()
        with open(os.path.join(self.data_dir, 'relaysgd-tree.json'), 'w') as f:
            json.dump({'topology': self.settings.topology, 'edges': sorted(self.graph.edges),
                       'diameter': nx.diameter(self.graph), 'normalization': self.settings.normalization,
                       'packet_size_bytes': self.packet_size}, f, indent=2)
        with open(os.path.join(self.data_dir, 'relaysgd-rounds.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['peer', 'round', 'gradient_ready_us', 'completed_time_us', 'population_count'])
            for c in self.clients:
                for r, state in sorted(c.rounds.items()):
                    if state['done']:
                        writer.writerow([c.index, r, state['gradient_ready'], state['completed_time'], state['population']])
