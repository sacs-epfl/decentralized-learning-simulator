"""Plain SGP (Algorithm 1), with explicit numerator and scalar mass packets."""
import csv
import json
import math
import os
from collections import defaultdict

import networkx as nx

from dasklearn.models import create_model
from dasklearn.simulation.events import Event, START_ROUND
from dasklearn.simulation.sgp.client import SGPClient
from dasklearn.simulation.simulation import Simulation
from dasklearn.tasks.references import TaskRef


class SGPSimulation(Simulation):
    CLIENT_CLASS = SGPClient

    def __init__(self, settings):
        if settings.participants < 1 or settings.learning.local_steps != 1:
            raise ValueError('SGP requires positive participants and local_steps=1')
        if settings.learning.momentum:
            raise ValueError('SGP currently implements plain SGD; set momentum=0')
        if not math.isfinite(settings.learning.learning_rate) or settings.learning.learning_rate <= 0:
            raise ValueError('SGP requires a positive finite learning rate')
        if not math.isfinite(settings.learning.weight_decay) or settings.learning.weight_decay < 0:
            raise ValueError('SGP requires nonnegative finite weight decay')
        if settings.churn != 'none' or settings.lora:
            raise ValueError('SGP requires stable membership and fully trainable models')
        if settings.test_method not in {'individual', 'snapshot'}:
            raise ValueError('SGP supports individual or snapshot evaluation')
        n = settings.participants
        self.graphs = []
        if settings.topology == 'exponential':
            # Appendix A: one peer per iteration; offsets 1,2,...,2^floor(log2(n-1)).
            offsets = [1 << k for k in range((n - 1).bit_length())] if n > 1 else [0]
        elif settings.topology in {'ring', 'irregular', 'complete'}:
            offsets = [1]
        else:
            raise ValueError('Unknown SGP topology')
        for offset in offsets:
            graph = nx.DiGraph()
            graph.add_nodes_from(range(n))
            if settings.topology == 'complete':
                graph.add_edges_from((i, j) for i in range(n) for j in range(n) if i != j)
            elif n > 1:
                graph.add_edges_from((i, (i + offset) % n) for i in range(n))
                if settings.topology == 'irregular' and n > 2:
                    graph.add_edge(0, 2)
            self.graphs.append(graph)
        # Some exponential phases need not be connected alone. Their periodic
        # union contains the directed cycle, so it is strongly connected.
        union = nx.compose_all(self.graphs)
        if not nx.is_strongly_connected(union):
            raise ValueError('SGP requires a jointly strongly connected schedule')
        model = create_model(settings.dataset, architecture=settings.model, settings=settings)
        parameters = list(model.parameters())
        if (not parameters or any(not p.requires_grad or not p.is_floating_point() for p in parameters)
                or len({p.dtype for p in parameters}) != 1):
            raise ValueError('SGP requires unfrozen floating parameters with a common dtype')
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
        self.packet_size = self.parameter_bytes + 8  # Numerator vector plus float64 scalar mass.
        self.ready = defaultdict(dict)
        super().__init__(settings)
        self.register_event_callback(START_ROUND, 'start_round')

    def graph(self, r):
        return self.graphs[(r - 1) % len(self.graphs)]

    def incoming(self, peer, r):
        return sorted(self.graph(r).predecessors(peer))

    def outgoing(self, peer, r):
        return sorted(self.graph(r).successors(peer))

    def weight(self, peer, r):
        return 1. / (1 + self.graph(r).out_degree(peer))

    def initialize_clients(self):
        if any(c.get_train_time() <= 0 or c.bw_scheduler.bw_limit <= 0 for c in self.clients):
            raise ValueError('SGP requires positive training durations and bandwidths')
        task = self.clients[0].create_compute_task('initialize', {'peer': 0, 'round': 0, 'time': 0})
        self.initial_model = TaskRef(task.name, 0)
        super().initialize_clients()

    def schedule_round(self, peer, r, model, numerator, mass):
        data = {'round': r, 'model': model, 'numerator': numerator, 'mass': mass}
        if not self.settings.synchronous:
            self.schedule(Event(self.current_time, peer, START_ROUND, data))
            return
        self.ready[r][peer] = data
        if len(self.ready[r]) == self.settings.participants:
            for p, values in sorted(self.ready.pop(r).items()):
                self.schedule(Event(self.current_time, p, START_ROUND, values))

    def save_measurements(self):
        super().save_measurements()
        phases = []
        for graph in self.graphs:
            matrix = [[(1. / (1 + graph.out_degree(j))) if i == j or graph.has_edge(j, i)
                       else 0. for j in graph.nodes] for i in graph.nodes]
            phases.append({'edges': sorted(graph.edges), 'mixing_matrix': matrix})
        with open(os.path.join(self.data_dir, 'sgp-mixing.json'), 'w') as f:
            json.dump({'topology': self.settings.topology, 'period': len(self.graphs), 'phases': phases,
                       'packet_size_bytes': self.packet_size, 'mass_bytes': 8}, f, indent=2)
        with open(os.path.join(self.data_dir, 'sgp-rounds.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['peer', 'round', 'gradient_ready_us', 'completed_time_us', 'pushsum_mass'])
            for c in self.clients:
                for r, state in sorted(c.rounds.items()):
                    if state['done']:
                        writer.writerow([c.index, r, state['gradient_ready'], state['completed_time'], state['mass']])
