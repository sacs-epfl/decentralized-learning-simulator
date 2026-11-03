"""Algorithm 1 of Bornstein et al. (ICLR 2023), with regular-graph CCS."""
import csv
import json
import math
import os

import networkx as nx
import numpy as np

from dasklearn.models import create_model
from dasklearn.simulation.simulation import Simulation
from dasklearn.simulation.swift.client import SwiftClient
from dasklearn.tasks.references import TaskRef


def communication_weights(graph, influence):
    """Regular-graph specialization of CCS (Algorithm 2), in receiver-row order.

    All neighbors tie in degree. Include self in the remaining CIS mass, as
    the authors' uniform-CIS implementation does (1 / (degree + 1)). Each
    edge uses the maximum of its endpoints' remaining masses. Nonuniform CIS
    reserves 1/n self-weight before distributing the remaining mass.
    """
    n = len(graph)
    p = np.asarray(influence, dtype=float)
    if (graph.is_directed() or set(graph) != set(range(n)) or n < 1 or
            nx.number_of_selfloops(graph) or not nx.is_connected(graph) or
            len({d for _, d in graph.degree}) != 1):
        raise ValueError('SWIFT CCS currently supports connected undirected regular graphs')
    if p.shape != (n,) or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('SWIFT influence must contain one positive finite score per peer')
    p = p / p.max()  # Avoid overflow when normalizing large finite scores.
    p /= p.sum()
    reserve = 0.0 if np.all(p == p[0]) else 1.0 / n
    mass = [p[i] + sum(p[j] for j in graph.neighbors(i)) for i in range(n)]
    weights = np.zeros((n, n))
    for i, j in graph.edges:
        scale = (1 - reserve) / max(mass[i], mass[j])
        weights[i, j] = scale * p[j]
        weights[j, i] = scale * p[i]
    for i in range(n):
        weights[i, i] = 1 - weights[i].sum()
    return p, weights


class SwiftSimulation(Simulation):
    CLIENT_CLASS = SwiftClient

    def __init__(self, settings):
        if settings.participants < 1 or settings.learning.local_steps != 1:
            raise ValueError('SWIFT requires local_steps=1; use swift_averaging_interval for periodic averaging')
        if settings.learning.momentum:
            raise ValueError('SWIFT implements SGD without momentum; set momentum=0')
        if not math.isfinite(settings.learning.learning_rate) or settings.learning.learning_rate <= 0:
            raise ValueError('SWIFT requires a positive finite learning rate')
        if not math.isfinite(settings.learning.weight_decay) or settings.learning.weight_decay < 0:
            raise ValueError('SWIFT requires nonnegative finite weight decay')
        if type(settings.averaging_interval) is not int or settings.averaging_interval < 1:
            raise ValueError('SWIFT averaging_interval must be a positive integer')
        if settings.synchronous or settings.churn != 'none':
            raise ValueError('SWIFT requires asynchronous execution and stable membership')
        if settings.lora:
            raise ValueError('SWIFT currently supports fully trainable models, not LoRA')
        if settings.test_method not in {'individual', 'snapshot'}:
            raise ValueError('SWIFT supports individual or snapshot evaluation')
        n = settings.participants
        if settings.topology == 'ring':
            graph = nx.cycle_graph(n)
            graph.remove_edges_from(nx.selfloop_edges(graph))
        elif settings.topology == 'complete':
            graph = nx.complete_graph(n)
        elif settings.topology == 'kreg':
            if type(settings.degree) is not int or not 0 <= settings.degree < n or n * settings.degree % 2:
                raise ValueError('Invalid SWIFT regular graph degree')
            for attempt in range(128):
                graph = nx.random_regular_graph(settings.degree, n, seed=settings.seed + 1009 * attempt)
                if nx.is_connected(graph):
                    break
            else:
                raise ValueError('SWIFT requires a connected graph')
        else:
            raise ValueError('Unknown SWIFT topology')
        self.graph = graph
        self.influence, self.weights = communication_weights(
            graph, settings.influence if settings.influence is not None else [1.] * n)
        model = create_model(settings.dataset, architecture=settings.model, settings=settings)
        parameters = list(model.parameters())
        if (not parameters or any(not p.requires_grad or not p.is_floating_point() for p in parameters)
                or len({p.dtype for p in parameters}) != 1):
            raise ValueError('SWIFT requires unfrozen floating parameters with a common dtype')
        self.packet_size = sum(p.numel() * p.element_size() for p in parameters)
        super().__init__(settings)

    def neighbors(self, peer):
        return sorted(self.graph.neighbors(peer))

    def initialize_clients(self):
        if any(c.get_train_time() <= 0 or c.bw_scheduler.bw_limit <= 0 for c in self.clients):
            raise ValueError('SWIFT requires positive training durations and bandwidths')
        task = self.clients[0].create_compute_task('initialize', {'peer': 0, 'round': 0, 'time': 0})
        self.initial_model = TaskRef(task.name, 0)
        packet = self.clients[0].create_compute_task('swift_pack', {'model': self.initial_model,
                                                                  'peer': 0, 'round': 0, 'time': 0})
        self.initial_packet = TaskRef(packet.name, 0)
        super().initialize_clients()

    def save_measurements(self):
        super().save_measurements()
        with open(os.path.join(self.data_dir, 'swift-weights.json'), 'w') as f:
            json.dump({'influence': self.influence.tolist(), 'receiver_row_weights': self.weights.tolist()}, f, indent=2)
        with open(os.path.join(self.data_dir, 'swift-rounds.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['peer', 'round', 'time_us', 'averaged', 'max_version_lag'])
            for client in self.clients:
                for r, state in sorted(client.rounds.items()):
                    if 'completed_time' in state:
                        writer.writerow([client.index, r, state['completed_time'], state['averaged'],
                                         max([0] + [r - 1 - v for v in state['versions'].values()])])
