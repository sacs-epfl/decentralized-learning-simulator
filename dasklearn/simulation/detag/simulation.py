"""Fixed-topology DeTAG with overlapped computation and accelerated gossip."""
import csv
import json
import math
import os
from collections import defaultdict

import networkx as nx
import numpy as np

from dasklearn.models import create_model
from dasklearn.simulation.detag.client import DeTAGClient
from dasklearn.simulation.events import Event, START_ROUND
from dasklearn.simulation.simulation import Simulation
from dasklearn.tasks.references import TaskRef


class DeTAGSimulation(Simulation):
    CLIENT_CLASS = DeTAGClient

    def __init__(self, settings):
        if settings.participants < 1 or settings.learning.local_steps != 1:
            raise ValueError('DeTAG requires local_steps=1; use detag_phase_length for fixed-parameter accumulation')
        if settings.learning.momentum:
            raise ValueError('DeTAG implements the plain SGD version; set momentum=0')
        if not math.isfinite(settings.learning.learning_rate) or settings.learning.learning_rate <= 0:
            raise ValueError('DeTAG requires a positive finite learning rate')
        if not math.isfinite(settings.learning.weight_decay) or settings.learning.weight_decay < 0:
            raise ValueError('DeTAG requires nonnegative finite weight decay')
        if type(settings.phase_length) is not int or settings.phase_length < 1:
            raise ValueError('DeTAG phase_length must be a positive integer')
        if settings.eta is not None and (not math.isfinite(settings.eta) or not 0 <= settings.eta < 1):
            raise ValueError('DeTAG eta must be finite in [0,1)')
        if settings.churn != 'none' or settings.lora:
            raise ValueError('DeTAG requires stable membership and fully trainable models')
        if settings.test_method not in {'individual', 'snapshot'}:
            raise ValueError('DeTAG supports individual or snapshot evaluation')
        n = settings.participants
        if settings.topology == 'ring':
            graph = nx.cycle_graph(n)
            graph.remove_edges_from(nx.selfloop_edges(graph))
        elif settings.topology == 'complete':
            graph = nx.complete_graph(n)
        elif settings.topology == 'kreg':
            if type(settings.degree) is not int or not 0 <= settings.degree < n or n * settings.degree % 2:
                raise ValueError('Invalid DeTAG regular graph degree')
            for attempt in range(128):
                graph = nx.random_regular_graph(settings.degree, n, seed=settings.seed + 1009 * attempt)
                if nx.is_connected(graph):
                    break
            else:
                raise ValueError('DeTAG requires a connected graph')
        else:
            raise ValueError('Unknown DeTAG topology')
        self.graph = graph
        self.weights = np.zeros((n, n))
        for i, j in graph.edges:
            self.weights[i, j] = self.weights[j, i] = 1 / (1 + max(graph.degree(i), graph.degree(j)))
        for i in range(n):
            self.weights[i, i] = 1 - self.weights[i].sum()
        self.spectral_radius = float(np.max(np.abs(np.linalg.eigvalsh(self.weights - np.ones((n, n)) / n))))
        if self.spectral_radius >= 1:
            raise ValueError('DeTAG requires a mixing matrix with non-consensus spectral radius below one')
        if settings.eta is None:
            root = math.sqrt(max(0., 1 - self.spectral_radius ** 2))
            settings.eta = (1 - root) / (1 + root)
        model = create_model(settings.dataset, architecture=settings.model, settings=settings)
        parameters = list(model.parameters())
        if (not parameters or any(not p.requires_grad or not p.is_floating_point() for p in parameters)
                or len({p.dtype for p in parameters}) != 1):
            raise ValueError('DeTAG requires unfrozen floating parameters with a common dtype')
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
        self.packet_size = 2 * self.parameter_bytes  # Model replica and tracker, fused in one message.
        self.ready = defaultdict(dict)
        super().__init__(settings)
        self.register_event_callback(START_ROUND, 'start_round')

    def neighbors(self, peer):
        return sorted(self.graph.neighbors(peer))

    def initialize_clients(self):
        if any(c.get_train_time() <= 0 or c.bw_scheduler.bw_limit <= 0 for c in self.clients):
            raise ValueError('DeTAG requires positive training durations and bandwidths')
        task = self.clients[0].create_compute_task('initialize', {'peer': 0, 'round': 0, 'time': 0})
        self.initial_model = TaskRef(task.name, 0)
        super().initialize_clients()

    def schedule_round(self, peer, r, model, tracker, previous_gradient):
        data = {'round': r, 'model': model, 'tracker': tracker, 'previous_gradient': previous_gradient}
        if not self.settings.synchronous:
            self.schedule(Event(self.current_time, peer, START_ROUND, data))
            return
        self.ready[r][peer] = data
        if len(self.ready[r]) == self.settings.participants:
            for p, values in sorted(self.ready.pop(r).items()):
                self.schedule(Event(self.current_time, p, START_ROUND, values))

    def save_measurements(self):
        super().save_measurements()
        with open(os.path.join(self.data_dir, 'detag-mixing.json'), 'w') as f:
            json.dump({'weights': self.weights.tolist(), 'spectral_radius': self.spectral_radius,
                       'eta': self.settings.eta, 'phase_length': self.settings.phase_length,
                       'packet_size_bytes': self.packet_size}, f, indent=2)
        with open(os.path.join(self.data_dir, 'detag-steps.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['peer', 'phase', 'step', 'started_us', 'gradient_ready_us', 'gossip_ready_us', 'completed_us'])
            for client in self.clients:
                for phase, state in sorted(client.rounds.items()):
                    for step, inner in sorted(state['steps'].items()):
                        if inner['done']:
                            writer.writerow([client.index, phase, step, inner['started_time'],
                                             inner['gradient_ready'], inner['gossip_ready'], inner['completed_time']])
