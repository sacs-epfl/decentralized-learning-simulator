"""MoTEF Algorithm 1 on a fixed symmetric Metropolis mixing matrix."""
import csv
import json
import math
import os
from collections import defaultdict

import networkx as nx

from dasklearn.models import create_model
from dasklearn.optimizer.motef import kept_count
from dasklearn.simulation.events import Event, START_ROUND
from dasklearn.simulation.motef.client import MoTEFClient
from dasklearn.simulation.simulation import Simulation
from dasklearn.tasks.references import TaskRef


class MoTEFSimulation(Simulation):
    CLIENT_CLASS = MoTEFClient
    MEASUREMENT_PREFIX = 'motef'

    def __init__(self, settings):
        if settings.participants < 1 or settings.learning.local_steps != 1:
            raise ValueError('MoTEF requires positive participants and local_steps=1')
        if settings.learning.momentum:
            raise ValueError('MoTEF uses its own momentum_weight; set shared momentum=0')
        if not math.isfinite(settings.learning.learning_rate) or settings.learning.learning_rate <= 0:
            raise ValueError('MoTEF requires a positive finite learning rate')
        if not math.isfinite(settings.learning.weight_decay) or settings.learning.weight_decay < 0:
            raise ValueError('MoTEF requires nonnegative finite weight decay')
        if settings.churn != 'none' or settings.lora:
            raise ValueError('MoTEF requires stable membership and fully trainable models')
        if settings.test_method not in {'individual', 'snapshot'}:
            raise ValueError('MoTEF supports individual or snapshot evaluation')
        if settings.compressor not in {'topk', 'randomk', 'identity'}:
            raise ValueError('Unknown MoTEF compressor')
        for name in ['keep_ratio', 'consensus_stepsize', 'momentum_weight']:
            value = getattr(settings, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError('MoTEF ' + name + ' must be finite and in (0,1]')
        n = settings.participants
        if settings.topology == 'ring':
            graph = nx.cycle_graph(n)
            graph.remove_edges_from(nx.selfloop_edges(graph))
        elif settings.topology == 'complete':
            graph = nx.complete_graph(n)
        elif settings.topology == 'kreg':
            if type(settings.degree) is not int or not 0 <= settings.degree < n or n * settings.degree % 2:
                raise ValueError('Invalid MoTEF regular graph degree')
            for attempt in range(128):
                graph = nx.random_regular_graph(settings.degree, n, seed=settings.seed + 1009 * attempt)
                if nx.is_connected(graph):
                    break
            else:
                raise ValueError('MoTEF requires a connected graph')
        else:
            raise ValueError('Unknown MoTEF topology')
        model = create_model(settings.dataset, architecture=settings.model, settings=settings)
        parameters = list(model.parameters())
        if (not parameters or any(not p.requires_grad or not p.is_floating_point() or not p.numel() for p in parameters)
                or len({p.dtype for p in parameters}) != 1):
            raise ValueError('MoTEF requires nonempty unfrozen floating parameters with a common dtype')
        self.parameter_count = sum(p.numel() for p in parameters)
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
        self.packet_size = (2 * self.parameter_bytes if settings.compressor == 'identity' else
            2 * kept_count(self.parameter_count, settings.keep_ratio) * (8 + parameters[0].element_size()))
        self.graph = graph
        self.ready = defaultdict(dict)
        super().__init__(settings)
        self.register_event_callback(START_ROUND, 'start_round')

    def neighbors(self, peer):
        return sorted(self.graph.neighbors(peer))

    def weight(self, peer, other):
        if peer == other:
            return 1 - sum(self.weight(peer, p) for p in self.neighbors(peer))
        return 1 / (1 + max(self.graph.degree(peer), self.graph.degree(other)))

    def initialize_clients(self):
        if any(c.get_train_time() <= 0 or c.bw_scheduler.bw_limit <= 0 for c in self.clients):
            raise ValueError('MoTEF requires positive training durations and bandwidths')
        task = self.clients[0].create_compute_task('initialize', {'peer': 0, 'round': 0, 'time': 0})
        self.initial_model = TaskRef(task.name, 0)
        super().initialize_clients()

    def schedule_round(self, peer, r, model, state):
        data = {'round': r, 'model': model, 'state': state}
        if not self.settings.synchronous:
            self.schedule(Event(self.current_time, peer, START_ROUND, data))
            return
        self.ready[r][peer] = data
        if len(self.ready[r]) == self.settings.participants:
            for p, values in sorted(self.ready.pop(r).items()):
                self.schedule(Event(self.current_time, p, START_ROUND, values))

    def save_measurements(self):
        super().save_measurements()
        with open(os.path.join(self.data_dir, self.MEASUREMENT_PREFIX + '-mixing.json'), 'w') as f:
            json.dump({'edges': sorted(self.graph.edges), 'mixing_matrix': [
                [self.weight(i, j) if i == j or self.graph.has_edge(i, j) else 0. for j in self.graph]
                for i in self.graph], 'compressor': self.settings.compressor,
                'keep_ratio': self.settings.keep_ratio, 'consensus_stepsize': self.settings.consensus_stepsize,
                'momentum_weight': self.settings.momentum_weight, 'packet_size_bytes': self.packet_size,
                'initialization': 'H0=G0=0; V0=M0=one charged local minibatch gradient'}, f, indent=2)
        with open(os.path.join(self.data_dir, self.MEASUREMENT_PREFIX + '-rounds.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['peer', 'round', 'gradient_ready_us', 'completed_time_us'])
            for c in self.clients:
                if c.bootstrap_done:
                    writer.writerow([c.index, 0, c.bootstrap_ready, c.bootstrap_ready])
                for r, state in sorted(c.rounds.items()):
                    if state['done']:
                        writer.writerow([c.index, r, state['gradient_ready'], state['completed_time']])
