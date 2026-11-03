"""FSPDA-SA on seeded random undirected edges and shared coordinate masks."""
import csv
import json
import math
import os
import random
from collections import defaultdict

import networkx as nx

from dasklearn.models import create_model
from dasklearn.optimizer.fspda import kept_count, edge_seed
from dasklearn.simulation.events import Event, START_ROUND
from dasklearn.simulation.fspda.client import FSPDAClient
from dasklearn.simulation.simulation import Simulation
from dasklearn.tasks.references import TaskRef


class FSPDASimulation(Simulation):
    CLIENT_CLASS = FSPDAClient

    def __init__(self, settings):
        if settings.participants < 1 or settings.learning.local_steps != 1:
            raise ValueError('FSPDA requires positive participants and local_steps=1')
        if settings.learning.momentum:
            raise ValueError('FSPDA-SA requires plain SGD; set momentum=0')
        if not math.isfinite(settings.learning.learning_rate) or settings.learning.learning_rate <= 0:
            raise ValueError('FSPDA requires a positive finite learning rate')
        if not math.isfinite(settings.learning.weight_decay) or settings.learning.weight_decay < 0:
            raise ValueError('FSPDA requires nonnegative finite weight decay')
        if settings.churn != 'none' or settings.lora:
            raise ValueError('FSPDA requires stable membership and fully trainable models')
        if settings.test_method not in {'individual', 'snapshot'}:
            raise ValueError('FSPDA supports individual or snapshot evaluation')
        if settings.sampling not in {'bernoulli', 'one-edge'}:
            raise ValueError('Unknown FSPDA edge sampling')
        for name in ['keep_ratio', 'edge_probability']:
            value = getattr(settings, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError('FSPDA ' + name + ' must be finite and in (0,1]')
        for name in ['consensus_stepsize', 'dual_weight', 'dual_stepsize']:
            value = getattr(settings, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('FSPDA ' + name + ' must be positive and finite')
        n = settings.participants
        if settings.topology == 'ring':
            graph = nx.cycle_graph(n)
            graph.remove_edges_from(nx.selfloop_edges(graph))
        elif settings.topology == 'complete':
            graph = nx.complete_graph(n)
        elif settings.topology == 'kreg':
            if type(settings.degree) is not int or not 0 <= settings.degree < n or n * settings.degree % 2:
                raise ValueError('Invalid FSPDA regular graph degree')
            for attempt in range(128):
                graph = nx.random_regular_graph(settings.degree, n, seed=settings.seed + 1009 * attempt)
                if nx.is_connected(graph):
                    break
            else:
                raise ValueError('FSPDA requires a connected graph')
        else:
            raise ValueError('Unknown FSPDA topology')
        if not nx.is_connected(graph):
            raise ValueError('FSPDA requires a connected underlying graph')
        model = create_model(settings.dataset, architecture=settings.model, settings=settings)
        parameters = list(model.parameters())
        if (not parameters or any(not p.requires_grad or not p.is_floating_point() or not p.numel() for p in parameters)
                or len({p.dtype for p in parameters}) != 1):
            raise ValueError('FSPDA requires nonempty unfrozen floating parameters with a common dtype')
        self.parameter_count = sum(p.numel() for p in parameters)
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
        self.packet_size = (self.parameter_bytes if settings.keep_ratio == 1 else
            kept_count(self.parameter_count, settings.keep_ratio) * (8 + parameters[0].element_size()))
        self.graph = graph
        self.sampled_edges = {}
        self.ready = defaultdict(dict)
        super().__init__(settings)
        self.register_event_callback(START_ROUND, 'start_round')

    def edges(self, r):
        if r not in self.sampled_edges:
            s = self.settings
            if s.sampling == 'one-edge' and s.participants > 1:
                initiator = (r - 1) % s.participants
                rng = random.Random(edge_seed(s.seed, r, initiator, initiator, 'edge'))
                peer = rng.choice(sorted(self.graph.neighbors(initiator)))
                edges = [tuple(sorted((initiator, peer)))]
            else:
                edges = [(i, j) for i, j in sorted(self.graph.edges) if
                    random.Random(edge_seed(s.seed, r, i, j, 'edge')).random() < s.edge_probability]
            self.sampled_edges[r] = edges
        return self.sampled_edges[r]

    def neighbors(self, peer, r):
        return sorted(j if i == peer else i for i, j in self.edges(r) if peer in (i, j))

    def initialize_clients(self):
        if any(c.get_train_time() <= 0 or c.bw_scheduler.bw_limit <= 0 for c in self.clients):
            raise ValueError('FSPDA requires positive training durations and bandwidths')
        task = self.clients[0].create_compute_task('initialize', {'peer': 0, 'round': 0, 'time': 0})
        self.initial_model = TaskRef(task.name, 0)
        super().initialize_clients()

    def schedule_round(self, peer, r, model, dual):
        data = {'round': r, 'model': model, 'dual': dual}
        if not self.settings.synchronous:
            self.schedule(Event(self.current_time, peer, START_ROUND, data))
            return
        self.ready[r][peer] = data
        if len(self.ready[r]) == self.settings.participants:
            for p, values in sorted(self.ready.pop(r).items()):
                self.schedule(Event(self.current_time, p, START_ROUND, values))

    def save_measurements(self):
        super().save_measurements()
        with open(os.path.join(self.data_dir, 'fspda-network.json'), 'w') as f:
            json.dump({'underlying_edges': sorted(self.graph.edges), 'sampling': self.settings.sampling,
                'edge_probability': self.settings.edge_probability, 'keep_ratio': self.settings.keep_ratio,
                'consensus_stepsize': self.settings.consensus_stepsize, 'dual_weight': self.settings.dual_weight,
                'dual_stepsize': self.settings.dual_stepsize, 'packet_size_bytes': self.packet_size,
                'initialization': 'shared initial model; node dual=0',
                'recurrence': 'simultaneous old-state primal/dual, positive Laplacian; paper (7), (35)-(36)',
                'schedule': 'known shared seed; no discovery or mask negotiation traffic'}, f, indent=2)
        with open(os.path.join(self.data_dir, 'fspda-rounds.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['peer', 'round', 'started_time_us', 'gradient_ready_us', 'completed_time_us', 'neighbors'])
            for c in self.clients:
                for r, state in sorted(c.rounds.items()):
                    if state['done']:
                        writer.writerow([c.index, r, state['started_time'], state['gradient_ready'],
                            state['completed_time'], json.dumps(self.neighbors(c.index, r))])
