"""Decentralized AMSGrad Algorithm 3 on a fixed symmetric Metropolis mixing matrix."""
import csv
import json
import math
import os
from collections import defaultdict

import networkx as nx
import torch

from dasklearn.models import create_model
from dasklearn.simulation.events import Event, START_ROUND
from dasklearn.simulation.damsgrad.client import DAMSGradClient
from dasklearn.simulation.simulation import Simulation
from dasklearn.tasks.references import TaskRef


class DAMSGradSimulation(Simulation):
    CLIENT_CLASS = DAMSGradClient

    def __init__(self, settings):
        if settings.participants < 1 or settings.learning.local_steps != 1:
            raise ValueError('DAMSGrad requires positive participants and local_steps=1')
        if settings.learning.momentum:
            raise ValueError('Decentralized AMSGrad uses beta1; set shared momentum=0')
        if not math.isfinite(settings.learning.learning_rate) or settings.learning.learning_rate <= 0:
            raise ValueError('DAMSGrad requires a positive finite learning rate')
        if not math.isfinite(settings.learning.weight_decay) or settings.learning.weight_decay < 0:
            raise ValueError('DAMSGrad requires nonnegative finite weight decay')
        if settings.churn != 'none' or settings.lora:
            raise ValueError('DAMSGrad requires stable membership and fully trainable models')
        if settings.test_method not in {'individual', 'snapshot'}:
            raise ValueError('DAMSGrad supports individual or snapshot evaluation')
        for name in ['beta1', 'beta2']:
            value = getattr(settings, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError('Decentralized AMSGrad ' + name + ' must be finite and in [0,1)')
        if not math.isfinite(settings.epsilon) or settings.epsilon <= 0:
            raise ValueError('Decentralized AMSGrad epsilon must be positive and finite')
        n = settings.participants
        if settings.topology == 'ring':
            graph = nx.cycle_graph(n)
            graph.remove_edges_from(nx.selfloop_edges(graph))
        elif settings.topology == 'complete':
            graph = nx.complete_graph(n)
        elif settings.topology == 'kreg':
            if type(settings.degree) is not int or not 0 <= settings.degree < n or n * settings.degree % 2:
                raise ValueError('Invalid DAMSGrad regular graph degree')
            for attempt in range(128):
                graph = nx.random_regular_graph(settings.degree, n, seed=settings.seed + 1009 * attempt)
                if nx.is_connected(graph):
                    break
            else:
                raise ValueError('DAMSGrad requires a connected graph')
        else:
            raise ValueError('Unknown DAMSGrad topology')
        if not nx.is_connected(graph):
            raise ValueError('Decentralized AMSGrad requires a connected graph')
        model = create_model(settings.dataset, architecture=settings.model, settings=settings)
        parameters = list(model.parameters())
        if (not parameters or any(not p.requires_grad or not p.is_floating_point() or not p.numel() for p in parameters)
                or len({p.dtype for p in parameters}) != 1):
            raise ValueError('DAMSGrad requires nonempty unfrozen floating parameters with a common dtype')
        floor = torch.tensor(settings.epsilon, dtype=parameters[0].dtype)
        if not torch.isfinite(floor) or floor <= 0:
            raise ValueError('Decentralized AMSGrad epsilon must be representable in parameter dtype')
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
        self.packet_size = 2 * self.parameter_bytes  # Old model and old adaptive consensus tracker.
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
            raise ValueError('DAMSGrad requires positive training durations and bandwidths')
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
        with open(os.path.join(self.data_dir, 'damsgrad-mixing.json'), 'w') as f:
            json.dump({'edges': sorted(self.graph.edges), 'mixing_matrix': [
                [self.weight(i, j) if i == j or self.graph.has_edge(i, j) else 0. for j in self.graph]
                for i in self.graph], 'beta1': self.settings.beta1, 'beta2': self.settings.beta2,
                'epsilon': self.settings.epsilon, 'packet_size_bytes': self.packet_size,
                'initialization': 'm0=v0=0; vhat0=u_half=epsilon',
                'denominator': 'sqrt(max(W @ old_u_half, epsilon)); no bias correction'}, f, indent=2)
        with open(os.path.join(self.data_dir, 'damsgrad-rounds.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['peer', 'round', 'started_time_us', 'gradient_ready_us', 'completed_time_us'])
            for c in self.clients:
                for r, state in sorted(c.rounds.items()):
                    if state['done']:
                        writer.writerow([c.index, r, state['started_time'], state['gradient_ready'], state['completed_time']])
