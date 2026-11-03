"""Stable-membership ATC GT-DSGD with same-iteration neighbor exchanges."""
import csv
import math
import os
from collections import defaultdict

import networkx as nx

from dasklearn.models import create_model
from dasklearn.simulation.events import Event, START_ROUND
from dasklearn.simulation.gt_dsgd.client import GTDSGDClient
from dasklearn.simulation.simulation import Simulation
from dasklearn.tasks.references import TaskRef


class GTDSGDSimulation(Simulation):
    CLIENT_CLASS = GTDSGDClient

    def __init__(self, settings):
        if settings.participants < 1 or settings.learning.local_steps != 1:
            raise ValueError('GT-DSGD requires local_steps=1 (one fresh minibatch gradient per iteration)')
        if settings.learning.momentum:
            raise ValueError('GT-DSGD implements plain SGD; set momentum=0')
        if not math.isfinite(settings.learning.learning_rate) or settings.learning.learning_rate <= 0:
            raise ValueError('GT-DSGD requires a positive finite learning rate')
        if not math.isfinite(settings.learning.weight_decay) or settings.learning.weight_decay < 0:
            raise ValueError('GT-DSGD requires nonnegative finite weight decay')
        if settings.churn != 'none' or settings.lora:
            raise ValueError('GT-DSGD requires stable membership and fully trainable models')
        if settings.test_method not in {'individual', 'snapshot'}:
            raise ValueError('GT-DSGD supports individual or snapshot evaluation')
        n = settings.participants
        if settings.topology == 'ring':
            graph = nx.cycle_graph(n)
            graph.remove_edges_from(nx.selfloop_edges(graph))
        elif settings.topology == 'complete':
            graph = nx.complete_graph(n)
        elif settings.topology == 'kreg':
            if type(settings.degree) is not int or not 0 <= settings.degree < n or n * settings.degree % 2:
                raise ValueError('Invalid GT-DSGD regular graph degree')
            for attempt in range(128):
                graph = nx.random_regular_graph(settings.degree, n, seed=settings.seed + 1009 * attempt)
                if nx.is_connected(graph):
                    break
            else:
                raise ValueError('GT-DSGD requires a connected graph')
        else:
            raise ValueError('Unknown GT-DSGD topology')
        model = create_model(settings.dataset, architecture=settings.model, settings=settings)
        parameters = list(model.parameters())
        if (not parameters or any(not p.requires_grad or not p.is_floating_point() for p in parameters)
                or len({p.dtype for p in parameters}) != 1):
            raise ValueError('GT-DSGD requires unfrozen floating parameters with a common dtype')
        self.packet_size = sum(p.numel() * p.element_size() for p in parameters)
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
            raise ValueError('GT-DSGD requires positive training durations and bandwidths')
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
        with open(os.path.join(self.data_dir, 'gt-dsgd-rounds.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['peer', 'round', 'gradient_ready_us', 'tracker_ready_us', 'completed_time_us'])
            for client in self.clients:
                for r, state in sorted(client.rounds.items()):
                    if state['done']:
                        writer.writerow([client.index, r, state['gradient_ready'], state['tracker_ready'],
                                         state['completed_time']])
