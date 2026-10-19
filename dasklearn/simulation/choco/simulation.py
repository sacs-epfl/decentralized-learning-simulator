"""CHOCO-SGD: local SGD, OLD-hat consensus correction, delta compression.

ICML 2019 Algorithm 2; H>1 is an explicit local-SGD block extension. Stable,
fixed undirected graphs use symmetric Metropolis weights including self loops.
"""
import csv
import math
import os
from collections import defaultdict

import networkx as nx

from dasklearn.models import create_model
from dasklearn.optimizer.choco import parameters, payload_bytes
from dasklearn.simulation.choco.client import ChocoClient
from dasklearn.simulation.events import Event, START_ROUND, FINISH_TRAIN
from dasklearn.simulation.simulation import Simulation
from dasklearn.tasks.task import Task
from dasklearn.tasks.references import TaskRef


class ChocoSimulation(Simulation):
    CLIENT_CLASS = ChocoClient

    def __init__(self, settings):
        if settings.participants < 1 or settings.learning.local_steps < 1:
            raise ValueError('CHOCO requires positive participants and local_steps')
        if settings.learning.momentum:
            raise ValueError('CHOCO persistent SGD momentum is not implemented; set momentum=0')
        if settings.churn != 'none':
            raise ValueError('CHOCO currently requires stable membership')
        if settings.test_method not in {'snapshot', 'individual'}:
            raise ValueError('CHOCO supports snapshot or individual evaluation')
        if settings.compressor not in {'topk', 'randomk', 'identity'}:
            raise ValueError('Unknown CHOCO compressor')
        if not math.isfinite(settings.keep_ratio) or not 0 < settings.keep_ratio <= 1:
            raise ValueError('CHOCO keep_ratio must be finite and in (0,1]')
        if not math.isfinite(settings.consensus_stepsize) or not 0 < settings.consensus_stepsize <= 1:
            raise ValueError('CHOCO consensus_stepsize must be finite and in (0,1]')
        n = settings.participants
        if settings.topology == 'ring':
            graph = nx.cycle_graph(n)
            graph.remove_edges_from(nx.selfloop_edges(graph))
        elif settings.topology == 'complete':
            graph = nx.complete_graph(n)
        elif settings.topology == 'kreg':
            if not 0 <= settings.degree < n or n * settings.degree % 2:
                raise ValueError('CHOCO regular graph requires degree < participants and even degree*participants')
            for attempt in range(128):
                graph = nx.random_regular_graph(settings.degree, n, seed=settings.seed + 1009 * attempt)
                if nx.is_connected(graph):
                    break
            else:
                raise ValueError('CHOCO requires a connected regular graph')
        else:
            raise ValueError('Unknown CHOCO topology')
        model = create_model(settings.dataset, architecture=settings.model)
        params = parameters(model)
        if not params or len({p.dtype for p in params}) != 1 or any(not p.is_floating_point() for p in params):
            raise ValueError('CHOCO requires trainable floating parameters with one common dtype')
        if min(p.numel() for p in params) <= 0:
            raise ValueError('CHOCO requires nonempty parameter tensors')
        self.parameter_count = sum(p.numel() for p in params)
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in params)
        self.packet_size = payload_bytes(model, settings.compressor, settings.keep_ratio)
        self.graph = graph
        self.ready = defaultdict(dict)
        super().__init__(settings)
        self.register_event_callback(START_ROUND, 'start_round')
        self.register_event_callback(FINISH_TRAIN, 'finish_train')

    def neighbors(self, peer):
        return sorted(self.graph.neighbors(peer))

    def weight(self, peer, other):
        if peer == other:
            return 1 - sum(self.weight(peer, p) for p in self.neighbors(peer))
        return 1 / (1 + max(self.graph.degree(peer), self.graph.degree(other)))

    def initialize_clients(self):
        if any(c.get_train_time() <= 0 or c.bw_scheduler.bw_limit <= 0 for c in self.clients):
            raise ValueError('CHOCO requires positive training durations and bandwidths')
        task = self.clients[0].create_compute_task('initialize', {'peer': 0, 'round': 0, 'time': 0}, name=Task.generate_name('initialize'))
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
        with open(os.path.join(self.data_dir, 'choco-rounds.csv'), 'w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['peer', 'round', 'time_us'])
            for c in self.clients:
                writer.writerows((c.index, r, s['completed_time']) for r, s in sorted(c.rounds.items()) if s['done'])
        with open(os.path.join(self.data_dir, 'choco-communication.csv'), 'w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['peer', 'neighbor', 'weight', 'parameter_bytes', 'packet_payload_bytes'])
            for peer in self.graph:
                for other in [peer] + self.neighbors(peer):
                    writer.writerow([peer, other, self.weight(peer, other), self.parameter_bytes, self.packet_size])
