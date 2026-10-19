"""Equation (2)/Algorithm 1, with an explicit H-step local-delta extension.

The DAG executes causal iterations without an artificial global barrier. A
symmetric link mask is shared by both endpoints; old same-iteration models are
exchanged. Metropolis weights use the physical graph's degrees, not the active
degree. No renormalization of surviving links is performed.
"""
import csv
import math
import os
import random

import networkx as nx

from dasklearn.simulation.simulation import Simulation
from dasklearn.simulation.events import START_ROUND, FINISH_TRAIN
from dasklearn.simulation.dspodfl.client import DSpodFLClient
from dasklearn.tasks.task import Task
from dasklearn.tasks.references import TaskRef


class DSpodFLSimulation(Simulation):
    CLIENT_CLASS = DSpodFLClient

    def __init__(self, settings):
        if settings.churn != 'none':
            raise ValueError('DSpodFL currently requires stable membership; sporadic work is not node failure recovery')
        if settings.learning.momentum or settings.learning.weight_decay:
            raise ValueError('DSpodFL uses plain SGD; set momentum and weight_decay to zero')
        if settings.test_method == 'global':
            raise ValueError('Use snapshot or individual evaluation for DSpodFL')
        if not 0 <= settings.degree < settings.participants or settings.degree*settings.participants % 2:
            raise ValueError('A regular graph requires degree < participants and even degree*participants')
        if settings.participants > 1 and settings.degree < 2:
            raise ValueError('Use degree >= 2 for a connected multi-node graph')
        if settings.probability_mode not in {'capacity','constant','beta'}:
            raise ValueError('Unknown DSpodFL probability mode')
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in [settings.sgd_probability,settings.communication_probability]):
            raise ValueError('DSpodFL probabilities must be finite and in [0,1]')
        if not math.isfinite(settings.minimum_round_ms) or settings.minimum_round_ms <= 0:
            raise ValueError('DSpodFL requires a positive minimum iteration duration')
        super().__init__(settings)
        for attempt in range(128):
            graph=nx.random_regular_graph(settings.degree,settings.participants,seed=settings.seed+attempt*1009)
            if nx.is_connected(graph):break
        else:raise ValueError('Could not generate a connected physical graph')
        self.graph=graph
        self.node_probabilities={};self.link_probabilities={}
        self.completed_iterations=[]
        self.register_event_callback(START_ROUND,'start_round')
        self.register_event_callback(FINISH_TRAIN,'finish_train')

    def initialize_clients(self):
        initial=self.clients[0].create_compute_task('initialize', {'peer':0,'round':0,'time':0}, name=Task.generate_name('initialize'))
        self.initial_model=TaskRef(initial.name, 0)
        bws=[c.bw_scheduler.bw_limit for c in self.clients]
        durations=[c.get_train_time() for c in self.clients]
        if min(bws)<=0 or min(durations)<=0:
            raise ValueError('DSpodFL requires positive bandwidth and training durations')
        mean_bw=sum(bws)/len(bws);mean_duration=sum(durations)/len(durations)
        rng=random.Random(self.settings.seed+78233)
        for i,duration in enumerate(durations):
            p=self.settings.sgd_probability
            if self.settings.probability_mode=='capacity':p*=min(1,mean_duration/duration)
            elif self.settings.probability_mode=='beta':p=rng.betavariate(.8,.8)
            self.node_probabilities[i]=p
        for i,j in sorted(tuple(sorted(e)) for e in self.graph.edges):
            p=self.settings.communication_probability
            if self.settings.probability_mode=='capacity':p*=min(1,min(bws[i],bws[j])/mean_bw)
            elif self.settings.probability_mode=='beta':p=rng.betavariate(.8,.8)
            self.link_probabilities[i,j]=p
        super().initialize_clients()

    def performs_sgd(self, peer, iteration):
        return random.Random(self.settings.seed+1000003*iteration+7919*peer).random()<self.node_probabilities[peer]

    def neighbors(self, peer, iteration):
        active=[]
        for other in sorted(self.graph.neighbors(peer)):
            i,j=sorted((peer,other))
            draw=random.Random(self.settings.seed+2147483647*iteration+1000003*i+9176*j).random()
            if draw<self.link_probabilities[i,j]:active.append(other)
        return active

    def weight(self, peer, other):
        return 1/(1+max(self.graph.degree(peer),self.graph.degree(other)))

    def save_measurements(self):
        super().save_measurements()
        with open(os.path.join(self.data_dir,'dspodfl_probabilities.csv'),'w') as stream:
            writer=csv.writer(stream);writer.writerow(['kind','client','neighbor','probability','weight'])
            for peer,p in sorted(self.node_probabilities.items()):writer.writerow(['sgd',peer,'',p,''])
            for (i,j),p in sorted(self.link_probabilities.items()):writer.writerow(['link',i,j,p,self.weight(i,j)])
        with open(os.path.join(self.data_dir,'dspodfl_iterations.csv'),'w') as stream:
            writer=csv.writer(stream);writer.writerow(['iteration','client','time_us','sgd','active_neighbors'])
            writer.writerows(self.completed_iterations)
