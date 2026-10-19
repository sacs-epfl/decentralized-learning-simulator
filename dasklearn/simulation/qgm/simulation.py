"""QG-DSGDm, Algorithm 1; one gradient step per neighbor exchange."""
import csv
import math
import os
from collections import defaultdict
import networkx as nx
from dasklearn.models import create_model
from dasklearn.simulation.simulation import Simulation
from dasklearn.simulation.events import Event, START_ROUND, FINISH_TRAIN
from dasklearn.simulation.qgm.client import QGMClient
from dasklearn.tasks.task import Task


class QGMSimulation(Simulation):
    CLIENT_CLASS=QGMClient

    def __init__(self,settings):
        if settings.participants<1 or settings.learning.local_steps!=1:
            raise ValueError('QGM implements the paper one-step rule; set local_steps=1')
        if settings.learning.momentum:
            raise ValueError('Use qgm beta for persistent momentum; set shared SGD momentum=0')
        if not math.isfinite(settings.learning.learning_rate) or settings.learning.learning_rate<=0:
            raise ValueError('QGM requires a positive finite learning rate')
        if settings.mode not in {'quasi','local','plain'}:raise ValueError('Unknown QGM mode')
        if settings.mu is None:settings.mu=settings.beta
        if any(not math.isfinite(v) or not 0<=v<1 for v in [settings.beta,settings.mu]):
            raise ValueError('QGM beta/mu must be finite in [0,1)')
        if settings.churn!='none':raise ValueError('QGM currently requires stable membership')
        if settings.test_method not in {'individual','snapshot'}:raise ValueError('Use snapshot or individual evaluation')
        n=settings.participants
        if settings.topology=='ring':
            graph=nx.cycle_graph(n);graph.remove_edges_from(nx.selfloop_edges(graph))
        elif settings.topology=='complete':graph=nx.complete_graph(n)
        elif settings.topology=='kreg':
            if not 0<=settings.degree<n or n*settings.degree%2:
                raise ValueError('Invalid regular graph degree')
            for attempt in range(128):
                graph=nx.random_regular_graph(settings.degree,n,seed=settings.seed+1009*attempt)
                if nx.is_connected(graph):break
            else:raise ValueError('QGM needs a connected graph')
        else:raise ValueError('Unknown QGM topology')
        model=create_model(settings.dataset,architecture=settings.model)
        params=list(model.parameters())
        if not params or any(not p.requires_grad or not p.is_floating_point() for p in params) or len({p.dtype for p in params})!=1:
            raise ValueError('QGM requires unfrozen floating parameters with a common dtype')
        self.packet_size=sum(p.numel()*p.element_size() for p in params)
        self.graph=graph;self.ready=defaultdict(dict)
        super().__init__(settings)
        self.register_event_callback(START_ROUND,'start_round')
        self.register_event_callback(FINISH_TRAIN,'finish_train')

    def neighbors(self,peer):return sorted(self.graph.neighbors(peer))

    def weight(self,peer,other):
        if peer==other:return 1-sum(self.weight(peer,p) for p in self.neighbors(peer))
        return 1/(1+max(self.graph.degree(peer),self.graph.degree(other)))

    def initialize_clients(self):
        if any(c.get_train_time()<=0 or c.bw_scheduler.bw_limit<=0 for c in self.clients):
            raise ValueError('QGM requires positive training duration/bandwidth')
        task=Task(Task.generate_name('initialize'),'initialize',{'peer':0,'round':0,'time':0})
        self.clients[0].add_compute_task(task);self.initial_model=(task.name,0)
        super().initialize_clients()

    def schedule_round(self,peer,r,model,state):
        data={'round':r,'model':model,'state':state}
        if not self.settings.synchronous:self.schedule(Event(self.current_time,peer,START_ROUND,data));return
        self.ready[r][peer]=data
        if len(self.ready[r])==self.settings.participants:
            for p,values in sorted(self.ready.pop(r).items()):self.schedule(Event(self.current_time,p,START_ROUND,values))

    def save_measurements(self):
        super().save_measurements()
        with open(os.path.join(self.data_dir,'qgm-rounds.csv'),'w') as f:
            w=csv.writer(f);w.writerow(['peer','round','time_us'])
            for c in self.clients:w.writerows((c.index,r,s['completed_time']) for r,s in sorted(c.rounds.items()) if s['done'])
