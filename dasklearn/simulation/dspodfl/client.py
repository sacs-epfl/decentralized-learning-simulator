from collections import defaultdict

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, START_ROUND, START_TRAIN
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS


class DSpodFLClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator,index)
        self.iterations=defaultdict(lambda:{'incoming':{},'started':False,'trained':False,'updated':False})

    def init_client(self,event):
        self.simulator.schedule(Event(event.time,self.index,START_ROUND,{'round':1,'model':self.simulator.initial_model}))

    def start_round(self,event):
        r=event.data['round'];s=self.iterations[r]
        if s['started']:return
        s.update(started=True,base=event.data['model'],start=self.simulator.current_time)
        s['neighbors']=self.simulator.neighbors(self.index,r)
        s['sgd']=self.simulator.performs_sgd(self.index,r)
        # Old-model exchange is independent of whether this node trains.
        for peer in s['neighbors']:
            self.send_model(peer,s['base'],{'round':r,'phase':'dspod-old-model'})
        if s['sgd']:
            self.simulator.schedule(Event(self.simulator.current_time,self.index,START_TRAIN,event.data))
        else:
            s.update(trained=True,local=s['base'])
            self.try_update(r)

    def start_transfer(self,event):
        self.bw_scheduler.add_transfer(self.simulator.clients[event.data['to']].bw_scheduler,
            self.data_transfer_size(self.simulator.model_size,event.data['metadata']),
            event.data['model'],event.data['metadata'])

    def finish_train(self,event):
        r=event.data['round'];s=self.iterations[r]
        if s['trained']:return
        s.update(trained=True,local=event.data['model'])
        self.compute_time+=event.data['train_time']
        self.try_update(r)

    def on_incoming_model(self,event):
        r=event.data['metadata']['round']
        self.iterations[r]['incoming'][event.data['from']]=event.data['model']
        self.try_update(r)

    def try_update(self,r):
        s=self.iterations[r]
        if not s['started'] or not s['trained'] or s['updated']:return
        if not set(s['neighbors']).issubset(s['incoming']):return
        s['updated']=True
        weights=[self.simulator.weight(self.index,p) for p in s['neighbors']]
        if not weights:
            model=s['local']
        else:
            # Eq. (2): local + sum_j r_ij*(OLD_neighbor - OLD_self).
            # H=1 is the paper update; H>1 is an explicitly labeled block-delta extension.
            models=([s['local'],s['base']] if s['sgd'] else [s['base']])
            coefficients=([1.,-sum(weights)] if s['sgd'] else [1-sum(weights)])
            models += [s['incoming'][p] for p in s['neighbors']]
            coefficients += weights
            task=Task(Task.generate_name('dspod_update'),'aggregate',{'models':models,'weights':coefficients,
                'round':r,'peer':self.index,'time':self.simulator.current_time})
            self.add_compute_task(task)
            model=(task.name,0)
        self.publish_model(r,model)
        self.simulator.completed_iterations.append((r,self.index,self.simulator.current_time,int(s['sgd']),len(weights)))
        if self.simulator.settings.test_method=='individual' and self.simulator.settings.test_interval>0 and r%self.simulator.settings.test_interval==0:
            self.add_compute_task(Task(Task.generate_name('test'),'test',{
                'model':model,'round':r,'peer':self.index,'time':self.simulator.current_time}))
        if self.simulator.settings.rounds<=0 or r<self.simulator.settings.rounds:
            time=max(self.simulator.current_time,s['start']+max(1,int(self.simulator.settings.minimum_round_ms*MICROSECONDS/1000)))
            self.simulator.schedule(Event(time,self.index,START_ROUND,{'round':r+1,'model':model}))
