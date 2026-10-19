from collections import defaultdict
from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, FINISH_TRAIN
from dasklearn.tasks.task import Task


class QGMClient(BaseClient):
    def __init__(self,simulator,index):
        super().__init__(simulator,index)
        self.rounds=defaultdict(lambda:{'started':False,'done':False,'incoming':{}})

    def task(self,function,data,refs):
        task=Task(Task.generate_name(function),function,data)
        self.add_compute_task(task,inputs=list(dict.fromkeys(r for r in refs if r is not None)))
        return task

    def init_client(self,event):
        self.simulator.schedule_round(self.index,1,self.simulator.initial_model,None)

    def start_round(self,event):
        r=event.data['round'];s=self.rounds[r]
        if s['started']:return
        s.update(started=True,base=event.data['model'],previous=event.data['state'])
        task=self.task('qgm_train',{'model':s['base'],'momentum':s['previous'],
            'round':r,'peer':self.index,'time':event.time},[s['base'],s['previous']])
        duration=self.get_train_time();self.training_until=event.time+duration
        self.simulator.schedule(Event(self.training_until,self.index,FINISH_TRAIN,
            {'round':r,'task':task.name,'train_time':duration}))

    def finish_train(self,event):
        r=event.data['round'];s=self.rounds[r]
        if s.get('packet'):return
        self.compute_time+=event.data['train_time']
        s.update(local=(event.data['task'],0),local_momentum=(event.data['task'],1),packet=(event.data['task'],2))
        for peer in self.simulator.neighbors(self.index):
            self.send_model(peer,s['packet'],{'round':r,'phase':'momentum-model'})
        self.try_finish(r)

    def start_transfer(self,event):
        self.bw_scheduler.add_transfer(self.simulator.clients[event.data['to']].bw_scheduler,
            self.data_transfer_size(self.simulator.packet_size,event.data['metadata']),
            event.data['model'],event.data['metadata'])

    def on_incoming_model(self,event):
        r=event.data['metadata']['round'];sender=event.data['from'];s=self.rounds[r]
        if sender not in self.simulator.neighbors(self.index):raise ValueError('Non-neighbor QGM packet')
        if s['done']:return
        s['incoming'].setdefault(sender,event.data['model']);self.try_finish(r)

    def try_finish(self,r):
        s=self.rounds[r];neighbors=self.simulator.neighbors(self.index)
        if s['done'] or not s.get('packet') or not set(neighbors).issubset(s['incoming']):return
        s['done']=True
        packets=[s['packet']]+[s['incoming'][p] for p in neighbors]
        task=self.task('qgm_finish',{'base':s['base'],'local_model':s['local'],
            'momentum':s['previous'],'local_momentum':s['local_momentum'],'packets':packets,
            'weights':[self.simulator.weight(self.index,p) for p in [self.index]+neighbors],
            'round':r,'peer':self.index,'time':self.simulator.current_time},
            [s['base'],s['local'],s['previous'],s['local_momentum']]+packets)
        s.update(model=(task.name,0),state=(task.name,1),completed_time=self.simulator.current_time)
        self.publish_model(r,s['model'])
        cfg=self.simulator.settings
        if cfg.test_method=='individual' and cfg.test_interval>0 and r%cfg.test_interval==0:
            self.add_compute_task(Task(Task.generate_name('test'),'test',{'model':s['model'],'round':r,
                'peer':self.index,'time':self.simulator.current_time}))
        if cfg.rounds<=0 or r<cfg.rounds:
            self.simulator.schedule_round(self.index,r+1,s['model'],s['state'])
