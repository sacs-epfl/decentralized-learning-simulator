"""Numerical-free, one-round forecast using only measured endpoint profiles.

The same port-sharing transport semantics as the simulator, but on independent
virtual nodes whose capacities come from probes. No live/future simulator state
is consulted. Startup optimization wall time is charged to the real protocol.
"""
from bisect import insort
from collections import defaultdict
from statistics import median
from types import SimpleNamespace
from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, FINISH_OUTGOING_TRANSFER, INCOMING_PAYLOAD, PROTOCOL_TIMER
from .planner import DISPATCH_US


class ForecastSettings(SimpleNamespace):
    def to_dict(self):
        return dict(control_message_bytes=self.control_message_bytes,
                    network_latency_ms=self.network_latency_ms, seed=0)


class Forecast:
    def __init__(self, owners, profiles, src, dst, model_bytes, overhead, optimizer_us):
        self.current_time=0; self.events=[]
        # The shared engine currently models one full-duplex endpoint capacity;
        # use the conservative measured rate and median common path latency.
        self.settings=ForecastSettings(control_message_bytes=0,
            network_latency_ms=median(p[3] for p in profiles.values())/1000)
        self.owners=owners; self.src=src; self.dst=dst; self.profiles=profiles
        self.model_bytes=model_bytes; self.overhead=overhead; self.optimizer_us=optimizer_us
        self.done={}; self.clients=[ForecastClient(self,i) for i in profiles]
        for i in src:
            self.schedule(Event(profiles[i][2],i,'upload'))

    def schedule(self,event):
        insort(self.events,(event.time,event.index,event))

    def run(self):
        while self.events and len(self.done)<len(self.dst):
            _,_,event=self.events.pop(0);self.current_time=event.time
            client=self.clients[event.client_id]
            if event.action=='upload':client.upload()
            elif event.action=='update':client.update(event)
            elif event.action==FINISH_OUTGOING_TRANSFER:client.finish_outgoing_transfer(event)
            elif event.action==INCOMING_PAYLOAD:client.on_payload(event)
            elif event.action==PROTOCOL_TIMER:client.on_protocol_timer(event)
            else:raise ValueError(event.action)
        if len(self.done)!=len(self.dst):raise ValueError('Forecast stalled')
        return max(self.done.values())/1e6


class ForecastClient(BaseClient):
    def __init__(self,sim,index):
        super().__init__(sim,index)
        self.bw_scheduler.bw_limit=max(1,min(sim.profiles[index][:2]))
        self.pending=[];self.busy=set();self.uploads=defaultdict(set);self.received=set()
        self.policy_until=0;self.optimizer_until=0

    def upload(self):
        s=self.simulator;m=len(s.src);d=len(s.dst)
        order=sorted(range(len(s.owners)),key=lambda c:(
            -(m/s.profiles[s.owners[c]][1]+(d-int(s.owners[c] in s.dst))/s.profiles[s.owners[c]][0]),c))
        for c in order:self.queue(s.owners[c],c,'upload')

    def queue(self,peer,c,phase):
        if peer==self.index:
            self.deliver(peer,c,phase)
            return
        self.policy_until=max(self.context.now,self.policy_until)+DISPATCH_US
        self.set_timer(self.policy_until-self.context.now,'ready',dict(peer=peer,c=c,phase=phase))

    def ready(self,event):
        self.pending.append(event.data);self.pump()

    def pump(self):
        s=self.simulator
        while True:
            idx=next((i for i,p in enumerate(self.pending) if p['peer'] not in self.busy),None)
            if idx is None:return
            p=self.pending.pop(idx);self.busy.add(p['peer'])
            # Simulated payload represents all wire bytes, avoiding model/task
            # construction and preserving the measured per-message charge.
            count=len(s.owners);b=s.model_bytes//count
            if p['c']==count-1:b+=s.model_bytes%count
            self.bw_scheduler.add_transfer(s.clients[p['peer']].bw_scheduler,
                b+s.overhead,None,dict(chunk=p['c'],phase=p['phase']),delivery_action=INCOMING_PAYLOAD)

    def finish_outgoing_transfer(self,event):
        peer=event.data['transfer'].receiver_scheduler.my_id
        super().finish_outgoing_transfer(event)
        self.busy.remove(peer);self.pump()

    def on_incoming_model(self,event):
        self.deliver(event.data['from'],event.data['metadata']['chunk'],event.data['metadata']['phase'])

    def deliver(self,sender,c,phase):
        s=self.simulator
        if phase=='upload':
            self.uploads[c].add(sender)
            if len(self.uploads[c])==len(s.src):
                self.optimizer_until=max(self.context.now,self.optimizer_until)+s.optimizer_us
                s.schedule(Event(self.optimizer_until,self.index,'update',dict(chunk=c)))
        else:
            self.received.add(c)
            if len(self.received)==len(s.owners):s.done[self.index]=self.context.now

    def update(self,event):
        s=self.simulator
        for peer in sorted(s.dst,key=lambda i:(-(s.model_bytes/s.profiles[i][1]+s.profiles[i][2]/1e6),i)):
            self.queue(peer,event.data['chunk'],'aggregate')


def forecast_score(owners,profiles,samples,model_bytes,overhead,optimizer_us):
    return sum(Forecast(owners,profiles,src,dst,model_bytes,overhead,optimizer_us).run()
               for src,dst in samples)/len(samples)
