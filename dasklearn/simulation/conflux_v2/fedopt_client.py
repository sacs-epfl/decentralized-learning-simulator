"""Conflux common-round optimizer shards, including charged ownership handoffs."""
from dasklearn.simulation.conflux_v2.client import ConfluxV2Client
from dasklearn.simulation.conflux_v2.optimizer import state_payload_bytes
from dasklearn.tasks.references import TaskRef


class ConfluxFedOptClient(ConfluxV2Client):
    def init_client(self,event):
        super().init_client(event)
        _,owners=self.layout(1)
        owned=[idx for idx,owner in enumerate(owners) if owner==self.index]
        if not owned:return
        initial=self.create_compute_task('initialize',dict(peer=self.index,round=0,time=0))
        chunks=self.create_compute_task('chunk',dict(model=TaskRef(initial.name,0),n=len(owners),
                                                     peer=self.index,round=0,time=0))
        for idx in owned:
            state=self.create_compute_task('conflux_opt_initialize',dict(base=TaskRef(chunks.name,idx),
                peer=self.index,round=0,time=0))
            self.rounds[1].setdefault('optimizer_states',{})[idx]=TaskRef(state.name,0)

    def on_incoming_model(self,event):
        meta=event.data['metadata']
        if meta['phase']!='optimizer_state':
            return super().on_incoming_model(event)
        round_nr,idx=meta['round'],meta['chunk']
        if round_nr<2 or meta['source_round']!=round_nr-1:
            raise ValueError('Invalid optimizer handoff round')
        if self.layout(round_nr)[1][idx]!=self.index or self.layout(round_nr-1)[1][idx]!=event.data['from']:
            raise ValueError('Optimizer handoff has invalid ownership')
        states=self.rounds[round_nr].setdefault('optimizer_states',{})
        if idx in states:raise ValueError('Duplicate optimizer state handoff')
        states[idx]=event.data['model']
        self.try_reduce(round_nr)

    def try_reduce(self,round_nr):
        state=self.rounds[round_nr]
        if not state.get('plan_received'):return
        _,owners=self.layout(round_nr)
        settings=self.simulator.settings
        for idx,owner in enumerate(owners):
            if owner!=self.index or idx in state['reduced']:continue
            opt=state.get('optimizer_states',{}).get(idx)
            if opt is None or not set(state['plan']).issubset(state['uploads'][idx]):continue
            state['reduced'].add(idx)
            updated,new_state=self.compute('conflux_opt_update',
                dict(state=opt,chunks=[state['uploads'][idx][peer] for peer in state['plan']]),
                params=dict(round=round_nr,chunk_index=idx,n=len(owners)),
                duration=round(settings.server_step_ms*1000),resource='optimizer')
            # The next optimizer owner needs x_t as well as its moment history.
            # State is never replicated to trainers or to unrelated shard owners.
            if settings.rounds<=0 or round_nr<settings.rounds:
                next_owner=self.layout(round_nr+1)[1][idx]
                if next_owner==self.index:
                    self.rounds[round_nr+1].setdefault('optimizer_states',{})[idx]=new_state
                else:
                    payload=new_state
                    if settings.state_bits==8:
                        payload=self.compute('conflux_opt_pack',dict(state=new_state),
                            params=dict(round=round_nr,chunk_index=idx),
                            duration=round(settings.state_pack_ms*1000),resource='optimizer')
                    elements=self.simulator.optimizer_elements//len(owners)
                    if idx==len(owners)-1:elements+=self.simulator.optimizer_elements%len(owners)
                    size=state_payload_bytes(elements,settings.server_optimizer,settings.state_bits,settings.state_block_size)
                    self.simulator.optimizer_handoffs.append((round_nr,idx,self.index,next_owner,size))
                    self.send(next_owner,payload,size_bytes=size,metadata=dict(round=round_nr+1,
                        chunk=idx,phase='optimizer_state',source_round=round_nr))
            for peer in self.sample(round_nr+1):
                self.send_chunk(peer,updated,round_nr,idx,'aggregate')
