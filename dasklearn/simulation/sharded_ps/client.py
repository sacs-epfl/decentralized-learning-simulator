"""Independent, optimized streaming parameter-server protocol.

Pure optimizer kernels and the historical placement rule are shared with V2;
the message/round state machine is independent. No full-model barrier, extra
server node, all-peer broadcast, or artificial state synchronization is added.
"""
from collections import defaultdict
from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.conflux.sample_manager import SampleManager
from dasklearn.simulation.events import Event, START_ROUND, START_TRAIN
from dasklearn.tasks.references import TaskRef


class ShardedPSClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(lambda: dict(uploads=defaultdict(dict), received={}, reduced=set(), states={}))
        self.owners = None

    def sample(self, r):
        return SampleManager.get_sample(r, list(range(self.simulator.settings.participants)),
                                        self.simulator.settings.sample_size, seed=self.simulator.settings.seed)

    def init_client(self, event):
        self.install_layout(self.simulator.ps_owners)

    def install_layout(self, owners):
        if self.owners is not None:
            raise ValueError('Layout already installed')
        self.owners = tuple(owners)
        s = self.simulator.settings
        owned = [c for c, i in enumerate(owners) if i == self.index]
        if owned and s.server_optimizer != 'fedavg':
            initial = self.create_compute_task('initialize', dict(peer=self.index, round=0, time=self.context.now))
            chunks = self.create_compute_task('chunk', dict(model=TaskRef(initial.name, 0), n=len(owners),
                                                          peer=self.index, round=0, time=self.context.now))
            for c in owned:
                state = self.create_compute_task('conflux_opt_initialize', dict(base=TaskRef(chunks.name, c),
                                                        peer=self.index, round=0, time=self.context.now))
                self.rounds[1]['states'][c] = TaskRef(state.name, 0)
        if self.index in self.sample(1):
            self.simulator.schedule(Event(self.context.now, self.index, START_ROUND, dict(round=1, model=None)))

    def start_round(self, event):
        r = event.data['round']; state = self.rounds[r]
        if state.get('started'):
            return
        state['started'] = True
        self.simulator.schedule(Event(self.context.now, self.index, START_TRAIN, event.data))

    def upload_order(self, r):
        return range(len(self.owners))

    def recipient_order(self, r):
        return self.sample(r+1)

    def finish_train(self, event):
        r = event.data['round']; state = self.rounds[r]
        if state.get('trained'):
            return
        state['trained'] = True
        state['train_us'] = event.data['train_time']
        chunks = self.create_compute_task('chunk', dict(model=event.data['model'], n=len(self.owners),
                                                      peer=self.index, round=r, time=self.context.now))
        for c in self.upload_order(r):
            self.send_chunk(self.owners[c], TaskRef(chunks.name, c), r, c, 'upload')

    def send_chunk(self, peer, payload, r, c, phase):
        self.send(peer, payload, size_bytes=self.chunk_transfer_size(c, len(self.owners)),
                  metadata=dict(round=r, chunk=c, phase=phase))

    def on_incoming_model(self, event):
        meta = event.data['metadata']; r, c = meta['round'], meta['chunk']
        if not 0 <= c < len(self.owners):
            raise ValueError('Invalid parameter interval')
        state = self.rounds[r]; sender = event.data['from']
        if meta['phase'] == 'upload':
            if self.owners[c] != self.index or sender not in self.sample(r):
                raise ValueError('Unauthorized contribution')
            if sender in state['uploads'][c]:
                return
            state['uploads'][c][sender] = event.data['model']
            self.try_reduce(r, c)
        elif meta['phase'] == 'aggregate':
            if sender != self.owners[c] or self.index not in self.sample(r+1):
                raise ValueError('Unauthorized aggregate')
            if c in state['received']:
                return
            state['received'][c] = event.data['model']
            if len(state['received']) != len(self.owners):
                return
            task = self.create_compute_task('reconstruct_from_chunks', dict(
                chunks=[[state['received'][idx]] for idx in range(len(self.owners))],
                peer=self.index, round=r, time=self.context.now))
            model = TaskRef(task.name, 0)
            self.publish_model(r, model)
            self.simulator.completed_rounds.append((r, self.index, self.context.now))
            s = self.simulator.settings
            if s.test_method == 'individual' and s.test_interval > 0 and r % s.test_interval == 0:
                self.create_compute_task('test', dict(model=model, peer=self.index, round=r, time=self.context.now))
            if s.rounds <= 0 or r < s.rounds:
                self.simulator.schedule(Event(self.context.now, self.index, START_ROUND, dict(round=r+1, model=model)))
        else:
            raise ValueError('Unknown parameter-server payload phase')

    def try_reduce(self, r, c):
        state = self.rounds[r]; contributors = self.sample(r)
        if c in state['reduced'] or not set(contributors).issubset(state['uploads'][c]):
            return
        s = self.simulator.settings
        if s.server_optimizer != 'fedavg' and c not in state['states']:
            return
        state['reduced'].add(c)
        self.simulator.contributor_sets[r] = contributors
        inputs = [state['uploads'][c][i] for i in contributors]
        if s.server_optimizer == 'fedavg':
            task = self.create_compute_task('aggregate_chunk', dict(chunks=inputs, peer=self.index, round=r, time=self.context.now))
            result = TaskRef(task.name, 0)
        else:
            result, next_state = self.compute('conflux_opt_update', dict(state=state['states'][c], chunks=inputs),
                params=dict(round=r, chunk_index=c, n=len(self.owners)),
                duration=round(s.server_step_ms*1000), resource='optimizer')
            self.rounds[r+1]['states'][c] = next_state
        for peer in self.recipient_order(r):
            self.send_chunk(peer, result, r, c, 'aggregate')

    def on_message(self, event):
        raise ValueError('The full-participation matched baseline needs no round control exchange')
