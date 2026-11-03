from collections import defaultdict

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.protocol import RoundInbox
from dasklearn.tasks.references import TaskRef, as_ref


class DAMSGradClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(self.new_round)

    def new_round(self):
        return {'started': False, 'done': False, 'inbox': RoundInbox(self.simulator.neighbors(self.index))}

    def init_client(self, event):
        self.simulator.schedule_round(self.index, 1, self.simulator.initial_model, None)

    def start_round(self, event):
        r = event.data['round']
        state = self.rounds[r]
        if state['started']:
            return
        state.update(started=True, base=as_ref(event.data['model']), previous=as_ref(event.data['state']),
                     started_time=self.context.now)
        task = self.create_compute_task('damsgrad_pack', {'model': state['base'], 'state': state['previous'],
            'peer': self.index, 'round': r, 'time': self.context.now})
        state['packet'] = TaskRef(task.name, 0)
        for peer in self.simulator.neighbors(self.index):
            self.send(peer, state['packet'], size_bytes=self.simulator.packet_size, metadata={'round': r})
        self.compute('damsgrad_gradient', {'model': state['base'], 'state': state['previous']},
                     params={'round': r}, duration=self.get_train_time(), on_complete='finish_gradient')

    def finish_gradient(self, event):
        r = event.data['params']['round']
        state = self.rounds[r]
        if 'local_state' in state:
            return
        state.update(local=event.data['outputs'][0], local_state=event.data['outputs'][1],
                     gradient_ready=self.context.now)
        self.try_finish(r)

    def on_incoming_model(self, event):
        r = event.data['metadata']['round']
        if type(r) is not int or r < 1:
            raise ValueError('Invalid Decentralized AMSGrad round')
        packet = as_ref(event.data['model'])
        if packet is None:
            raise ValueError('Decentralized AMSGrad expects a packet reference')
        state = self.rounds[r]
        if state['inbox'].add(event.data['from'], packet):
            self.incoming_counter[event.data['from']] += 1
            self.try_finish(r)

    def try_finish(self, r):
        state = self.rounds[r]
        if not state['inbox'].finish(local_ready='local_state' in state):
            return
        peers = [self.index] + self.simulator.neighbors(self.index)
        task = self.create_compute_task('damsgrad_finish', {'local': state['local'],
            'local_state': state['local_state'], 'state': state['previous'],
            'packets': [state['packet']] + [state['inbox'].messages[p] for p in peers[1:]],
            'weights': [self.simulator.weight(self.index, p) for p in peers],
            'peer': self.index, 'round': r, 'time': self.context.now})
        state.update(model=TaskRef(task.name, 0), state=TaskRef(task.name, 1), done=True,
                     completed_time=self.context.now)
        self.publish_model(r, state['model'])
        settings = self.simulator.settings
        if settings.test_method == 'individual' and settings.test_interval > 0 and r % settings.test_interval == 0:
            self.create_compute_task('test', {'model': state['model'], 'peer': self.index,
                                             'round': r, 'time': self.context.now})
        if settings.stop == 'duration' or settings.rounds <= 0 or r < settings.rounds:
            self.simulator.schedule_round(self.index, r + 1, state['model'], state['state'])
