from collections import defaultdict

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.protocol import RoundInbox
from dasklearn.tasks.references import TaskRef, as_ref
from dasklearn.tasks.task import Task


class QGMClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(self.new_round)

    def new_round(self):
        inbox = RoundInbox(self.simulator.neighbors(self.index))
        return {'started': False, 'done': False, 'incoming': inbox.messages, 'inbox': inbox}

    def init_client(self, event):
        self.simulator.schedule_round(self.index, 1, self.simulator.initial_model, None)

    def start_round(self, event):
        r = event.data['round']
        state = self.rounds[r]
        if state['started']:
            return
        state.update(started=True, base=as_ref(event.data['model']), previous=as_ref(event.data['state']))
        self.compute('qgm_train', {'model': state['base'], 'momentum': state['previous']},
                     duration=self.get_train_time(), params={'round': r}, on_complete='finish_train')

    def finish_train(self, event):
        r = event.data['params']['round']
        state = self.rounds[r]
        if state.get('packet'):
            return
        local, local_momentum, packet = event.data['outputs']
        state.update(local=local, local_momentum=local_momentum, packet=packet)
        for peer in self.simulator.neighbors(self.index):
            self.send(peer, packet, size_bytes=self.simulator.packet_size,
                      metadata={'round': r, 'phase': 'momentum-model'})
        self.try_finish(r)

    def on_incoming_model(self, event):
        r = event.data['metadata']['round']
        state = self.rounds[r]
        state['inbox'].add(event.data['from'], as_ref(event.data['model']))
        self.try_finish(r)

    def try_finish(self, r):
        state = self.rounds[r]
        neighbors = self.simulator.neighbors(self.index)
        if not state['inbox'].finish(local_ready=state.get('packet') is not None):
            return
        state['done'] = True
        packets = [state['packet']] + [state['incoming'][peer] for peer in neighbors]
        task = self.create_compute_task('qgm_finish', {
            'base': state['base'], 'local_model': state['local'],
            'momentum': state['previous'], 'local_momentum': state['local_momentum'],
            'packets': packets,
            'weights': [self.simulator.weight(self.index, peer) for peer in [self.index] + neighbors],
            'round': r, 'peer': self.index, 'time': self.context.now})
        state.update(model=TaskRef(task.name, 0), state=TaskRef(task.name, 1),
                     completed_time=self.context.now)
        self.publish_model(r, state['model'])
        settings = self.simulator.settings
        if settings.test_method == 'individual' and settings.test_interval > 0 and r % settings.test_interval == 0:
            self.create_compute_task('test', {
                'model': state['model'], 'round': r, 'peer': self.index, 'time': self.context.now}, name=Task.generate_name('test'))
        if settings.rounds <= 0 or r < settings.rounds:
            self.simulator.schedule_round(self.index, r + 1, state['model'], state['state'])
