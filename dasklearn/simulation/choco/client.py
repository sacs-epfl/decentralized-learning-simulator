from collections import defaultdict

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, START_TRAIN
from dasklearn.tasks.task import Task
from dasklearn.tasks.references import TaskRef, as_ref
from dasklearn.simulation.protocol import RoundInbox


class ChocoClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(self.new_round)

    def new_round(self):
        inbox = RoundInbox(self.simulator.neighbors(self.index))
        return {'incoming': inbox.messages, 'inbox': inbox, 'started': False, 'done': False}

    def init_client(self, event):
        self.simulator.schedule_round(self.index, 1, self.simulator.initial_model, None)

    def start_round(self, event):
        r = event.data['round']
        state = self.rounds[r]
        if state['started']:
            return
        state.update(started=True, previous=as_ref(event.data['state']))
        self.simulator.schedule(Event(event.time, self.index, START_TRAIN, event.data))

    def finish_train(self, event):
        r = event.data['round']
        state = self.rounds[r]
        if state.get('packet') is not None:
            return
        task = self.create_compute_task('choco_prepare', {
            'model': as_ref(event.data['model']), 'state': state['previous'], 'round': r,
            'peer': self.index, 'time': event.time})
        state.update(model=TaskRef(task.name, 0), packet=TaskRef(task.name, 1))
        for peer in self.simulator.neighbors(self.index):
            self.send(peer, state['packet'], size_bytes=self.simulator.packet_size,
                      metadata={'round': r, 'phase': 'choco-delta'})
        self.try_finish(r)

    def on_incoming_model(self, event):
        r = event.data['metadata']['round']
        sender = event.data['from']
        state = self.rounds[r]
        state['inbox'].add(sender, as_ref(event.data['model']))
        self.try_finish(r)

    def try_finish(self, r):
        state = self.rounds[r]
        neighbors = self.simulator.neighbors(self.index)
        if not state['inbox'].finish(local_ready=state.get('packet') is not None):
            return
        state['done'] = True
        packets = [state['packet']] + [state['incoming'][p] for p in neighbors]
        task = self.create_compute_task('choco_receive', {
            'state': state['previous'], 'packets': packets,
            'size': self.simulator.parameter_count,
            'weights': [self.simulator.weight(self.index, p) for p in [self.index] + neighbors],
            'round': r, 'peer': self.index, 'time': self.simulator.current_time})
        state['state'] = TaskRef(task.name, 0)
        state['completed_time'] = self.simulator.current_time
        self.publish_model(r, state['model'])
        s = self.simulator.settings
        if s.test_method == 'individual' and s.test_interval > 0 and r % s.test_interval == 0:
            self.create_compute_task('test', {
                'model': state['model'], 'round': r, 'peer': self.index, 'time': self.simulator.current_time}, name=Task.generate_name('test'))
        if s.rounds <= 0 or r < s.rounds:
            self.simulator.schedule_round(self.index, r + 1, state['model'], state['state'])
