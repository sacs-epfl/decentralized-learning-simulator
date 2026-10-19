from collections import defaultdict

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, START_TRAIN
from dasklearn.tasks.task import Task


class ChocoClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(lambda: {'incoming': {}, 'started': False, 'done': False})

    def task(self, function, data, refs):
        task = Task(Task.generate_name(function), function, data)
        # Custom functions explicitly include private state and packet dependencies.
        refs = list(dict.fromkeys(ref for ref in refs if ref is not None))
        self.add_compute_task(task, inputs=refs)
        return task

    def init_client(self, event):
        self.simulator.schedule_round(self.index, 1, self.simulator.initial_model, None)

    def start_round(self, event):
        r = event.data['round']
        state = self.rounds[r]
        if state['started']:
            return
        state.update(started=True, previous=event.data['state'])
        self.simulator.schedule(Event(event.time, self.index, START_TRAIN, event.data))

    def finish_train(self, event):
        r = event.data['round']
        state = self.rounds[r]
        if state.get('packet') is not None:
            return
        self.compute_time += event.data['train_time']
        task = self.task('choco_prepare', {
            'model': event.data['model'], 'state': state['previous'], 'round': r,
            'peer': self.index, 'time': event.time}, [event.data['model'], state['previous']])
        state.update(model=(task.name, 0), packet=(task.name, 1))
        for peer in self.simulator.neighbors(self.index):
            self.send_model(peer, state['packet'], {'round': r, 'phase': 'choco-delta'})
        self.try_finish(r)

    def start_transfer(self, event):
        self.bw_scheduler.add_transfer(self.simulator.clients[event.data['to']].bw_scheduler,
            self.data_transfer_size(self.simulator.packet_size, event.data['metadata']),
            event.data['model'], event.data['metadata'])

    def on_incoming_model(self, event):
        r = event.data['metadata']['round']
        sender = event.data['from']
        if sender not in self.simulator.neighbors(self.index):
            raise ValueError('CHOCO received a packet from a non-neighbor')
        state = self.rounds[r]
        if state['done']:
            return
        state['incoming'].setdefault(sender, event.data['model'])
        self.try_finish(r)

    def try_finish(self, r):
        state = self.rounds[r]
        neighbors = self.simulator.neighbors(self.index)
        if state['done'] or not state.get('packet') or not set(neighbors).issubset(state['incoming']):
            return
        state['done'] = True
        packets = [state['packet']] + [state['incoming'][p] for p in neighbors]
        task = self.task('choco_receive', {
            'state': state['previous'], 'packets': packets,
            'size': self.simulator.parameter_count,
            'weights': [self.simulator.weight(self.index, p) for p in [self.index] + neighbors],
            'round': r, 'peer': self.index, 'time': self.simulator.current_time},
            [state['previous']] + packets)
        state['state'] = (task.name, 0)
        state['completed_time'] = self.simulator.current_time
        self.publish_model(r, state['model'])
        s = self.simulator.settings
        if s.test_method == 'individual' and s.test_interval > 0 and r % s.test_interval == 0:
            self.add_compute_task(Task(Task.generate_name('test'), 'test', {
                'model': state['model'], 'round': r, 'peer': self.index, 'time': self.simulator.current_time}))
        if s.rounds <= 0 or r < s.rounds:
            self.simulator.schedule_round(self.index, r + 1, state['model'], state['state'])
