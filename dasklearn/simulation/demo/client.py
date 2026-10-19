from collections import defaultdict
from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, START_TRAIN, FINISH_TRAIN
from dasklearn.tasks.task import Task
from dasklearn.tasks.references import TaskRef


class DeMoClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(lambda: {'packets': {}, 'stage': 0, 'started': False, 'done': False})
        self.residual = None


    def init_client(self, event):
        # Snapshot simulations install the common initialize task before events run.
        self.simulator.schedule(Event(event.time, self.index, START_TRAIN, {'round': 1, 'model': (None, 0)}))

    def start_train(self, event):
        r = event.data['round']; state = self.rounds[r]
        if state['started']:return
        state['started'] = True
        model = event.data['model']
        if model[0] is None:
            initial = next((t for t in self.simulator.workflow_dag.tasks.values() if t.func == 'initialize'), None)
            if initial is None:
                initial = self.create_compute_task('initialize', {'peer': self.index, 'round': 0, 'time': 0})
            model = TaskRef(initial.name, 0)
        state['base'] = model
        _, self.residual = self.compute('demo_prepare', {'model': model, 'residual': self.residual},
            params={'round': r}, duration=self.get_train_time(), on_complete='training_finished')

    def finish_train(self, event):
        r = event.data['round']; state = self.rounds[r]
        state['packets'][self.index] = event.data['model']
        state['ready'] = True
        self.advance(r)

    def advance(self, r):
        state = self.rounds[r]; n = self.simulator.settings.participants
        if not state.get('ready') or state['done']:return
        # Stage s sends origin i-s to i+1; receive i-s-1 before advancing.
        while state['stage'] < n - 1:
            stage = state['stage']; origin = (self.index - stage) % n
            if state.get('sent') != stage:
                self.send((self.index + 1) % n, state['packets'][origin], size_bytes=self.simulator.packet_size, metadata={'round': r, 'stage': stage, 'origin': origin})
                state['sent'] = stage
            if (self.index - stage - 1) % n not in state['packets']:return
            state['stage'] += 1
        state['done'] = True
        state['completed_time'] = self.simulator.current_time
        if r not in self.simulator.updates:
            packets = [state['packets'][i] for i in range(n)]
            task = self.create_compute_task('demo_update', {'model': state['base'], 'packets': packets,
                'peer': self.index, 'round': r, 'time': self.simulator.current_time})
            self.simulator.updates[r] = TaskRef(task.name, 0)
        model = self.simulator.updates[r]
        # Memoize the identical collective value, but only publish locally after
        # this worker has received every packet. There is no free broadcast.
        self.publish_model(r, model)
        settings = self.simulator.settings
        if settings.test_method == 'individual' and settings.test_interval > 0 and r % settings.test_interval == 0:
            self.create_compute_task('test', {
                'model': model, 'peer': self.index, 'round': r, 'time': self.simulator.current_time}, name=Task.generate_name('test'))
        if settings.rounds <= 0 or r < settings.rounds:
            self.simulator.schedule(Event(self.simulator.current_time, self.index, START_TRAIN, {'round': r+1, 'model': model}))


    def on_incoming_model(self, event):
        meta = event.data['metadata']; r = meta['round']
        if event.data['from'] != (self.index - 1) % self.simulator.settings.participants:
            raise ValueError('Invalid ring predecessor')
        self.rounds[r]['packets'][meta['origin']] = event.data['model']
        self.advance(r)
