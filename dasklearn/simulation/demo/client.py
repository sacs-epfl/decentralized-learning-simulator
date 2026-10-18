from collections import defaultdict
from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import Event, START_TRAIN, FINISH_TRAIN
from dasklearn.tasks.task import Task


class DeMoClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(lambda: {'packets': {}, 'stage': 0, 'started': False, 'done': False})
        self.residual = None

    def task(self, function, data, refs):
        task = Task(Task.generate_name(function), function, data)
        # Explicit immutable value dependencies, including private residual state.
        self.simulator.workflow_dag.tasks[task.name] = task
        for ref in refs:
            if ref is not None:
                source = self.simulator.workflow_dag.tasks[ref[0]]
                source.outputs.append((task, ref[1])); task.inputs.append((source, ref[1]))
        return task

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
                initial = self.task('initialize', {'peer': self.index, 'round': 0, 'time': 0}, [])
            model = (initial.name, 0)
        state['base'] = model
        task = self.task('demo_prepare', {'model': model, 'residual': self.residual, 'peer': self.index,
            'round': r, 'time': self.simulator.current_time}, [model, self.residual])
        self.residual = (task.name, 1)
        train_time = self.get_train_time()
        self.training_until = self.simulator.current_time + train_time
        self.simulator.schedule(Event(self.training_until, self.index, FINISH_TRAIN,
            {'round': r, 'model': (task.name, 0), 'train_time': train_time}))

    def finish_train(self, event):
        self.compute_time += event.data['train_time']
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
                self.send_model((self.index + 1) % n, state['packets'][origin],
                    {'round': r, 'stage': stage, 'origin': origin})
                state['sent'] = stage
            if (self.index - stage - 1) % n not in state['packets']:return
            state['stage'] += 1
        state['done'] = True
        state['completed_time'] = self.simulator.current_time
        if r not in self.simulator.updates:
            packets = [state['packets'][i] for i in range(n)]
            task = self.task('demo_update', {'model': state['base'], 'packets': packets,
                'peer': self.index, 'round': r, 'time': self.simulator.current_time}, [state['base'], *packets])
            self.simulator.updates[r] = (task.name, 0)
        model = self.simulator.updates[r]
        # Memoize the identical collective value, but only publish locally after
        # this worker has received every packet. There is no free broadcast.
        self.publish_model(r, model)
        settings = self.simulator.settings
        if settings.test_method == 'individual' and settings.test_interval > 0 and r % settings.test_interval == 0:
            self.add_compute_task(Task(Task.generate_name('test'), 'test', {
                'model': model, 'peer': self.index, 'round': r, 'time': self.simulator.current_time}))
        if settings.rounds <= 0 or r < settings.rounds:
            self.simulator.schedule(Event(self.simulator.current_time, self.index, START_TRAIN, {'round': r+1, 'model': model}))

    def start_transfer(self, event):
        self.bw_scheduler.add_transfer(self.simulator.clients[event.data['to']].bw_scheduler,
            self.data_transfer_size(self.simulator.packet_size, event.data['metadata']),
            event.data['model'], event.data['metadata'])

    def on_incoming_model(self, event):
        meta = event.data['metadata']; r = meta['round']
        if event.data['from'] != (self.index - 1) % self.simulator.settings.participants:
            raise ValueError('Invalid ring predecessor')
        self.rounds[r]['packets'][meta['origin']] = event.data['model']
        self.advance(r)
