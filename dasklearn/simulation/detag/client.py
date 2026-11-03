from collections import defaultdict

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.protocol import RoundInbox
from dasklearn.tasks.references import TaskRef, as_ref


class DeTAGClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(self.new_round)

    def new_round(self):
        return {'started': False, 'done': False, 'steps': defaultdict(self.new_step)}

    def new_step(self):
        return {'started': False, 'done': False, 'inbox': RoundInbox(self.simulator.neighbors(self.index))}

    def init_client(self, event):
        self.simulator.schedule_round(self.index, 1, self.simulator.initial_model, None, None)

    def start_round(self, event):
        phase = event.data['round']
        state = self.rounds[phase]
        if state['started']:
            return
        state.update(started=True, base=as_ref(event.data['model']),
                     previous_gradient=as_ref(event.data['previous_gradient']))
        task = self.create_compute_task('detag_packet', {'model': state['base'],
            'tracker': as_ref(event.data['tracker']), 'peer': self.index, 'round': phase, 'time': self.context.now})
        packet = TaskRef(task.name, 0)
        # Restart the accelerated recurrence at every phase: z_{-1}=z_0.
        self.start_step(phase, 1, packet, packet, state['base'], None)

    def start_step(self, phase, step, packet, previous, local_model, accumulator):
        inner = self.rounds[phase]['steps'][step]
        inner.update(started=True, packet=packet, previous=previous, started_time=self.context.now)
        for peer in self.simulator.neighbors(self.index):
            self.send(peer, packet, size_bytes=self.simulator.packet_size,
                      metadata={'phase': phase, 'step': step})
        oracle_round = (phase - 1) * self.simulator.settings.phase_length + step
        self.compute('detag_gradient', {'model': local_model, 'accumulator': accumulator},
            params={'round': oracle_round, 'phase': phase, 'step': step},
            duration=self.get_train_time(), on_complete='finish_gradient')
        self.advance(phase, step)

    def finish_gradient(self, event):
        phase, step = event.data['params']['phase'], event.data['params']['step']
        inner = self.rounds[phase]['steps'][step]
        if 'local_model' in inner:
            return
        inner.update(local_model=event.data['outputs'][0], gradient=event.data['outputs'][1],
                     gradient_ready=self.context.now)
        self.advance(phase, step)

    def on_incoming_model(self, event):
        phase, step = event.data['metadata']['phase'], event.data['metadata']['step']
        if type(phase) is not int or phase < 1 or type(step) is not int or not 1 <= step <= self.simulator.settings.phase_length:
            raise ValueError('Invalid DeTAG phase/step')
        inner = self.rounds[phase]['steps'][step]
        if inner['inbox'].add(event.data['from'], as_ref(event.data['model'])):
            self.incoming_counter[event.data['from']] += 1
            self.advance(phase, step)

    def advance(self, phase, step):
        state = self.rounds[phase]
        inner = state['steps'][step]
        if inner['inbox'].finish(local_ready=inner['started']):
            peers = self.simulator.neighbors(self.index)
            task = self.create_compute_task('detag_gossip', {
                'packets': [inner['packet']] + [inner['inbox'].messages[p] for p in peers],
                'previous': inner['previous'],
                'weights': [float(self.simulator.weights[self.index, p]) for p in [self.index] + peers],
                'peer': self.index, 'round': phase, 'step': step, 'time': self.context.now})
            inner.update(mixed=TaskRef(task.name, 0), gossip_ready=self.context.now)
        if inner['done'] or 'mixed' not in inner or 'local_model' not in inner:
            return
        inner.update(done=True, completed_time=self.context.now)
        settings = self.simulator.settings
        if step < settings.phase_length:
            self.start_step(phase, step + 1, inner['mixed'], inner['packet'],
                            inner['local_model'], inner['gradient'])
            return
        task = self.create_compute_task('detag_finish', {
            'packet': inner['mixed'], 'local_model': inner['local_model'],
            'gradient': inner['gradient'], 'previous_gradient': state['previous_gradient'],
            'peer': self.index, 'round': phase, 'time': self.context.now})
        state.update(done=True, model=TaskRef(task.name, 0), tracker=TaskRef(task.name, 1),
                     gradient=inner['gradient'], completed_time=self.context.now)
        self.publish_model(phase, state['model'])
        if settings.test_method == 'individual' and settings.test_interval > 0 and phase % settings.test_interval == 0:
            self.create_compute_task('test', {'model': state['model'], 'peer': self.index,
                                             'round': phase, 'time': self.context.now})
        if settings.stop == 'duration' or settings.rounds <= 0 or phase < settings.rounds:
            self.simulator.schedule_round(self.index, phase + 1, state['model'], state['tracker'], state['gradient'])
