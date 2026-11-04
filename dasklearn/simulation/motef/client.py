from collections import defaultdict

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.protocol import RoundInbox
from dasklearn.tasks.references import TaskRef, as_ref


class MoTEFClient(BaseClient):
    BOOTSTRAP_OPERATION = 'motef_bootstrap'
    GRADIENT_OPERATION = 'motef_gradient'
    GRADIENT_EVALUATIONS = 1

    def gradient_inputs(self, state):
        return {'model': state['proposal'], 'state': state['previous'],
                'model_packet': state['model_packet']}

    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(self.new_round)
        self.bootstrap_done = False

    def new_round(self):
        return {'started': False, 'done': False, 'inbox': RoundInbox(self.simulator.neighbors(self.index))}

    def init_client(self, event):
        self.compute(self.BOOTSTRAP_OPERATION, {'model': self.simulator.initial_model}, params={'round': 0},
                     duration=self.get_train_time(), on_complete='finish_bootstrap')

    def finish_bootstrap(self, event):
        if self.bootstrap_done:
            return
        self.bootstrap_done = True
        self.bootstrap_ready = self.context.now
        self.bootstrap_model, self.bootstrap_state = event.data['outputs']
        self.simulator.schedule_round(self.index, 1, self.bootstrap_model, self.bootstrap_state)

    def start_round(self, event):
        r = event.data['round']
        state = self.rounds[r]
        if state['started']:
            return
        state.update(started=True, previous=as_ref(event.data['state']), base=as_ref(event.data['model']))
        task = self.create_compute_task('motef_propose', {'model': state['base'], 'state': state['previous'],
            'peer': self.index, 'round': r, 'time': self.context.now})
        state.update(proposal=TaskRef(task.name, 0), model_packet=TaskRef(task.name, 1))
        self.compute(self.GRADIENT_OPERATION, self.gradient_inputs(state), params={'round': r},
                     duration=self.GRADIENT_EVALUATIONS * self.get_train_time(), on_complete='finish_gradient')

    def finish_gradient(self, event):
        r = event.data['params']['round']
        state = self.rounds[r]
        if 'packet' in state:
            return
        model, local_state, packet = event.data['outputs']
        state.update(model=model, local_state=local_state, packet=packet, gradient_ready=self.context.now)
        for peer in self.simulator.neighbors(self.index):
            self.send(peer, packet, size_bytes=self.simulator.packet_size, metadata={'round': r})
        self.try_finish(r)

    def on_incoming_model(self, event):
        r = event.data['metadata']['round']
        if type(r) is not int or r < 1:
            raise ValueError('Invalid MoTEF round')
        packet = as_ref(event.data['model'])
        if packet is None:
            raise ValueError('MoTEF expects a packet reference')
        state = self.rounds[r]
        if state['inbox'].add(event.data['from'], packet):
            self.incoming_counter[event.data['from']] += 1
            self.try_finish(r)

    def try_finish(self, r):
        state = self.rounds[r]
        if not state['inbox'].finish(local_ready='packet' in state):
            return
        peers = [self.index] + self.simulator.neighbors(self.index)
        packets = [state['packet']] + [state['inbox'].messages[p] for p in peers[1:]]
        task = self.create_compute_task('motef_receive', {'state': state['previous'],
            'local_state': state['local_state'], 'packets': packets,
            'weights': [self.simulator.weight(self.index, p) for p in peers],
            'peer': self.index, 'round': r, 'time': self.context.now})
        state.update(state=TaskRef(task.name, 0), done=True, completed_time=self.context.now)
        self.publish_model(r, state['model'])
        settings = self.simulator.settings
        if settings.test_method == 'individual' and settings.test_interval > 0 and r % settings.test_interval == 0:
            self.create_compute_task('test', {'model': state['model'], 'peer': self.index,
                                             'round': r, 'time': self.context.now})
        if settings.stop == 'duration' or settings.rounds <= 0 or r < settings.rounds:
            self.simulator.schedule_round(self.index, r + 1, state['model'], state['state'])
