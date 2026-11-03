import math

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.protocol import RoundInbox
from dasklearn.tasks.references import TaskRef, as_ref


class SGPClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = {}

    def get_round(self, r):
        if r not in self.rounds:
            self.rounds[r] = {'started': False, 'done': False,
                              'inbox': RoundInbox(self.simulator.incoming(self.index, r))}
        return self.rounds[r]

    def init_client(self, event):
        self.simulator.schedule_round(self.index, 1, self.simulator.initial_model, None, 1.)

    def start_round(self, event):
        r = event.data['round']
        state = self.get_round(r)
        if state['started']:
            return
        state.update(started=True, base=as_ref(event.data['model']),
                     previous_numerator=as_ref(event.data['numerator']), previous_mass=event.data['mass'])
        weight = self.simulator.weight(self.index, r)
        self.compute('sgp_gradient', {'model': state['base'], 'numerator': state['previous_numerator']},
                     params={'round': r, 'weight': weight}, duration=self.get_train_time(),
                     on_complete='finish_gradient')

    def finish_gradient(self, event):
        r = event.data['params']['round']
        state = self.get_round(r)
        if 'packet' in state:
            return
        local, numerator = event.data['outputs']
        packet = {'numerator': numerator, 'mass': state['previous_mass'] * self.simulator.weight(self.index, r)}
        state.update(local=local, packet=packet, gradient_ready=self.context.now)
        for peer in self.simulator.outgoing(self.index, r):
            self.send(peer, packet, size_bytes=self.simulator.packet_size, metadata={'round': r})
        self.try_finish(r)

    def on_incoming_model(self, event):
        r = event.data['metadata']['round']
        packet = event.data['model']
        if type(r) is not int or r < 1:
            raise ValueError('Invalid SGP round')
        if (not isinstance(packet, dict) or type(packet.get('mass')) is not float
                or not math.isfinite(packet['mass']) or packet['mass'] <= 0):
            raise ValueError('Invalid SGP PushSum mass')
        as_ref(packet['numerator'])
        state = self.get_round(r)
        if state['inbox'].add(event.data['from'], packet):
            self.incoming_counter[event.data['from']] += 1
            self.try_finish(r)

    def try_finish(self, r):
        state = self.get_round(r)
        if not state['inbox'].finish(local_ready='packet' in state):
            return
        packets = [state['packet']] + [state['inbox'].messages[p] for p in self.simulator.incoming(self.index, r)]
        mass = sum(packet['mass'] for packet in packets)
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError('Invalid SGP PushSum mass')
        task = self.create_compute_task('sgp_mix', {'local_model': state['local'],
            'packets': [packet['numerator'] for packet in packets], 'mass': mass,
            'peer': self.index, 'round': r, 'time': self.context.now})
        state.update(model=TaskRef(task.name, 0), numerator=TaskRef(task.name, 1), mass=mass,
                     done=True, completed_time=self.context.now)
        self.publish_model(r, state['model'])
        settings = self.simulator.settings
        if settings.test_method == 'individual' and settings.test_interval > 0 and r % settings.test_interval == 0:
            self.create_compute_task('test', {'model': state['model'], 'peer': self.index,
                                             'round': r, 'time': self.context.now})
        if settings.stop == 'duration' or settings.rounds <= 0 or r < settings.rounds:
            self.simulator.schedule_round(self.index, r + 1, state['model'], state['numerator'], mass)
