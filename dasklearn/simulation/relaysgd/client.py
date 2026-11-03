from collections import defaultdict

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.protocol import RoundInbox
from dasklearn.tasks.references import TaskRef, as_ref


class RelaySGDClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(self.new_round)

    def new_round(self):
        return {'started': False, 'done': False, 'inbox': RoundInbox(self.simulator.neighbors(self.index))}

    def init_client(self, event):
        self.simulator.schedule_round(self.index, 1, self.simulator.initial_model, {})

    def start_round(self, event):
        r = event.data['round']
        state = self.rounds[r]
        if state['started']:
            return
        state.update(started=True, base=as_ref(event.data['model']), previous=dict(event.data['previous']))
        self.compute('train', {'model': state['base']}, params={'round': r},
                     duration=self.get_train_time(), on_complete='finish_train')

    def finish_train(self, event):
        r = event.data['params']['round']
        state = self.rounds[r]
        if 'local' in state:
            return
        state.update(local=event.data['outputs'][0], gradient_ready=self.context.now, sent={})
        for peer in self.simulator.neighbors(self.index):
            # Freeze the previous iteration's inputs. Same-round early arrivals
            # never become forward inputs; exclude the recipient to prevent echoes.
            previous = [message for sender, message in sorted(state['previous'].items()) if sender != peer]
            task = self.create_compute_task('relaysgd_forward', {
                'local_model': state['local'], 'packets': [message['sum'] for message in previous],
                'peer': self.index, 'round': r, 'to': peer, 'time': self.context.now})
            packet = {'sum': TaskRef(task.name, 0), 'count': 1 + sum(message['count'] for message in previous)}
            state['sent'][peer] = packet
            self.send(peer, packet, size_bytes=self.simulator.packet_size, metadata={'round': r})
        self.try_finish(r)

    def on_incoming_model(self, event):
        r = event.data['metadata']['round']
        packet = event.data['model']
        if type(r) is not int or r < 1:
            raise ValueError('Invalid RelaySGD round')
        if (not isinstance(packet, dict) or type(packet.get('count')) is not int
                or not 1 <= packet['count'] < self.simulator.settings.participants):
            raise ValueError('Invalid RelaySum contribution count')
        as_ref(packet['sum'])
        state = self.rounds[r]
        if state['inbox'].add(event.data['from'], packet):
            self.incoming_counter[event.data['from']] += 1
            self.try_finish(r)

    def try_finish(self, r):
        state = self.rounds[r]
        if not state['inbox'].finish(local_ready='local' in state):
            return
        packets = [state['inbox'].messages[p] for p in self.simulator.neighbors(self.index)]
        population = 1 + sum(packet['count'] for packet in packets)
        if population > self.simulator.settings.participants:
            raise ValueError('RelaySum population count exceeds the tree population')
        task = self.create_compute_task('relaysgd_finish', {
            'local_model': state['local'], 'packets': [packet['sum'] for packet in packets],
            'counts': [packet['count'] for packet in packets], 'initial_model': self.simulator.initial_model,
            'peer': self.index, 'round': r, 'time': self.context.now})
        state.update(model=TaskRef(task.name, 0), done=True, population=population, completed_time=self.context.now)
        self.publish_model(r, state['model'])
        settings = self.simulator.settings
        if settings.test_method == 'individual' and settings.test_interval > 0 and r % settings.test_interval == 0:
            self.create_compute_task('test', {'model': state['model'], 'peer': self.index,
                                             'round': r, 'time': self.context.now})
        if settings.stop == 'duration' or settings.rounds <= 0 or r < settings.rounds:
            self.simulator.schedule_round(self.index, r + 1, state['model'], state['inbox'].messages)
