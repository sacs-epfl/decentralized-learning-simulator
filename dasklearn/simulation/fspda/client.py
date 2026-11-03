from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.protocol import RoundInbox
from dasklearn.tasks.references import TaskRef, as_ref


class FSPDAClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = {}

    def get_round(self, r):
        if r not in self.rounds:
            self.rounds[r] = {'started': False, 'done': False,
                              'inbox': RoundInbox(self.simulator.neighbors(self.index, r))}
        return self.rounds[r]

    def init_client(self, event):
        self.simulator.schedule_round(self.index, 1, self.simulator.initial_model, None)

    def start_round(self, event):
        r = event.data['round']
        state = self.get_round(r)
        if state['started']:
            return
        state.update(started=True, base=as_ref(event.data['model']), previous=as_ref(event.data['dual']),
                     started_time=self.context.now, outgoing={})
        # Old model exchange overlaps the charged gradient at that same model.
        for peer in self.simulator.neighbors(self.index, r):
            task = self.create_compute_task('fspda_pack', {'model': state['base'], 'peer': self.index,
                'neighbor': peer, 'round': r, 'time': self.context.now})
            packet = TaskRef(task.name, 0)
            state['outgoing'][peer] = packet
            self.send(peer, packet, size_bytes=self.simulator.packet_size, metadata={'round': r})
        self.compute('fspda_gradient', {'model': state['base']}, params={'round': r},
                     duration=self.get_train_time(), on_complete='finish_gradient')

    def finish_gradient(self, event):
        r = event.data['params']['round']
        state = self.get_round(r)
        if 'gradient' in state:
            return
        state.update(local=event.data['outputs'][0], gradient=event.data['outputs'][1],
                     gradient_ready=self.context.now)
        self.try_finish(r)

    def on_incoming_model(self, event):
        r = event.data['metadata']['round']
        if type(r) is not int or r < 1:
            raise ValueError('Invalid FSPDA round')
        packet = as_ref(event.data['model'])
        if packet is None:
            raise ValueError('FSPDA expects a packet reference')
        state = self.get_round(r)
        if state['inbox'].add(event.data['from'], packet):
            self.incoming_counter[event.data['from']] += 1
            self.try_finish(r)

    def try_finish(self, r):
        state = self.get_round(r)
        if not state['inbox'].finish(local_ready='gradient' in state):
            return
        task = self.create_compute_task('fspda_finish', {'base': state['base'], 'local': state['local'],
            'gradient': state['gradient'], 'dual': state['previous'],
            'packets': [state['inbox'].messages[p] for p in self.simulator.neighbors(self.index, r)],
            'peer': self.index, 'round': r, 'time': self.context.now})
        state.update(model=TaskRef(task.name, 0), dual=TaskRef(task.name, 1), done=True,
                     completed_time=self.context.now)
        self.publish_model(r, state['model'])
        settings = self.simulator.settings
        if settings.test_method == 'individual' and settings.test_interval > 0 and r % settings.test_interval == 0:
            self.create_compute_task('test', {'model': state['model'], 'peer': self.index,
                                             'round': r, 'time': self.context.now})
        if settings.stop == 'duration' or settings.rounds <= 0 or r < settings.rounds:
            self.simulator.schedule_round(self.index, r + 1, state['model'], state['dual'])
