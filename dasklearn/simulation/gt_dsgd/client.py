from collections import defaultdict

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.protocol import RoundInbox
from dasklearn.tasks.references import TaskRef, as_ref


class GTDSGDClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.rounds = defaultdict(self.new_round)

    def new_round(self):
        peers = self.simulator.neighbors(self.index)
        return {'started': False, 'done': False,
                'tracker_inbox': RoundInbox(peers), 'model_inbox': RoundInbox(peers)}

    def init_client(self, event):
        # Algorithm 1 initializes y_0=0 and g_{-1}=0, not a fresh extra gradient.
        self.simulator.schedule_round(self.index, 1, self.simulator.initial_model, None, None)

    def start_round(self, event):
        r = event.data['round']
        state = self.rounds[r]
        if state['started']:
            return
        state.update(started=True, base=as_ref(event.data['model']),
                     previous_tracker=as_ref(event.data['tracker']),
                     previous_gradient=as_ref(event.data['previous_gradient']))
        self.compute('gt_dsgd_gradient', {'model': state['base'], 'tracker': state['previous_tracker'],
                     'previous_gradient': state['previous_gradient']}, params={'round': r},
                     duration=self.get_train_time(), on_complete='finish_gradient')

    def finish_gradient(self, event):
        r = event.data['params']['round']
        state = self.rounds[r]
        if 'corrected' in state:
            return
        local, gradient, corrected = event.data['outputs']
        state.update(local=local, gradient=gradient, corrected=corrected, gradient_ready=self.context.now)
        self.broadcast(r, 'tracker', corrected)
        self.try_track(r)

    def broadcast(self, r, phase, packet):
        for peer in self.simulator.neighbors(self.index):
            self.send(peer, packet, size_bytes=self.simulator.packet_size,
                      metadata={'round': r, 'phase': phase})

    def on_incoming_model(self, event):
        r = event.data['metadata']['round']
        phase = event.data['metadata']['phase']
        if phase not in {'tracker', 'model'}:
            raise ValueError('Unknown GT-DSGD message phase')
        state = self.rounds[r]
        accepted = state[phase + '_inbox'].add(event.data['from'], as_ref(event.data['model']))
        if not accepted:
            return
        self.incoming_counter[event.data['from']] += 1
        if phase == 'tracker':
            self.try_track(r)
        else:
            self.try_mix(r)

    def phase_inputs(self, state, phase, own):
        peers = self.simulator.neighbors(self.index)
        return {'local_model': state['local'],
                'packets': [own] + [state[phase + '_inbox'].messages[p] for p in peers],
                'weights': [self.simulator.weight(self.index, p) for p in [self.index] + peers]}

    def try_track(self, r):
        state = self.rounds[r]
        if not state['tracker_inbox'].finish(local_ready='corrected' in state):
            return
        data = self.phase_inputs(state, 'tracker', state['corrected'])
        data.update(peer=self.index, round=r, time=self.context.now)
        task = self.create_compute_task('gt_dsgd_track', data)
        state.update(tracker=TaskRef(task.name, 0), proposal=TaskRef(task.name, 1), tracker_ready=self.context.now)
        self.broadcast(r, 'model', state['proposal'])
        self.try_mix(r)

    def try_mix(self, r):
        state = self.rounds[r]
        if not state['model_inbox'].finish(local_ready='proposal' in state):
            return
        data = self.phase_inputs(state, 'model', state['proposal'])
        data.update(peer=self.index, round=r, time=self.context.now)
        task = self.create_compute_task('gt_dsgd_mix', data)
        state.update(model=TaskRef(task.name, 0), done=True, completed_time=self.context.now)
        self.publish_model(r, state['model'])
        settings = self.simulator.settings
        if settings.test_method == 'individual' and settings.test_interval > 0 and r % settings.test_interval == 0:
            self.create_compute_task('test', {'model': state['model'], 'peer': self.index,
                                             'round': r, 'time': self.context.now})
        if settings.stop == 'duration' or settings.rounds <= 0 or r < settings.rounds:
            self.simulator.schedule_round(self.index, r + 1, state['model'], state['tracker'], state['gradient'])
