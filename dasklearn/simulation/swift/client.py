from dasklearn.simulation.client import BaseClient
from dasklearn.tasks.references import TaskRef, as_ref


class SwiftClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.mailbox = {}
        self.rounds = {}

    def init_client(self, event):
        # All clients and all cache slots share the prescribed initial model.
        self.mailbox = {peer: (0, self.simulator.initial_packet)
                        for peer in self.simulator.neighbors(self.index)}
        self.start_step(1, self.simulator.initial_model)

    def start_step(self, r, model):
        self.rounds[r] = {'base': model, 'started_time': self.context.now}
        # Algorithm 1 line 7: broadcast pre-update parameters at every step,
        # including steps outside C_s. Transfers overlap the gradient compute.
        packet = self.create_compute_task('swift_pack', {'model': model, 'peer': self.index,
                                                        'round': r, 'time': self.context.now})
        for peer in self.simulator.neighbors(self.index):
            self.send(peer, TaskRef(packet.name, 0), size_bytes=self.simulator.packet_size,
                      metadata={'version': r - 1})
        self.compute('compute_gradient', {'model': model}, params={'round': r},
                     duration=self.get_train_time(), on_complete='finish_gradient')

    def on_incoming_model(self, event):
        sender = event.data['from']
        version = event.data['metadata']['version']
        if sender in self.mailbox and version > self.mailbox[sender][0]:
            self.mailbox[sender] = (version, as_ref(event.data['model']))

    def finish_gradient(self, event):
        r = event.data['params']['round']
        state = self.rounds[r]
        averaged = r % self.simulator.settings.averaging_interval == 0
        peers = self.simulator.neighbors(self.index) if averaged else []
        task = self.create_compute_task('swift_update', {
            'gradient_model': event.data['outputs'][0],
            'packets': [self.mailbox[p][1] for p in peers],
            'weights': [float(self.simulator.weights[self.index, p]) for p in peers],
            'peer': self.index, 'round': r, 'time': self.context.now})
        model = TaskRef(task.name, 0)
        state.update(model=model, completed_time=self.context.now, averaged=averaged,
                     versions={p: self.mailbox[p][0] for p in peers})
        self.publish_model(r, model)
        settings = self.simulator.settings
        if settings.test_method == 'individual' and settings.test_interval > 0 and r % settings.test_interval == 0:
            self.create_compute_task('test', {'model': model, 'peer': self.index,
                                             'round': r, 'time': self.context.now})
        if settings.stop == 'duration' or settings.rounds <= 0 or r < settings.rounds:
            self.start_step(r + 1, model)
