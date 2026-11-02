"""Measured bootstrap and sample-aware direct-only transfer queues."""
from collections import defaultdict
from math import ceil
from time import perf_counter
from dasklearn.simulation.events import Event
from .client import ShardedPSClient
from .planner import (pairings, summarize_probes, choose_layout, RouteFeedback,
                      DISPATCH_US, CANDIDATE_US)


class AutoDirectClient(ShardedPSClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.profiles = None
        self.pending = []
        self.busy_peers = set()
        self.route_feedback = RouteFeedback()
        self.policy_until = 0
        self.early = []
        self.counter = 0
        self.matches = pairings(simulator.settings.participants, simulator.settings.seed)
        self.probe_records = []
        self.training_profiles = {}
        self.phase_records = defaultdict(list)
        self.phase = 0
        self.planning = False

    def init_client(self, event):
        # A discarded round-0 local training probe, with its own deterministic
        # RNG seed. The ordinary training kernel resets task RNG/data sampling.
        self.compute('train', dict(model=None), duration=self.get_train_time(),
                     params=dict(round=0), on_complete='calibration_finished')
        if self.index == 0:
            self.set_timer(0, 'begin_probe')

    def calibration_finished(self, event):
        self.send_control(0, 'training_profile', dict(duration=event.data['duration']))

    def begin_probe(self, event=None):
        if self.phase == 2*len(self.matches):
            self.maybe_plan()
            return
        # Model-derived payload, two sizes cancel fixed latency. No capacity
        # values or future simulator event timestamps are inspected.
        size = max(1, ceil(self.simulator.model_size/self.simulator.settings.participants))
        if self.phase % 2:
            size *= 2
        for a, b in self.matches[self.phase//2]:
            for src, dst in [(a, b), (b, a)]:
                self.send_control(src, 'probe', dict(phase=self.phase, dst=dst, size=size))

    def probe_received(self, event):
        p = event.data['payload']
        self.send_control(0, 'probe_report', dict(src=event.data['from'], dst=self.index,
            phase=p['phase'], size=p['size'], elapsed=self.context.now-p['sent']))

    def on_message(self, event):
        kind, message = event.data['type'], event.data['message']
        if kind == 'probe':
            self.send(message['dst'], dict(phase=message['phase'], size=message['size'], sent=self.context.now),
                      size_bytes=message['size'], on_receive='probe_received')
        elif kind == 'training_profile':
            if self.index != 0:
                raise ValueError('Profile sent to wrong coordinator')
            self.training_profiles[event.data['from']] = message['duration']
            self.maybe_plan()
        elif kind == 'probe_report':
            if self.index != 0 or message['phase'] != self.phase:
                raise ValueError('Invalid probe phase')
            self.phase_records[self.phase].append(message)
            self.probe_records.append(message)
            if len(self.phase_records[self.phase]) == 2*len(self.matches[self.phase//2]):
                self.phase += 1
                self.begin_probe()
        elif kind == 'layout':
            if event.data['from'] != 0 or self.owners is not None:
                raise ValueError('Unauthorized or duplicate layout')
            self.profiles = {i: tuple(p) for i, p in enumerate(message['profiles'])}
            self.simulator.plan_arrivals.append((self.index, self.context.now))
            self.install_layout(message['owners'])
            early, self.early = self.early, []
            for payload in early:
                super().on_incoming_model(payload)
        else:
            raise ValueError('Unknown scheduler control message')

    def maybe_plan(self):
        n = self.simulator.settings.participants
        if self.planning or self.phase != 2*len(self.matches) or len(self.training_profiles) != n:
            return
        self.planning = True
        overhead = self.data_transfer_size(0, {})
        self.profiles = summarize_probes(self.probe_records, self.training_profiles, overhead)
        started = perf_counter()
        best, candidates = choose_layout(self.profiles, self.sample, self.simulator.model_size,
            self.simulator.optimizer_elements, self.data_transfer_size(0, dict(round=1, chunk=0, phase='aggregate', dispatch=0)),
            round(self.simulator.settings.server_step_ms*1000))
        self.simulator.planner_wall_seconds = perf_counter()-started
        self.simulator.scheduler_candidates = candidates
        self.simulator.measured_profiles = self.profiles
        self.simulator.probe_records = self.probe_records
        self.simulator.ps_owners = best['owners']
        self.simulator.settings.chunks_in_sample = best['chunks']
        self.simulator.settings.relay_count = best['reducers']
        delay = max(CANDIDATE_US*len(candidates), ceil(self.simulator.planner_wall_seconds*1e6))
        self.simulator.scheduler_cpu_us += delay
        self.set_timer(delay, 'announce_layout', dict(owners=best['owners'],
            profiles=[self.profiles[i] for i in range(n)]))

    def announce_layout(self, event):
        for peer in range(self.simulator.settings.participants):
            self.send_control(peer, 'layout', event.data)

    def upload_order(self, r):
        # Common ordering helps complete ranges early. Expensive owner tails go
        # first; the next cohort changes this estimate even with fixed homes.
        m = len(self.sample(r)); dst = self.sample(r+1)
        def cost(c):
            home = self.owners[c]; up, down, _, _ = self.profiles[home]
            remote = len(dst)-int(home in dst)
            # All contributors must agree on chunk order. Sender-local route
            # feedback here would desynchronize their pipelines and delay the
            # last contribution of many chunks at once.
            return m/down+remote/up
        return sorted(range(len(self.owners)), key=lambda c: (-cost(c), c))

    def recipient_order(self, r):
        # Slow downloads / subsequent training get a head start; all recipients
        # still reconstruct exactly the same complete model before training.
        return sorted(self.sample(r+1), key=lambda i: (
            -(self.simulator.model_size/self.profiles[i][1]+self.profiles[i][2]/1e6)
            * self.route_feedback.factor(i), i))

    def send_chunk(self, peer, payload, r, c, phase):
        if peer == self.index:
            return super().send_chunk(peer, payload, r, c, phase)
        size = self.chunk_transfer_size(c, len(self.owners))
        self.counter += 1
        meta = dict(round=r, chunk=c, phase=phase, dispatch=self.counter)
        self.policy_until = max(self.context.now, self.policy_until)+DISPATCH_US
        self.simulator.scheduler_cpu_us += DISPATCH_US
        self.set_timer(max(self.policy_until, self.payload_available_at(payload))-self.context.now,
                       'queue_transfer', dict(peer=peer, payload=payload, size=size, meta=meta))

    def queue_transfer(self, event):
        self.pending.append(event.data)
        self.pump()

    def pump(self):
        # At most one active transfer per destination. Other destinations can
        # use free send capacity when one receiver is blocked. Completion is a
        # local transport event, so no uncharged application ACK is assumed.
        while True:
            eligible = [(idx, p) for idx, p in enumerate(self.pending) if p['peer'] not in self.busy_peers]
            if not eligible:
                return
            def priority(item):
                _, p = item; meta = p['meta']
                return meta['round'], meta['dispatch']
            idx, p = min(eligible, key=priority)
            self.pending.pop(idx); self.busy_peers.add(p['peer'])
            self.send(p['peer'], p['payload'], size_bytes=p['size'], metadata=p['meta'])

    def finish_outgoing_transfer(self, event):
        t = event.data['transfer']
        super().finish_outgoing_transfer(event)
        if 'dispatch' not in t.metadata:
            return
        peer = t.receiver_scheduler.my_id
        predicted = t.transfer_size/min(self.profiles[self.index][0], self.profiles[peer][1])*1e6
        # Queueing/receiver effects stay a route-delay factor, not a purported
        # bandwidth measurement. Only this sender's transport timing is used.
        self.route_feedback.observe(peer, t.duration, predicted)
        self.busy_peers.remove(peer)
        self.pump()

    def on_incoming_model(self, event):
        if self.owners is None:
            self.early.append(event)
        else:
            super().on_incoming_model(event)
