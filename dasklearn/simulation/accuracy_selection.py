"""Small executable feedback example, not a paper baseline.

All nodes train once and report validation accuracy to node zero. Node zero
requests the winning checkpoint, receives its charged model transfer, and trains
it once more. Only reports delivered to node zero can influence its selection.
"""
import json
from pathlib import Path

from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.protocol import RoundInbox
from dasklearn.simulation.simulation import Simulation


class AccuracySelectionClient(BaseClient):
    def __init__(self, simulator, index):
        super().__init__(simulator, index)
        self.reports = RoundInbox(range(simulator.settings.participants))
        self.selected_peer = None
        self.selected_at = None
        self.received_at = None
        self.model = None

    def init_client(self, event):
        initial = self.compute('initialize', {}, duration=0)
        self.compute('train', {'model': initial}, params={'round': 1},
                     duration=self.get_train_time(), on_complete='trained')

    def trained(self, event):
        self.model = event.data['outputs'][0]
        self.publish_model(1, self.model)
        self.compute('evaluate', {'model': self.model},
                     params={'round': 1, 'evaluation_split': 'validation'},
                     duration=50_000, feedback=True, on_complete='evaluated')

    def evaluated(self, event):
        report = dict(event.data['values'][0], completed_at=event.time)
        self.send(0, report, size_bytes=64, metadata={'round': 1}, on_receive='report_received')

    def report_received(self, event):
        self.reports.add(event.data['from'], dict(event.data['payload'], received_at=event.time))
        if not self.reports.finish():
            return
        self.selected_peer = min(self.reports.messages,
            key=lambda peer: (-self.reports.messages[peer]['accuracy'], peer))
        self.selected_at = event.time
        self.send(self.selected_peer, {'round': 1}, size_bytes=16, on_receive='checkpoint_requested')

    def checkpoint_requested(self, event):
        self.send(event.data['from'], self.model, size_bytes=self.simulator.model_size,
                  metadata={'round': 1}, on_receive='checkpoint_received')

    def checkpoint_received(self, event):
        self.received_at = event.time
        self.compute('train', {'model': event.data['payload']}, params={'round': 2},
                     duration=self.get_train_time(), on_complete='finished')

    def finished(self, event):
        self.model = event.data['outputs'][0]
        self.publish_model(2, self.model)
        self.create_compute_task('test', {'model': self.model, 'peer': self.index,
                                        'round': 2, 'time': event.time})


class AccuracySelectionSimulation(Simulation):
    CLIENT_CLASS = AccuracySelectionClient

    def save_measurements(self):
        super().save_measurements()
        selector = self.clients[0]
        Path(self.data_dir, 'selection.json').write_text(json.dumps({
            'selected_peer': selector.selected_peer, 'selected_at_us': selector.selected_at,
            'checkpoint_received_at_us': selector.received_at,
            'reports': selector.reports.messages,
        }, indent=2) + '\n')
