import csv
import json
from pathlib import Path
from dasklearn.simulation.simulation import Simulation
from dasklearn.simulation.conflux_v2.simulation import ConfluxV2Simulation
from dasklearn.simulation.conflux_v2.client import _layout
from .client import ShardedPSClient
from .auto_client import AutoDirectClient


class ShardedPSSimulation(ConfluxV2Simulation):
    def __init__(self, settings):
        if settings.success_fraction != 1 or settings.reducer_epoch_rounds != 0:
            raise ValueError('Matched sharded PS requires all sampled contributors and fixed owners')
        if settings.participants < 2:
            raise ValueError('Sharded comparison requires at least two nodes')
        super().__init__(settings)
        self.CLIENT_CLASS = ShardedPSClient

    def initialize_clients(self):
        super().initialize_clients()
        _, self.ps_owners = _layout(1, self.settings.chunks_in_sample,
            self.settings.relay_count or self.settings.chunks_in_sample,
            tuple(self.clients[0].other_nodes_bws.items()))


class AutoDirectSimulation(ShardedPSSimulation):
    def __init__(self, settings):
        super().__init__(settings)
        self.CLIENT_CLASS = AutoDirectClient
        self.scheduler_cpu_us = 0
        self.planner_wall_seconds = 0
        self.scheduler_candidates = []
        self.measured_profiles = {}
        self.probe_records = []
        self.plan_arrivals = []

    def initialize_clients(self):
        # Deliberately bypass V2's capacity-map oracle. Only the numerical model
        # geometry is public; each client learns timing through charged probes.
        from dasklearn.functions import initialize
        model = initialize(self.settings, {})[0]
        self.optimizer_elements = sum(v.numel() for v in model.state_dict().values())
        Simulation.initialize_clients(self)

    def save_measurements(self):
        super().save_measurements()
        directory = Path(self.data_dir)
        record = dict(chunks=self.settings.chunks_in_sample, reducers=len(set(getattr(self, 'ps_owners', []))),
            owners=getattr(self, 'ps_owners', []), profiles=self.measured_profiles,
            plan_arrivals=self.plan_arrivals, scheduler_cpu_us=self.scheduler_cpu_us,
            planner_wall_seconds=self.planner_wall_seconds, candidate_count=len(self.scheduler_candidates),
            startup_first_plan_us=min((t for _, t in self.plan_arrivals), default=None),
            startup_last_plan_us=max((t for _, t in self.plan_arrivals), default=None),
            feedback='Median of five local transport route-delay ratios, not endpoint-capacity updates',
            costs='Startup max(measured planner wall time, 100us/candidate); 50us/remote dispatch controlled CPU assumption')
        (directory/'scheduler.json').write_text(json.dumps(record, indent=2)+'\n')
        for name, rows in [('scheduler-candidates.csv', self.scheduler_candidates), ('scheduler-probes.csv', self.probe_records)]:
            if rows:
                with (directory/name).open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                    writer.writeheader(); writer.writerows(rows)
