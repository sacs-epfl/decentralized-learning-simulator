"""Timing-only V2 diagnostic; no learning execution or production source edits.

Run from repository root with PYTHONPATH=.:tests. The two experimental variants
only reorder uploads / aggregate sends by loading the current methods and
replacing their enumeration order. Contributor sets and reduction order stay
unchanged. All work directories are temporary.
"""
import asyncio
import csv
import hashlib
import inspect
import json
import logging
from pathlib import Path
import statistics
import tempfile
import textwrap

from helpers import settings
from dasklearn.simulation.conflux_v2 import client as client_module
from dasklearn.simulation.conflux_v2.client import ConfluxV2Client
from dasklearn.simulation.conflux_v2.settings import ConfluxV2Settings
from dasklearn.simulation.conflux_v2.simulation import ConfluxV2Simulation

logging.disable(logging.CRITICAL)
asyncio.set_event_loop(asyncio.new_event_loop())


def reordered_method(name, old, new):
    source = textwrap.dedent(inspect.getsource(getattr(ConfluxV2Client, name)))
    assert source.count(old) == 1
    namespace = dict(vars(client_module))
    exec(source.replace(old, new), namespace)
    return namespace[name]


class UploadOrderClient(ConfluxV2Client):
    finish_train = reordered_method('finish_train',
        'for chunk_idx, owner in enumerate(owners):',
        '''offset = self.sample(round_nr).index(self.index) * len(owners) // len(self.sample(round_nr))
        for chunk_idx in [(offset + i) % len(owners) for i in range(len(owners))]:
            owner = owners[chunk_idx]''')


class BothOrdersClient(UploadOrderClient):
    try_reduce = reordered_method('try_reduce',
        'for peer in self.sample(round_nr + 1):',
        '''recipients = self.sample(round_nr + 1)
        offset = idx % len(recipients)
        for peer in recipients[offset:] + recipients[:offset]:''')


CLIENTS = {'current': ConfluxV2Client, 'upload-order': UploadOrderClient,
           'both-orders': BothOrdersClient}


def run(root, peers, sample, chunks, relays, mode, seed, rounds=20):
    config = settings(ConfluxV2Settings, root, participants=peers, sample_size=sample,
        seed=seed, chunks_in_sample=chunks, relay_count=relays, immediate_uploads=True,
        rounds=rounds, duration=3600_000_000, test_period=0, network_latency_ms=20,
        capacity_profile='uniform', bandwidth_bytes_per_s=100_000, compute_step_ms=20,
        model_size_bytes=346251)
    config.learning.local_steps = 5
    sim = ConfluxV2Simulation(config)
    sim.CLIENT_CLASS = CLIENTS[mode]
    sim.setup_directories()
    sim.simulate()
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    assert len(sim.completed_rounds) == sample * rounds
    sent = sum(c.bw_scheduler.total_bytes_sent for c in sim.clients)
    assert sent == sum(c.bw_scheduler.total_bytes_received for c in sim.clients)
    first = [min(t for r, _, t in sim.completed_rounds if r == i) / 1e6
             for i in range(1, rounds + 1)]
    last = [max(t for r, _, t in sim.completed_rounds if r == i) / 1e6
            for i in range(1, rounds + 1)]
    return dict(peers=peers, sample=sample, chunks=chunks, relays=relays, mode=mode, seed=seed,
        rounds=rounds, mean_interval_s=statistics.mean(b-a for a,b in zip(first,first[1:])),
        last_recipient_s=last[-1], bytes_sent=sent, train_tasks=sum(
            t.func == 'train' for t in sim.workflow_dag.tasks.values()))


def main():
    output = Path(__file__).resolve().parent
    cases = ([(16,4,c,16,'current') for c in (8,16,32,64)] +
             [(16,4,16,16,m) for m in ('upload-order','both-orders')] +
             [(16,s,16,16,'current') for s in (8,16)] +
             [(64,16,c,r,'current') for c,r in ((8,8),(16,16),(32,16),(64,16),(32,32),(64,64))] +
             [(64,16,16,16,m) for m in ('upload-order','both-orders')])
    rows = []
    with tempfile.TemporaryDirectory(prefix='conflux-latency-') as scratch:
        for case in cases:
            batch = [run(Path(scratch) / str(len(rows)+seed), *case, seed) for seed in (0,1,2)]
            rows.extend(batch)
            print(json.dumps(dict(zip(('peers','sample','chunks','relays','mode'), case)) |
                {'mean_interval_s':statistics.mean(r['mean_interval_s'] for r in batch),
                 'last_recipient_s':statistics.mean(r['last_recipient_s'] for r in batch)}), flush=True)
    with (output/'timing.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths = ['dasklearn/simulation/conflux_v2/client.py',
             'dasklearn/simulation/bandwidth_scheduler.py', 'dasklearn/simulation/client.py']
    (output/'metadata.json').write_text(json.dumps(dict(
        description='Timing-only dry runs, synthetic task graph, no accuracy measurement',
        model_size_bytes=346251, bandwidth_bytes_per_s=100000, network_latency_ms=20,
        local_steps=5, compute_step_ms=20, seeds=[0,1,2], rounds=20,
        source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')


if __name__ == '__main__':
    main()
