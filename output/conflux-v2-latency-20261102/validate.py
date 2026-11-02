"""Check numerical preservation for the timing probe's representative cases."""
import json
from pathlib import Path
import tempfile

import torch
from probe import CLIENTS, run, settings, ConfluxV2Settings, ConfluxV2Simulation
from dasklearn.tasks.local import execute_local
from dasklearn.simulation.conflux.chunk_manager import ChunkManager


def main():
    output = Path(__file__).resolve().parent
    checks = []
    with tempfile.TemporaryDirectory(prefix='conflux-latency-check-') as scratch:
        models = []
        for chunks, mode in [(16,'current'), (64,'current'), (16,'upload-order'), (16,'both-orders')]:
            config = settings(ConfluxV2Settings, Path(scratch)/f'{chunks}-{mode}',
                participants=16, sample_size=4, chunks_in_sample=chunks, relay_count=16,
                immediate_uploads=True, rounds=3, duration=3600_000_000,
                test_period=0, capacity_profile='uniform', bandwidth_bytes_per_s=100000,
                compute_step_ms=20, model_size_bytes=346251, network_latency_ms=20)
            sim = ConfluxV2Simulation(config)
            sim.CLIENT_CLASS = CLIENTS[mode]
            sim.setup_directories()
            sim.simulate()
            sim.workflow_dag.build_task_indices()
            sim.workflow_dag.check_validity()
            results = execute_local(sim)
            current = {(t.data['round'],t.data['peer']):ChunkManager.get_flat_params(results[t.name][0]).clone()
                for t in sim.workflow_dag.tasks.values() if t.func == 'reconstruct_from_chunks'}
            assert len(current) == 12
            if models:
                assert current.keys() == models[0].keys()
                for key in current:
                    torch.testing.assert_close(current[key], models[0][key], rtol=1e-6, atol=1e-7)
            models.append(current)
            checks.append({'chunks':chunks,'mode':mode,'aggregate_checks':len(current),'passed':True})
        extra = [run(Path(scratch)/f'extra-{c}-{seed}',16,4,c,16,'current',seed)
                 for c in [128,256] for seed in [0,1,2]]
    (output/'validation.json').write_text(json.dumps(dict(
        numerical_checks=checks, numerical_local_steps=2,
        note='Synthetic numerical smoke check only; no target-accuracy comparison.'),indent=2)+'\n')
    import csv
    with (output/'extra-timing.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(extra[0]))
        writer.writeheader()
        writer.writerows(extra)
    print(json.dumps(checks),flush=True)
    for c in [128,256]:
        print(c,sum(r['mean_interval_s'] for r in extra if r['chunks']==c)/3,flush=True)


if __name__ == '__main__':
    main()
