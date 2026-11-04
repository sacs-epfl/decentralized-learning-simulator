"""Short, isolated MoTEF-VR runtime measurement; never edits a leaderboard catalog."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from args import get_args
from dasklearn.benchmark import configure_simulation
from dasklearn.settings_factory import session_options
from dasklearn.simulation.motef_vr.settings import MoTEFVRSettings
from dasklearn.simulation.motef_vr.simulation import MoTEFVRSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.task import Task


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--durations', type=float, nargs='+', default=[1, 3])
    parser.add_argument('--profiles', nargs='+', choices=['uniform', 'heterogeneous'], default=['uniform', 'heterogeneous'])
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    if any(d <= 0 for d in args.durations):
        parser.error('Durations must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for profile in args.profiles:
        for duration in args.durations:
            directory = args.output / f'{profile}-{duration:g}s'
            cli = get_args(['--algorithm', 'motef-vr', '--dataset', 'cifar10', '--dataset-base-path', args.data,
                            '--model', 'gnlenet', '--peers', '32', '--partitioner', 'dirichlet', '--alpha', '.1',
                            '--seed', '10', '--local-steps', '1', '--batch-size', '32', '--momentum', '0',
                            '--weight-decay', '0', '--learning-rate', '.01', '--duration', '1',
                            '--stop', 'duration', '--test-method', 'snapshot', '--test-period', '0',
                            '--network-latency-ms', '10', '--torch-threads', str(args.threads),
                            '--motef-keep-ratio', '.01', '--local', '--work-dir', str(directory), '--log-level', 'WARNING'])
            s = MoTEFVRSettings(**session_options(cli, args.device))
            s.duration = int(duration * 1e6) + 1
            Task.COUNTER.clear()
            started = time.perf_counter()
            sim = MoTEFVRSimulation(s)
            configure_simulation(sim, [0, duration], dict(profile=profile,
                                 bandwidth_bytes_per_second=1000000, compute_step_ms=20), 'global')
            sim.setup_directories()
            sim.simulate()
            sim.workflow_dag.build_task_indices()
            sim.workflow_dag.check_validity()
            simulation_seconds = time.perf_counter()-started
            last = {}
            def progress(status):
                last.update(status)
            started = time.perf_counter()
            execute_local(sim, retain_results=False, progress=progress)
            execution_seconds = time.perf_counter()-started
            sim.merge_accuracies_files()
            counts = Counter(t.func for t in sim.workflow_dag.tasks.values())
            row = dict(profile=profile, simulated_seconds=duration, device=args.device,
                       event_simulation_seconds=simulation_seconds, execution_seconds=execution_seconds,
                       tasks=dict(counts), gradient_evaluations=counts['motef_vr_bootstrap']+2*counts['motef_vr_gradient'],
                       completed_updates=sum(state['done'] for c in sim.clients for state in c.rounds.values()),
                       simulated_compute_seconds=sum(c.compute_time for c in sim.clients)/1e6,
                       packet_size_bytes=sim.packet_size, parameter_count=sim.parameter_count,
                       bytes_sent=sum(c.bw_scheduler.total_bytes_sent for c in sim.clients),
                       timings=last.get('timings_s',{}), artifacts=sim.data_dir)
            rows.append(row)
            (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(row),flush=True)


if __name__ == '__main__':
    asyncio.run(main())
