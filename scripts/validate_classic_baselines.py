#!/usr/bin/env python3
"""Small offline validation matrix, optionally replayed through real brokers.

This is a correctness/smoke check, not a comparative performance benchmark.
All artifacts are placed under --output; frozen experiment directories are unused.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def child(args):
    from dasklearn.algorithms import class_paths, import_class
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if args.coordinator_config:
        values = json.loads(Path(args.coordinator_config).read_text())
        settings_type, simulation_type = map(import_class, class_paths(values['algorithm']))
        simulation = simulation_type(settings_type.from_dict(values))
        loop.create_task(simulation.run())
    else:
        from dasklearn.broker import Broker
        broker = Broker(SimpleNamespace(port=args.port, workers=args.workers, profile=False,
            coordinator=args.coordinator, advertise_host='127.0.0.1'))
        loop.create_task(broker.start())
    try:
        loop.run_forever()
    finally:
        loop.close()


def free_ports(count):
    held = []
    try:
        for _ in range(count):
            connection = socket.socket()
            connection.bind(('127.0.0.1', 0))
            held.append(connection)
        return [connection.getsockname()[1] for connection in held]
    finally:
        for connection in held:
            connection.close()


def frame(path):
    import pandas as pd
    return pd.read_csv(path).sort_values(['peer', 'time', 'round']).reset_index(drop=True)


def replay(simulation, output, brokers, workers, timeout):
    import pandas as pd
    target = output / f'brokers-{brokers}-workers-{workers}'
    target.mkdir(parents=True, exist_ok=False)
    expected = frame(Path(simulation.data_dir) / 'accuracies.csv')
    config = simulation.settings.to_dict()
    ports = free_ports(brokers + 1)
    config.update(from_dir=simulation.data_dir, data_dir=str(target), unit_testing=False,
                  dry_run=False, brokers=brokers, port=ports[0])
    config_file = target / 'settings.json'
    config_file.write_text(json.dumps(config, indent=2))
    commands = [[sys.executable, str(Path(__file__).resolve()), '--coordinator-config', str(config_file)]]
    commands += [[sys.executable, str(Path(__file__).resolve()), '--broker', '--port', str(port),
                  '--workers', str(workers), '--coordinator', f'tcp://127.0.0.1:{ports[0]}'] for port in ports[1:]]
    processes, handles = [], []
    start = time.monotonic()
    try:
        for i, command in enumerate(commands):
            handle = (target / f'process-{i}.log').open('w')
            handles.append(handle)
            processes.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                               start_new_session=True))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f'Broker replay failed; inspect {target}')
            if time.monotonic() - start > timeout:
                raise TimeoutError(f'Broker replay timed out; inspect {target}')
            time.sleep(.1)
        if any(p.returncode for p in processes):
            raise RuntimeError(f'Broker replay failed; inspect {target}')
        actual = frame(target / 'accuracies.csv')
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=1e-6)
        # FL also emits server evaluations, which are deliberately not part of
        # the mean of clients' stored local models.
        server_file = f'accuracies_{simulation.settings.participants}.csv'
        if (Path(simulation.data_dir) / server_file).exists():
            pd.testing.assert_frame_equal(frame(target / server_file), frame(Path(simulation.data_dir) / server_file),
                                          check_exact=False, rtol=0, atol=1e-6)
        return {'brokers': brokers, 'workers_per_broker': workers,
                'metric_rows': len(actual), 'status': 'matched-local',
                'wall_seconds': round(time.monotonic() - start, 3)}
    finally:
        for process in processes:
            # Also remove descendants after timeout/failure, including workers.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()


def validate(args):
    import numpy as np
    import torch
    from dasklearn.algorithms import class_paths, import_class
    from dasklearn.session_settings import LearningSettings
    from dasklearn.tasks.local import execute_local
    from dasklearn.tasks.task import Task
    from dasklearn.util import MICROSECONDS
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    cases = []
    for algorithm in args.algorithms:
        for seed in args.seeds:
            for partitioner in ['iid', 'dirichlet']:
                for skew in [False, True]:
                    name = f'{algorithm}-s{seed}-{partitioner}-' + ('hetero' if skew else 'uniform')
                    case_dir = output / name
                    settings_type, simulation_type = map(import_class, class_paths(algorithm))
                    extra = {'topology': 'ring', 'k': 2} if algorithm in {'dpsgd', 'epidemic'} else {}
                    if algorithm == 'fl':
                        extra = {'sample_size': 3, 'aggregation_weighting': 'examples'}
                    config = settings_type(algorithm=algorithm, seed=seed, work_dir=str(case_dir),
                        dataset='synthetic', model='mlp', participants=4,
                        learning=LearningSettings(.1, 0, 0, 32, 2), rounds=4,
                        duration=2*MICROSECONDS, test_period=MICROSECONDS//2,
                        test_method='snapshot', test_interval=1 if algorithm == 'fl' else 0,
                        dry_run=True, unit_testing=True, model_size_bytes=1024,
                        network_latency_ms=3, partitioner=partitioner, alpha=.1, torch_threads=1, **extra)
                    Task.COUNTER.clear()
                    sim = simulation_type(config)
                    sim.setup_directories()
                    def capacities():
                        for peer in sim.clients:
                            peer.simulated_speed = [4, 17, 8, 31][peer.index] if skew else 10
                            peer.bw_scheduler.bw_limit = [50000, 100000, 200000, 50000][peer.index] if skew else 100000
                    sim.apply_traces = capacities
                    sim.simulate()
                    sim.workflow_dag.build_task_indices()
                    sim.workflow_dag.check_validity()
                    execute_local(sim, retain_results=False)
                    sim.merge_accuracies_files()
                    metrics = frame(Path(sim.data_dir) / 'accuracies.csv')
                    assert len(metrics) == 16 and metrics.groupby('time').peer.nunique().eq(4).all()
                    assert np.isfinite(metrics[['accuracy', 'loss']]).all().all()
                    assert metrics['round'].gt(0).all()
                    sent = sum(c.bw_scheduler.total_bytes_sent for c in sim.clients)
                    received = sum(c.bw_scheduler.total_bytes_received for c in sim.clients)
                    assert sent == received
                    row = {'case': name, 'algorithm': algorithm, 'seed': seed, 'partitioner': partitioner,
                           'heterogeneous': skew, 'snapshots': 4, 'nodes_per_snapshot': 4,
                           'tasks': len(sim.workflow_dag.tasks), 'traffic_bytes': sent,
                           'final_mean_accuracy': float(metrics[metrics.time == 2].accuracy.mean()),
                           'final_mean_loss': float(metrics[metrics.time == 2].loss.mean()),
                           'dag_sha256': hashlib.sha256((Path(sim.data_dir)/'workflow_dag').read_bytes()).hexdigest(),
                           'status': 'passed'}
                    if args.distributed and seed == args.seeds[0] and partitioner == 'dirichlet' and skew:
                        row['distributed'] = [replay(sim, case_dir, n, w, args.timeout)
                                              for n, w in [(1, 2), (2, 1)]]
                    cases.append(row)
                    print(name, 'passed', flush=True)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        revision = None
    report = {'purpose': 'correctness validation, not an algorithm ranking', 'base_revision': revision,
              'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__,
              'source_sha256': hashlib.sha256(b''.join(str(p.relative_to(ROOT)).encode()+p.read_bytes()
                  for p in sorted((ROOT/'dasklearn').rglob('*.py')))).hexdigest(),
              'cases': cases}
    (output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    loop.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--algorithms', nargs='+', default=['adpsgd', 'dpsgd', 'fl', 'epidemic'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--coordinator-config', help=argparse.SUPPRESS)
    parser.add_argument('--broker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--coordinator', help=argparse.SUPPRESS)
    parser.add_argument('--port', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--workers', type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.coordinator_config or args.broker:
        child(args)
    elif not args.output:
        parser.error('--output is required')
    else:
        validate(args)
