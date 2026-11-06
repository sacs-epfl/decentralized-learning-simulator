"""Wait for a detached campaign to succeed, then run another without GPU overlap."""
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def launch_directory(output):
    return output.with_name(output.name + '-launch')


def process_alive(identity):
    try:
        process = psutil.Process(identity['pid'])
        return (process.create_time() == identity['create_time'] and
                process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def dependency_ready(output, settings, active):
    """Require both process exit and recorded success of every evaluation scenario."""
    from dasklearn.leaderboard_details import read_json
    if active:
        return False
    campaign = read_json(output / 'campaign.json')
    pipeline = read_json(output / 'pipeline.json')
    if (not campaign or campaign.get('status') != 'completed' or
            set(campaign.get('settings_completed', [])) != set(settings) or
            not pipeline or pipeline.get('status') != 'completed'):
        raise ValueError('Preceding campaign exited without completing tuning and every evaluation setting')
    return True


def run_queue(catalog, entry, dependency, settings, identity, *, poll_seconds=15):
    from dasklearn.experiment_runner import write_json
    output = entry['_benchmark'].output
    launch = launch_directory(output)
    launch.mkdir(parents=True, exist_ok=True)
    state = dict(version=1, status='queued', phase=f"waiting for {dependency['id']} to finish",
                 algorithm=entry['id'], after=dependency['id'], dependency_process=identity,
                 started_at=datetime.now(timezone.utc).isoformat(), error=None)
    with (launch / 'queue.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another queue already owns this campaign')
        if output.exists():
            raise ValueError(f'Output directory already exists: {output}')
        write_json(launch / 'queue.json', state)
        try:
            while not dependency_ready(dependency['_benchmark'].output, settings, process_alive(identity)):
                time.sleep(poll_seconds)
            if output.exists():
                raise ValueError(f'Output directory already exists: {output}')
            command = [sys.executable, '-u', str(Path(__file__).with_name('run_methodology.py')),
                       '--catalog', str(catalog), '--algorithm', entry['id'], '--workers-per-gpu', '3']
            log_path = launch / 'campaign.log'
            with log_path.open('ab') as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         cwd=Path(__file__).resolve().parents[1], start_new_session=True,
                                         close_fds=True)
            write_json(launch / 'background.json', dict(pid=child.pid, command=command,
                       started_at=datetime.now(timezone.utc).isoformat(), log=str(log_path), output=str(output)))
            state.update(status='running', phase='campaign launched', campaign_pid=child.pid,
                         launched_at=datetime.now(timezone.utc).isoformat())
            write_json(launch / 'queue.json', state)
            code = child.wait()
            state.update(status='completed' if code == 0 else 'failed', phase='campaign exited', returncode=code)
            if code:
                state['error'] = f'Campaign exited with status {code}; inspect campaign.log'
        except KeyboardInterrupt:
            # If already launched, leave the independent campaign running.
            code = 130
            state.update(status='interrupted', phase='queue monitor interrupted',
                         error='Queue monitor interrupted; an already launched campaign continues independently')
        except (OSError, ValueError, psutil.Error) as exc:
            code = 1
            state.update(status='failed', phase='queue failed', error=str(exc))
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(launch / 'queue.json', state)
        return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path(__file__).resolve().parents[1] / 'experiments/leaderboard-settings.json')
    parser.add_argument('--after', required=True, help='Registered campaign that must finish first')
    parser.add_argument('--algorithm', required=True, help='Registered campaign to launch next')
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.after == args.algorithm:
        parser.error('Use a positive polling interval and different campaigns')
    from dasklearn.leaderboard_details import algorithm_entry, load_algorithms, read_json
    from scripts.run_methodology import graph_plan
    registry = load_algorithms(args.catalog)
    entry, dependency = (algorithm_entry(registry, identifier) for identifier in (args.algorithm, args.after))
    if not entry or not dependency:
        parser.error('Unknown registered campaign')
    graph_plan(entry['_benchmark'])
    background = read_json(launch_directory(dependency['_benchmark'].output) / 'background.json')
    if not background:
        parser.error('Preceding campaign has no detached process metadata')
    try:
        process = psutil.Process(background['pid'])
        command = process.cmdline()
        if (not any(item.endswith('run_methodology.py') for item in command) or
                '--algorithm' not in command or command[command.index('--algorithm') + 1] != args.after):
            parser.error('Preceding campaign PID does not match its registered algorithm')
        identity = dict(pid=process.pid, create_time=process.create_time())
    except psutil.NoSuchProcess:
        # A successfully completed predecessor can also be queued after it exits.
        dependency_ready(dependency['_benchmark'].output, registry['scenarios'], False)
        identity = dict(pid=background['pid'], create_time=-1)
    return run_queue(args.catalog.resolve(), entry, dependency, registry['scenarios'], identity,
                     poll_seconds=args.poll_seconds)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
