"""Run accuracy-driven peer selection locally or with persistent brokers."""
import argparse
import asyncio
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dasklearn.communication import Communication
from dasklearn.session_settings import LearningSettings, SessionSettings
from dasklearn.simulation.accuracy_selection import AccuracySelectionSimulation
from dasklearn.tasks.incremental import RemoteExecutionSession
from dasklearn.tasks.local import execute_local


async def run(args):
    settings = SessionSettings(algorithm='accuracy-selection-example', seed=args.seed,
        work_dir=str(args.output), dataset='synthetic', model='mlp', participants=args.peers,
        learning=LearningSettings(.1, 0, 0, 16, 2), duration=10_000_000,
        test_period=0, test_method='individual', torch_threads=1, model_size_bytes=4096,
        network_latency_ms=10, brokers=args.brokers, port=args.port,
        feedback_execution='distributed' if args.distributed else 'local')
    simulation = AccuracySelectionSimulation(settings)
    simulation.setup_directories()
    if args.distributed:
        communication = Communication('coordinator', args.port, simulation.on_message)
        simulation.communication = communication
        communication.start()
        session = RemoteExecutionSession(simulation)
        try:
            await simulation.simulate_async(session, replay=args.replay)
            await session.execute_all()
        finally:
            if session.configured:
                communication.send_message_to_all_brokers(pickle.dumps({'type': 'shutdown'}))
            if communication.receive_msg_task:
                communication.receive_msg_task.cancel()
            if communication.listen_socket:
                communication.listen_socket.close(linger=0)
            for socket in communication.broker_connections.values():
                socket.close(linger=1000)
    else:
        simulation.simulate(replay=args.replay)
        execute_local(simulation, retain_results=False)
    print(Path(simulation.data_dir, 'selection.json').read_text())
    print(f'Results: {simulation.data_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--peers', type=int, default=3)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--brokers', type=int, default=1)
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--replay', help='Trusted feedback.pkl from a matching run')
    asyncio.run(run(parser.parse_args()))
