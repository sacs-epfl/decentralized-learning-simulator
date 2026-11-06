"""Changing regular topology, queued launch ordering, and live leaderboard visibility."""
from dataclasses import replace
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from args import create_parser
from dasklearn.benchmark import load_benchmark
from dasklearn.leaderboard_details import algorithm_document, combined_leaderboard, load_algorithms
from dasklearn.simulation.dpsgd.simulation import DPSGDSimulation
from scripts import queue_methodology
from scripts.run_methodology import graph_plan

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = 'epidemic-kreg-log2'


def test_epidemic_campaign_graph_matches_actual_round_schedule():
    benchmark = load_benchmark(ROOT / f'experiments/methodology/{IDENTIFIER}-tpe-dirichlet01-h20.json', create_parser())
    assert benchmark.execution_devices == ['cuda:0', 'cuda:1'] * 3
    assert benchmark.tuning['batch_size'] * len(benchmark.tuning['seeds']) == 6
    assert benchmark.tuning['metric'] == 'best_mean_accuracy'
    assert benchmark.tuning['budget'] == 900 and benchmark.tuning['n_trials'] == 40
    assert benchmark.tuning['search_spaces'][IDENTIFIER]['local_steps']['choices'] == [1, 5, 10, 15, 20]
    for seed, record in graph_plan(benchmark).items():
        assert record['graph_nodes'] == record['real_nodes'] == 32 and record['degree'] == 5
        assert record['resampled_each_round'] and record['graph_seed_rule'] == 'seed + round - 1'
        sim = DPSGDSimulation.__new__(DPSGDSimulation)
        sim.settings = SimpleNamespace(algorithm='epidemic', el='oracle', topology='kreg', k=5,
                                       participants=32, seed=int(seed))
        sim.topologies = []
        for round_index in range(3):
            sim.add_topology()
            expected = nx.random_regular_graph(5, 32, seed=int(seed) + round_index).to_directed()
            assert set(sim.topologies[-1].edges) == set(expected.edges)
            assert all(sim.topologies[-1].in_degree(node) == sim.topologies[-1].out_degree(node) == 5 for node in expected)
        assert set(sim.topologies[0].edges) != set(sim.topologies[1].edges)
    invalid = replace(benchmark, runs=[replace(run, options=run.options | dict(el='local')) for run in benchmark.runs])
    with pytest.raises(ValueError, match='EL-Oracle'):
        graph_plan(invalid)


def completed_dependency(output, settings):
    output.mkdir()
    (output / 'pipeline.json').write_text(json.dumps(dict(status='completed')))
    (output / 'campaign.json').write_text(json.dumps(dict(status='completed', settings_completed=list(settings))))


def test_dependency_waits_for_exit_and_requires_all_evaluations(tmp_path):
    output = tmp_path / 'predecessor'
    settings = ['iid', 'non-iid']
    assert not queue_methodology.dependency_ready(output, settings, active=True)
    with pytest.raises(ValueError, match='every evaluation'):
        queue_methodology.dependency_ready(output, settings, active=False)
    completed_dependency(output, settings)
    assert not queue_methodology.dependency_ready(output, settings, active=True)
    assert queue_methodology.dependency_ready(output, settings, active=False)
    (output / 'campaign.json').write_text(json.dumps(dict(status='completed', settings_completed=['iid'])))
    with pytest.raises(ValueError, match='every evaluation'):
        queue_methodology.dependency_ready(output, settings, active=False)


def test_recycled_pid_is_not_treated_as_the_preceding_campaign(monkeypatch):
    process = SimpleNamespace(create_time=lambda: 2., is_running=lambda: True,
                              status=lambda: 'running')
    monkeypatch.setattr(queue_methodology.psutil, 'Process', lambda pid: process)
    assert queue_methodology.process_alive(dict(pid=123, create_time=2.))
    assert not queue_methodology.process_alive(dict(pid=123, create_time=1.))


def test_second_queue_cannot_launch_the_same_campaign(tmp_path):
    output = tmp_path / 'next'
    launch = queue_methodology.launch_directory(output)
    launch.mkdir()
    with (launch / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='Another queue'):
            queue_methodology.run_queue(tmp_path / 'catalog.json',
                dict(id='next', _benchmark=SimpleNamespace(output=output)),
                dict(id='previous'), ['a'], dict(pid=1, create_time=1))
    assert not output.exists()


@pytest.mark.parametrize('predecessor_success, successor_code', [(True, 0), (True, 1), (False, 0)])
def test_queue_launches_only_after_successful_exit_and_records_failures(tmp_path, monkeypatch, predecessor_success, successor_code):
    output, previous = tmp_path / 'next', tmp_path / 'previous'
    entry = dict(id='next', _benchmark=SimpleNamespace(output=output))
    dependency = dict(id='previous', _benchmark=SimpleNamespace(output=previous))
    settings = ['a', 'b']
    if predecessor_success:
        completed_dependency(previous, settings)
    alive = iter([True, False])
    monkeypatch.setattr(queue_methodology, 'process_alive', lambda identity: next(alive))
    waits, starts = [], []
    monkeypatch.setattr(queue_methodology.time, 'sleep', waits.append)
    def launch(command, **kwargs):
        assert waits == [15] and predecessor_success
        assert command[-2:] == ['--workers-per-gpu', '3']
        assert command[command.index('--algorithm') + 1] == 'next'
        assert not output.exists()
        starts.append(command)
        return SimpleNamespace(pid=123, wait=lambda: successor_code)
    monkeypatch.setattr(queue_methodology.subprocess, 'Popen', launch)
    code = queue_methodology.run_queue(tmp_path / 'catalog.json', entry, dependency, settings,
                                       dict(pid=1, create_time=1))
    state = json.loads((queue_methodology.launch_directory(output) / 'queue.json').read_text())
    assert len(starts) == int(predecessor_success)
    assert code == (successor_code if predecessor_success else 1)
    assert state['status'] == ('completed' if predecessor_success and successor_code == 0 else 'failed')
    if code:
        assert state['error']


def test_queued_third_algorithm_is_visible_without_creating_trial_outputs(tmp_path):
    registry = load_algorithms(ROOT / 'experiments/leaderboard-settings.json')
    for item in registry['algorithms']:
        item['_benchmark'] = replace(item['_benchmark'], output=tmp_path / item['id'])
    entry = next(item for item in registry['algorithms'] if item['id'] == IDENTIFIER)
    launch = queue_methodology.launch_directory(entry['_benchmark'].output)
    launch.mkdir()
    (launch / 'queue.json').write_text(json.dumps(dict(status='queued', phase='waiting for shatter-k16-rlog2 to finish')))
    details = algorithm_document(entry, registry, 'dirichlet01-heterogeneous')
    assert details['status'] == 'queued' and 'shatter' in details['phase']
    assert details['tuning']['completed_trials'] == 0 and details['tuning']['total_trials'] == 120
    assert details['tuning']['hardware_profile'] == 'uniform'
    combined = combined_leaderboard(registry, 'dirichlet01-heterogeneous')
    row = next(row for row in combined['tables'][-1]['rows'] if row['variant'] == IDENTIFIER)
    assert row['accuracy'] is None
    assert not entry['_benchmark'].output.exists()
