"""Teleportation search validity, token topology, and frozen four-setting evaluation."""
from dataclasses import replace
import json
from pathlib import Path

import networkx as nx
import pytest

from args import create_parser
from dasklearn.benchmark import load_benchmark
from dasklearn.leaderboard_details import algorithm_document, combined_leaderboard, load_algorithms
from scripts.run_methodology import execute_campaign, graph_plan

ROOT = Path(__file__).resolve().parents[1]


def test_teleportation_search_checks_every_active_count_and_topology():
    benchmark = load_benchmark(ROOT / 'experiments/methodology/teleportation-tpe-dirichlet01-h20.json', create_parser())
    space = benchmark.tuning['search_spaces']['teleportation']
    assert set(space) == {'learning_rate', 'local_steps', 'sample_size', 'teleportation_topology'}
    assert benchmark.tuning['n_trials'] == 40 and benchmark.tuning['budget'] == 900
    assert benchmark.tuning['metric'] == 'best_mean_accuracy'
    assert benchmark.tuning['seeds'] == [100, 101, 102] and benchmark.seeds == [110, 111, 112]
    assert benchmark.execution_devices == ['cuda:0', 'cuda:1'] * 3
    plans = graph_plan(benchmark)
    assert len(plans) == 6
    for record in plans.values():
        assert record['real_nodes'] == 32 and record['client_assignment_resampled_each_round']
        assert len(record['token_graphs']) == 15
        for item in record['token_graphs']:
            size, topology = item['active_clients'], item['topology']
            graph = nx.Graph()
            graph.add_nodes_from(range(size))
            graph.add_edges_from(item['edges'])
            expected_degree = min(2, size - 1) if topology == 'ring' else size.bit_length() - 1 if topology == 'hypercube' else size - 1
            assert nx.is_connected(graph) and all(degree == expected_degree for _, degree in graph.degree())
            assert item['self_and_neighbor_weight'] == pytest.approx(1 / (expected_degree + 1))
            assert len(set(item['first_token_owners'])) == len(set(item['next_token_owners'])) == size
            assert set(item['first_token_owners']) <= set(range(32))
            assert item['first_token_owners'] != item['next_token_owners']
    # Cross-parameter constraints must fail in preflight, before a worker starts.
    space['sample_size']['choices'] = [3]
    with pytest.raises(ValueError, match='power of two'):
        graph_plan(benchmark)
    space['sample_size']['choices'] = [33]
    with pytest.raises(ValueError, match='active clients'):
        graph_plan(benchmark)


def test_teleportation_details_show_sampling_without_invented_regular_degree(tmp_path):
    registry = load_algorithms(ROOT / 'experiments/leaderboard-settings.json')
    entry = next(item for item in registry['algorithms'] if item['id'] == 'teleportation')
    entry['_benchmark'] = replace(entry['_benchmark'], output=tmp_path / 'teleportation')
    registry = registry | dict(algorithms=[entry])
    details = algorithm_document(entry, registry, 'iid-heterogeneous')
    parameters = {item['parameter']: item['value'] for item in details['experiment_parameters']}
    assert 'Clients n / degree k' not in parameters
    assert parameters['Total clients'] == 32
    assert 'Joint search' in parameters['Active clients / model tokens']
    assert 'complete' in parameters['Token mixing topology']
    assert 'inactive' in parameters['Readout']
    assert details['tuning']['total_trials'] == 120
    assert details['tuning']['hardware_profile'] == 'uniform'
    for scenario in registry['scenarios']:
        comparison = combined_leaderboard(registry, scenario)
        assert comparison['tables'][-1]['rows'][0]['accuracy'] is None
    assert not entry['_benchmark'].output.exists()


def test_teleportation_freezes_active_count_topology_steps_lr_and_checkpoint(tmp_path, monkeypatch):
    registry = load_algorithms(ROOT / 'experiments/leaderboard-settings.json')
    entry = next(item for item in registry['algorithms'] if item['id'] == 'teleportation')
    entry['_benchmark'] = replace(entry['_benchmark'], output=tmp_path / 'teleportation')
    parameters = dict(learning_rate=.03, local_steps=15, sample_size=16, teleportation_topology='hypercube')
    selection = dict(selected_variant='teleportation', selected={'teleportation': .03},
                     selected_parameters={'teleportation': parameters}, selected_time=300,
                     rule='Maximize mean validation accuracy at one shared checkpoint')
    calls = []

    def tune(benchmark, **kwargs):
        benchmark.output.mkdir()
        (benchmark.output / 'selection.json').write_text(json.dumps(selection))
        from dasklearn.lr_tuning import frozen_runs
        calls.append(replace(benchmark, runs=frozen_runs(benchmark, selection)))
        return 0

    monkeypatch.setattr('dasklearn.lr_tuning.execute_tuned_benchmark', tune)
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', lambda benchmark: calls.append(benchmark) or 0)
    assert execute_campaign(registry, entry) == 0
    assert len(calls) == 4
    for benchmark in calls:
        assert benchmark.seeds == [110, 111, 112]
        for run in benchmark.runs:
            assert all(run.options[key] == value for key, value in parameters.items())
    assert {(b.runs[0].options['partitioner'], b.network['profile']) for b in calls} == {
        ('iid', 'uniform'), ('iid', 'heterogeneous'), ('dirichlet', 'uniform'), ('dirichlet', 'heterogeneous')}
    assert all(benchmark.selection == selection for benchmark in calls[1:])
    details = algorithm_document(entry, registry, 'dirichlet01-uniform')
    params = {item['parameter']: item['value'] for item in details['experiment_parameters']}
    assert params['Active clients / model tokens'] == 16 and params['Token mixing topology'] == 'hypercube'
    assert details['tuning']['selected_parameters'] == parameters
    campaign = json.loads((entry['_benchmark'].output / 'campaign.json').read_text())
    assert campaign['status'] == 'completed' and set(campaign['settings_completed']) == set(registry['scenarios'])
