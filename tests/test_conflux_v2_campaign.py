"""Original V2 search, charged snapshots, and frozen four-scenario transfer."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from args import create_parser
from dasklearn.benchmark import load_benchmark
from dasklearn.leaderboard_details import algorithm_document, combined_leaderboard, load_algorithms
from scripts.run_methodology import execute_campaign, graph_plan

ROOT = Path(__file__).resolve().parents[1]


def test_v2_declares_valid_placements_and_equal_search_budget():
    benchmark = load_benchmark(ROOT / 'experiments/methodology/conflux-v2-tpe-dirichlet01-h20.json', create_parser())
    space = benchmark.tuning['search_spaces']['conflux-v2']
    assert set(space) == {'learning_rate', 'local_steps', 'sample_size', 'chunks_in_sample', 'relay_count'}
    assert benchmark.tuning['n_trials'] == 40 and benchmark.tuning['budget'] == 900
    assert benchmark.tuning['metric'] == 'best_mean_accuracy'
    assert benchmark.tuning['seeds'] == [100, 101, 102] and benchmark.seeds == [110, 111, 112]
    assert benchmark.execution_devices == ['cuda:0', 'cuda:1'] * 3
    assert benchmark.runs[0].options['success_fraction'] == 1
    assert benchmark.runs[0].options['immediate_uploads']
    plans = graph_plan(benchmark)
    assert len(plans) == 6
    for record in plans.values():
        assert record['real_nodes'] == 32 and record['full_participation']
        assert len(record['samples']) == 5 and len(record['layouts']) == 36
        for sample in record['samples']:
            assert len(set(sample['first_sample'])) == len(set(sample['next_sample'])) == sample['active_clients']
            assert set(sample['first_sample']) <= set(range(32))
        for layout in record['layouts']:
            assert len(layout['owners']) == layout['chunks']
            assert len(set(layout['owners'])) == min(layout['chunks'], layout['reducer_cap'])
    space['sample_size']['choices'] = [33]
    with pytest.raises(ValueError, match='sample_size'):
        graph_plan(benchmark)


def test_v2_freezes_every_parameter_and_checkpoint_across_four_settings(tmp_path, monkeypatch):
    registry = load_algorithms(ROOT / 'experiments/leaderboard-settings.json')
    entry = next(item for item in registry['algorithms'] if item['id'] == 'conflux-v2')
    entry['_benchmark'] = replace(entry['_benchmark'], output=tmp_path / 'v2')
    parameters = dict(learning_rate=.03, local_steps=15, sample_size=16, chunks_in_sample=8, relay_count=4)
    selection = dict(selected_variant='conflux-v2', selected={'conflux-v2': .03},
                     selected_parameters={'conflux-v2': parameters}, selected_time=300,
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
        assert benchmark.seeds == [110, 111, 112] and 300 in benchmark.times
        for run in benchmark.runs:
            assert all(run.options[key] == value for key, value in parameters.items())
            assert run.options['success_fraction'] == 1 and run.options['immediate_uploads']
    assert {(b.runs[0].options['partitioner'], b.network['profile']) for b in calls} == {
        ('iid', 'uniform'), ('iid', 'heterogeneous'), ('dirichlet', 'uniform'), ('dirichlet', 'heterogeneous')}
    assert all(b.selection == selection for b in calls[1:])
    details = algorithm_document(entry, registry, 'iid-heterogeneous')
    params = {item['parameter']: item['value'] for item in details['experiment_parameters']}
    assert 'Clients n / degree k' not in params
    assert params['Active clients'] == 16 and params['Model chunks'] == 8 and params['Reducer cap'] == 4
    assert details['tuning']['selected_parameters'] == parameters
    campaign = json.loads((entry['_benchmark'].output / 'campaign.json').read_text())
    assert campaign['status'] == 'completed' and set(campaign['settings_completed']) == set(registry['scenarios'])


def test_v2_pending_details_never_invent_scores(tmp_path):
    registry = load_algorithms(ROOT / 'experiments/leaderboard-settings.json')
    entry = next(item for item in registry['algorithms'] if item['id'] == 'conflux-v2')
    entry['_benchmark'] = replace(entry['_benchmark'], output=tmp_path / 'v2')
    registry = registry | dict(algorithms=[entry])
    details = algorithm_document(entry, registry, 'dirichlet01-uniform')
    assert details['tuning']['total_trials'] == 120
    assert details['paper'] is None and details['tuning']['selected_parameters'] is None
    for scenario in registry['scenarios']:
        comparison = combined_leaderboard(registry, scenario)
        assert comparison['tables'][-1]['rows'][0]['accuracy'] is None
    assert not entry['_benchmark'].output.exists()
