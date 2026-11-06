"""DeMo campaign transport, compression search, and frozen evaluation settings."""
from dataclasses import replace
import json
from pathlib import Path

from args import create_parser
from dasklearn.benchmark import load_benchmark
from dasklearn.leaderboard_details import algorithm_document, combined_leaderboard, load_algorithms
from scripts.run_methodology import execute_campaign, graph_plan

ROOT = Path(__file__).resolve().parents[1]


def test_demo_campaign_declares_ring_and_searches_compression():
    benchmark = load_benchmark(ROOT / 'experiments/methodology/demo-tpe-dirichlet01-h20.json', create_parser())
    space = benchmark.tuning['search_spaces']['demo']
    assert set(space) == {'learning_rate', 'local_steps', 'demo_decay', 'demo_chunk', 'demo_topk'}
    assert benchmark.tuning['budget'] == 900 and benchmark.tuning['n_trials'] == 40
    assert benchmark.tuning['seeds'] == [100, 101, 102] and benchmark.seeds == [110, 111, 112]
    assert benchmark.execution_devices == ['cuda:0', 'cuda:1'] * 3
    assert benchmark.tuning['batch_size'] * len(benchmark.tuning['seeds']) == 6
    for record in graph_plan(benchmark).values():
        assert record['directed'] and record['connected'] and record['stages_per_round'] == 31
        assert record['edges'] == [[node, (node + 1) % 32] for node in range(32)]
        assert record['gradient_accumulation'] and not record['resampled_each_round']


def test_demo_pending_details_describe_accumulation_and_compression(tmp_path):
    registry = load_algorithms(ROOT / 'experiments/leaderboard-settings.json')
    entry = next(item for item in registry['algorithms'] if item['id'] == 'demo')
    entry['_benchmark'] = replace(entry['_benchmark'], output=tmp_path / 'demo')
    registry = registry | dict(algorithms=[entry])
    launch = tmp_path / 'demo-launch'
    launch.mkdir()
    (launch / 'queue.json').write_text(json.dumps(dict(status='queued', phase='waiting for epidemic-kreg-log2 to finish')))
    details = algorithm_document(entry, registry, 'iid-heterogeneous')
    assert details['status'] == 'queued' and details['tuning']['total_trials'] == 120
    parameters = {item['parameter']: item['value'] for item in details['experiment_parameters']}
    assert 'Local SGD steps H' not in parameters and 'Clients n / degree k' not in parameters
    assert 'Gradient accumulation minibatches H' in parameters
    assert all(label in parameters for label in ['Compression residual decay', 'DCT block size', 'Coefficients per block'])
    comparison = combined_leaderboard(registry, 'iid-heterogeneous')
    assert comparison['tables'][-1]['rows'][0]['accuracy'] is None
    assert not entry['_benchmark'].output.exists()


def test_demo_freezes_every_compression_parameter_for_all_scenarios(tmp_path, monkeypatch):
    registry = load_algorithms(ROOT / 'experiments/leaderboard-settings.json')
    entry = next(item for item in registry['algorithms'] if item['id'] == 'demo')
    entry['_benchmark'] = replace(entry['_benchmark'], output=tmp_path / 'demo')
    parameters = dict(learning_rate=.0003, local_steps=10, demo_decay=.99, demo_chunk=128, demo_topk=2)
    selection = dict(selected_variant='demo', selected={'demo': parameters['learning_rate']},
                     selected_parameters={'demo': parameters}, selected_time=300)
    calls = []

    def tune(benchmark, **kwargs):
        benchmark.output.mkdir()
        (benchmark.output / 'selection.json').write_text(json.dumps(selection))
        # The tuning pipeline evaluates the source scenario with frozen_runs.
        from dasklearn.lr_tuning import frozen_runs
        calls.append(replace(benchmark, runs=frozen_runs(benchmark, selection)))
        return 0

    def evaluate(benchmark):
        calls.append(benchmark)
        return 0

    monkeypatch.setattr('dasklearn.lr_tuning.execute_tuned_benchmark', tune)
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', evaluate)
    assert execute_campaign(registry, entry) == 0
    assert len(calls) == 4
    for benchmark in calls:
        assert benchmark.seeds == [110, 111, 112]
        for run in benchmark.runs:
            assert all(run.options[key] == value for key, value in parameters.items())
    assert {(b.runs[0].options['partitioner'], b.network['profile']) for b in calls} == {
        ('iid', 'uniform'), ('iid', 'heterogeneous'), ('dirichlet', 'uniform'), ('dirichlet', 'heterogeneous')}
    for benchmark in calls[1:]:
        assert benchmark.selection == selection
    campaign = json.loads((entry['_benchmark'].output / 'campaign.json').read_text())
    assert campaign['status'] == 'completed'
    assert set(campaign['settings_completed']) == set(registry['scenarios'])
