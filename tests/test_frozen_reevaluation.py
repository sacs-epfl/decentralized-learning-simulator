"""Longer evaluations replace held-out data while keeping tuning immutable."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from dasklearn.benchmark import save_results
from dasklearn.leaderboard_details import load_algorithms
from scripts.run_methodology import reevaluate_campaign

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def campaign(tmp_path):
    registry = load_algorithms(ROOT / 'experiments/leaderboard-cifar100-settings.json')
    entry = registry['algorithms'][0]
    benchmark = replace(entry['_benchmark'], output=tmp_path / 'campaign')
    entry['_benchmark'] = benchmark
    benchmark.output.mkdir()
    previous = replace(benchmark, budgets=[600, 1800, 3600], times=list(range(0, 3601, 300)),
        runs=[replace(run, options=run.options | dict(duration=3600)) for run in benchmark.runs])
    (benchmark.output / 'resolved.json').write_text(json.dumps(previous.manifest()))
    selection = dict(selected_variant='dpsgd-kreg-log2', selected={'dpsgd-kreg-log2': .028495662538780223},
        selected_parameters={'dpsgd-kreg-log2': dict(local_steps=20, learning_rate=.028495662538780223)},
        selected_time=3600, specification=benchmark.tuning)
    (benchmark.output / 'selection.json').write_text(json.dumps(selection))
    (benchmark.output / 'tuning').mkdir()
    (benchmark.output / 'tuning' / 'keep.json').write_text('original validation observations')
    (benchmark.output / 'leaderboard.json').write_text('old held-out results')
    for setting in registry['scenarios']:
        directory = benchmark.output / ('evaluation' if setting == entry['tuning_setting'] else 'evaluation-' + setting)
        directory.mkdir()
        (directory / 'obsolete').write_text('old one-hour evaluation')
    return registry, entry, selection


def test_reevaluation_replaces_only_held_out_runs_and_preserves_all_frozen_options(campaign, monkeypatch):
    registry, entry, selection = campaign
    output = entry['_benchmark'].output
    before = {name: (output / name).read_bytes() for name in ['selection.json', 'resolved.json', 'tuning/keep.json']}
    calls = []
    def evaluate(benchmark):
        calls.append(benchmark)
        assert not list(output.glob('evaluation*/obsolete'))
        assert not (output / 'leaderboard.json').exists()
        assert benchmark.budgets[-1] == 18000 and benchmark.times[-1] == 18000
        assert benchmark.selection == selection and benchmark.seeds == [110, 111, 112]
        for run in benchmark.runs:
            assert run.options['duration'] == 18000 and run.options['evaluation_split'] == 'test'
            assert run.options['local_steps'] == 20 and run.options['learning_rate'] == .028495662538780223
        benchmark.output.mkdir()
        suite = benchmark.manifest() | dict(status='completed')
        for run in suite['runs']:
            run.update(status='completed', wall_seconds=1, curve=[dict(time=t, accuracy=20., loss=3.,
                bytes_sent=t * 100, worst_decile_accuracy=None) for t in benchmark.times])
        save_results(benchmark.output, suite)
        return 0
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', evaluate)
    assert reevaluate_campaign(registry, entry, replace_evaluations=True) == 0
    assert len(calls) == 4
    assert {(b.runs[0].options['partitioner'], b.network['profile']) for b in calls} == {
        ('iid', 'uniform'), ('iid', 'heterogeneous'), ('dirichlet', 'uniform'), ('dirichlet', 'heterogeneous')}
    assert {name: (output / name).read_bytes() for name in before} == before
    state = json.loads((output / 'campaign.json').read_text())
    assert state['status'] == 'completed' and state['evaluation_budget'] == 18000
    assert set(state['settings_completed']) == set(registry['scenarios'])


@pytest.mark.parametrize('changed', [False, True])
def test_reevaluation_checks_authorization_and_compatibility_before_removing_results(campaign, changed):
    registry, entry, _ = campaign
    if changed:
        benchmark = entry['_benchmark']
        entry['_benchmark'] = replace(benchmark, runs=[replace(run, options=run.options | dict(batch_size=16)) for run in benchmark.runs])
    with pytest.raises(ValueError, match='preserve|replace-evaluations'):
        reevaluate_campaign(registry, entry, replace_evaluations=changed)
    assert len(list(entry['_benchmark'].output.glob('evaluation*/obsolete'))) == 4


def test_failed_evaluation_stops_the_campaign_and_is_reported(campaign, monkeypatch):
    registry, entry, _ = campaign
    calls = []
    def fail(benchmark):
        calls.append(benchmark)
        return 1
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', fail)
    assert reevaluate_campaign(registry, entry, replace_evaluations=True) == 1
    assert len(calls) == 1
    output = entry['_benchmark'].output
    assert json.loads((output / 'campaign.json').read_text())['status'] == 'failed'
    assert json.loads((output / 'pipeline.json').read_text())['status'] == 'failed'
    assert (output / 'tuning/keep.json').read_text() == 'original validation observations'
