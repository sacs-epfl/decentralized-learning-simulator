"""Adaptive proposals, paired scoring, frozen evaluation, and live progress."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from args import create_parser
from dasklearn.benchmark import load_benchmark, save_results
from dasklearn.lr_tuning import execute_tuned_benchmark, frozen_runs
from dasklearn.leaderboard_details import algorithm_document, pending_leaderboard
from dasklearn.tpe_tuning import candidate_scores, search_progress, select_search, tuning_suite
from scripts.run_methodology import execute_campaign

ROOT = Path(__file__).resolve().parents[1]


def configuration(tmp_path, *, algorithms=('dpsgd',), **tuning):
    document = dict(version=1, name='tpe-test', output='results',
        benchmark=dict(budgets=[60, 900], seeds=[110, 111, 112], evaluation_period=60,
                       execution_devices=['cpu', 'cpu'], evaluation_model='global'),
        settings=dict(dataset='synthetic', model='mlp', peers=4, device='cpu', evaluation_split='test'),
        runs=[dict(name=name, settings=dict(algorithm=name, topology='ring')) for name in algorithms],
        tuning=dict(method='tpe', budget=900, seeds=[100, 101, 102], evaluation_period=60,
                    evaluation_examples=100, metric='best_mean_accuracy', select_variant=len(algorithms)==1,
                    n_trials=4, n_startup_trials=2, batch_size=2, sampler_seed=17,
                    search_spaces={name: dict(learning_rate=dict(type='float', low=.001, high=1., log=True),
                                             local_steps=dict(type='categorical', choices=[1, 5, 10])) for name in algorithms}))
    document['tuning'].update(tuning)
    path = tmp_path/'tpe.json'
    path.write_text(json.dumps(document))
    return load_benchmark(path, create_parser())


def completed(benchmark):
    suite = benchmark.manifest() | dict(status='completed')
    for record in suite['runs']:
        # Nonlinear joint objective, unrelated to the proposal implementation.
        score = 80-abs(record['options']['learning_rate']-.1)*20-abs(record['options']['local_steps']-5)
        record.update(status='completed', wall_seconds=1, curve=[dict(time=t, accuracy=score if t else 99,
            loss=1., worst_decile_accuracy=None, bytes_sent=t*100) for t in benchmark.times])
    return suite


def fake_runner(calls, *, invert=False, failure=False):
    def execute(benchmark, on_update=None, **kwargs):
        calls.append(benchmark)
        benchmark.output.mkdir(parents=True)
        suite = completed(benchmark)
        if invert:
            for record in suite['runs']:
                for point in record['curve'][1:]:
                    point['accuracy'] = 100-point['accuracy']
        if failure:
            suite['status'] = 'failed'
            suite['runs'][0]['status'] = 'failed'
        result = save_results(benchmark.output, suite)
        if on_update:
            on_update(result)
        return int(failure)
    return execute


def test_active_campaign_declares_full_paired_search_without_accessing_data(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Planning loaded data or models')
    monkeypatch.setattr('dasklearn.datasets.create_dataset', forbidden)
    monkeypatch.setattr('dasklearn.models.create_model', forbidden)
    benchmark = load_benchmark(ROOT/'experiments/methodology/dpsgd-kreg-tpe.json', create_parser())
    assert benchmark.tuning['method'] == 'tpe'
    assert benchmark.tuning['budget'] == 900 and benchmark.tuning['seeds'] == [100, 101, 102]
    assert benchmark.tuning['metric'] == 'best_mean_accuracy'
    assert benchmark.tuning['trials_per_algorithm'] == 120
    assert benchmark.tuning['simulated_seconds_per_algorithm'] == 108000
    assert benchmark.tuning['batch_size']*3 > len(benchmark.execution_devices)


@pytest.mark.parametrize('patch,match', [
    ({'learning_rates': {'dpsgd': [.1]}}, 'instead'),
    ({'search_spaces': {}}, 'exactly'),
    ({'n_trials': True}, 'integer'),
    ({'n_startup_trials': 4}, 'smaller'),
    ({'batch_size': 3}, 'exceed'),
    ({'sampler_seed': -1}, 'nonnegative'),
    ({'search_spaces': {'dpsgd': {'dataset': {'type':'categorical', 'choices':['synthetic']}}}}, 'searchable'),
    ({'search_spaces': {'dpsgd': {'algorithm': {'type':'categorical', 'choices':['qgm']}}}}, 'searchable'),
    ({'search_spaces': {'dpsgd': {'local_steps': {'type':'int', 'low':0, 'high':10}}}}, 'positive'),
    ({'search_spaces': {'dpsgd': {'local_steps': {'type':'float', 'low':1., 'high':10.}}}}, 'integer'),
    ({'search_spaces': {'dpsgd': {'learning_rate': {'type':'float', 'low':0, 'high':1, 'log':True}}}}, 'positive'),
    ({'search_spaces': {'dpsgd': {'learning_rate': {'type':'float', 'low':.01, 'high':1, 'log':True, 'step':.01}}}}, 'step/log'),
    ({'search_spaces': {'dpsgd': {'local_steps': {'type':'categorical', 'choices':[1,1]}}}}, 'unique'),
    ({'search_spaces': {'dpsgd': {'local_steps': {'type':'int', 'low':1, 'high':10, 'step':2}}}}, 'aligned'),
])
def test_invalid_search_is_rejected_before_output(tmp_path, patch, match):
    with pytest.raises(ValueError, match=match):
        configuration(tmp_path, **patch)
    assert not (tmp_path/'results').exists()


def two_candidates():
    return [dict(id=f'tpe{index:05d}', study='joint', number=index, variant='dpsgd',
                 parameters=dict(learning_rate=.01 if index==0 else .1, local_steps=1 if index==0 else 5),
                 status='completed', sampling_phase='startup') for index in range(2)]


def test_selection_preserves_shared_checkpoint_objective_and_freezes_all_parameters(tmp_path):
    benchmark = configuration(tmp_path)
    benchmark.tuning['n_trials'] = 2
    candidates = two_candidates()
    suite = completed(tuning_suite(benchmark, candidates))
    for record in suite['runs']:
        early, late = (100., 0.) if record['seed']==100 else (0., 100.) if record['seed']==101 else (50., 50.)
        if record['options']['local_steps']==5:
            early, late = 70., 65.
        for point in record['curve'][1:]:
            point['accuracy'] = early if point['time']==60 else late
    selected = select_search(benchmark, suite, candidates)
    assert selected['selected_time'] == 60
    assert selected['scores']['dpsgd'][0]['mean_score'] == 50
    assert selected['scores']['dpsgd'][1]['mean_score'] == 70
    assert selected['selected_candidates'] == {'dpsgd': 'tpe00001'}
    assert selected['selection_split'] == 'validation'
    for run in frozen_runs(benchmark, selected):
        assert run.options['local_steps'] == 5 and run.options['learning_rate'] == .1
        assert run.options['evaluation_split'] == 'test' and run.seed in [110, 111, 112]


@pytest.mark.parametrize('mutation', ['missing_seed', 'duplicate_seed', 'wrong_parameter', 'partial_curve', 'failed', 'test_split'])
def test_partial_or_mismatched_observations_cannot_enter_tpe(tmp_path, mutation):
    benchmark = configuration(tmp_path)
    candidates = two_candidates()
    suite = completed(tuning_suite(benchmark, candidates))
    if mutation=='missing_seed': suite['runs'].pop()
    elif mutation=='duplicate_seed': suite['runs'][-1] = suite['runs'][0]
    elif mutation=='wrong_parameter': suite['runs'][0]['options']['local_steps'] = 50
    elif mutation=='partial_curve': suite['runs'][0]['curve'].pop(1)
    elif mutation=='failed': suite['runs'][0]['status'] = 'failed'
    else: suite['runs'][0]['options']['evaluation_split'] = 'test'
    with pytest.raises(ValueError):
        candidate_scores(benchmark, suite, candidates)


def test_full_budget_divergence_is_ineligible_and_visible(tmp_path):
    benchmark = configuration(tmp_path)
    benchmark.tuning['n_trials'] = 2
    candidates = two_candidates()
    suite = completed(tuning_suite(benchmark, candidates))
    for row in suite['runs'][:3]:
        row.update(status='diverged', divergence_time=900, tuning_budget_completed=True)
    scores = candidate_scores(benchmark, suite, candidates)
    assert scores['tpe00000']['mean_score'] is None
    assert select_search(benchmark, suite, candidates)['selected_candidates']['dpsgd'] == 'tpe00001'
    suite['runs'][0]['divergence_time'] = 60
    with pytest.raises(ValueError, match='exact global validation budget'):
        candidate_scores(benchmark, suite, candidates)


def test_optuna_proposals_learn_from_completed_batches_and_are_reproducible(tmp_path, monkeypatch):
    searches = []
    for name, invert in [('first',False), ('repeat',False), ('reversed',True)]:
        path = tmp_path/name
        path.mkdir()
        benchmark = configuration(path, n_trials=8)
        calls = []
        runner = fake_runner(calls, invert=invert)
        monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', runner)
        monkeypatch.setattr('dasklearn.lr_tuning.execute_benchmark', runner)
        assert execute_tuned_benchmark(benchmark) == 0
        search = json.loads((benchmark.output/'tuning/search.json').read_text())
        searches.append([row['parameters'] for row in search['candidates']])
        assert [row['sampling_phase'] for row in search['candidates']] == ['startup']*2+['adaptive']*6
        assert all(run.options['duration']==900 for stage in calls[:-1] for run in stage.runs)
        assert all({run.seed for run in stage.runs}=={100,101,102} for stage in calls[:-1])
        assert len(calls)==5 and all(len(stage.runs)==6 for stage in calls[:-1])
        assert search['pruning'] is False
    assert searches[0] == searches[1]
    assert searches[0][:2] == searches[2][:2]
    assert searches[0][2:] != searches[2][2:]


def test_multiple_algorithms_share_worker_slots_but_keep_separate_studies(tmp_path, monkeypatch):
    benchmark = configuration(tmp_path, algorithms=('dpsgd','qgm'))
    calls = []
    runner = fake_runner(calls)
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', runner)
    monkeypatch.setattr('dasklearn.lr_tuning.execute_benchmark', runner)
    assert execute_tuned_benchmark(benchmark)==0
    assert len(calls)==3
    assert all(len(stage.runs)==12 for stage in calls[:-1])
    assert all({run.options['algorithm'] for run in stage.runs}=={'dpsgd','qgm'} for stage in calls)
    assert all(stage.execution_devices==['cpu','cpu'] for stage in calls)
    selection = json.loads((benchmark.output/'selection.json').read_text())
    assert set(selection['selected_parameters'])=={'dpsgd','qgm'}
    assert set(selection['selected_times'])=={'dpsgd','qgm'}
    for run in calls[-1].runs:
        assert all(run.options[key]==value for key,value in selection['selected_parameters'][run.variant].items())


def test_variant_specific_parameters_and_integer_ranges_are_conditional(tmp_path, monkeypatch):
    benchmark = configuration(tmp_path, algorithms=('dpsgd','qgm'), n_trials=8)
    document = json.loads(benchmark.path.read_text())
    document['runs'][1]['name'] = 'momentum'
    document['runs'][1]['settings']['algorithm'] = 'dpsgd'
    space = document['tuning']['search_spaces'].pop('qgm')
    space['momentum'] = dict(type='float', low=0., high=.9)
    document['tuning']['search_spaces']['momentum'] = space
    document['tuning']['search_spaces']['dpsgd']['local_steps'] = dict(type='int', low=1, high=9, step=2)
    document['tuning']['select_variant'] = True
    benchmark.path.write_text(json.dumps(document))
    benchmark = load_benchmark(benchmark.path, create_parser())
    calls = []
    runner = fake_runner(calls)
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', runner)
    monkeypatch.setattr('dasklearn.lr_tuning.execute_benchmark', runner)
    assert execute_tuned_benchmark(benchmark)==0
    search = json.loads((benchmark.output/'tuning/search.json').read_text())
    assert {row['variant'] for row in search['candidates']}=={'dpsgd','momentum'}
    assert {row['study'] for row in search['candidates']}=={'joint'}
    for row in search['candidates']:
        assert ('momentum' in row['parameters'])==(row['variant']=='momentum')
        if row['variant']=='dpsgd':
            assert row['parameters']['local_steps'] in [1,3,5,7,9]
    assert len(calls[-1].runs)==3


def test_all_divergent_startup_batch_still_allows_finite_adaptive_candidates(tmp_path, monkeypatch):
    benchmark = configuration(tmp_path)
    calls = []
    def runner(stage, on_update=None, **kwargs):
        calls.append(stage)
        stage.output.mkdir(parents=True)
        suite = completed(stage)
        if len(calls)==1:
            for row in suite['runs']:
                row.update(status='diverged', divergence_time=900, tuning_budget_completed=True, curve=row['curve'][:-1])
        result = save_results(stage.output, suite)
        if on_update:
            on_update(result)
        return 0
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', runner)
    monkeypatch.setattr('dasklearn.lr_tuning.execute_benchmark', runner)
    assert execute_tuned_benchmark(benchmark)==0
    search = json.loads((benchmark.output/'tuning/search.json').read_text())
    assert [row['status'] for row in search['candidates']]==['diverged','diverged','completed','completed']
    assert all(row['mean_score'] is None for row in search['candidates'][:2])
    selection = json.loads((benchmark.output/'selection.json').read_text())
    assert selection['selected_candidates']['dpsgd'] in ['tpe00002','tpe00003']


def test_failed_batch_blocks_selection_and_held_out_evaluation(tmp_path, monkeypatch):
    benchmark = configuration(tmp_path)
    calls = []
    runner = fake_runner(calls, failure=True)
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', runner)
    monkeypatch.setattr('dasklearn.lr_tuning.execute_benchmark', runner)
    assert execute_tuned_benchmark(benchmark)==1
    assert len(calls)==1
    assert not (benchmark.output/'selection.json').exists()
    assert not (benchmark.output/'evaluation').exists()
    assert json.loads((benchmark.output/'tuning/search.json').read_text())['status']=='failed'


def test_recovery_replays_the_same_proposals_and_retries_only_incomplete_seeds(tmp_path, monkeypatch):
    reference_dir, interrupted_dir = tmp_path/'reference', tmp_path/'interrupted'
    reference_dir.mkdir(); interrupted_dir.mkdir()
    reference = configuration(reference_dir, n_trials=8)
    benchmark = configuration(interrupted_dir, n_trials=8)
    reference_calls = []
    runner = fake_runner(reference_calls)
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', runner)
    monkeypatch.setattr('dasklearn.lr_tuning.execute_benchmark', runner)
    assert execute_tuned_benchmark(reference) == 0
    calls = []
    def interrupted(stage, on_update=None, **kwargs):
        calls.append(stage)
        stage.output.mkdir(parents=True)
        suite = completed(stage)
        if len(calls) == 3:
            suite['status'] = 'failed'
            for row in suite['runs'][-3:-1]:
                row.update(status='failed', curve=row['curve'][:1])
        result = save_results(stage.output, suite)
        if on_update: on_update(result)
        return int(suite['status'] == 'failed')
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', interrupted)
    assert execute_tuned_benchmark(benchmark) == 1
    original_files = {path:path.read_bytes() for path in (benchmark.output/'tuning').glob('batch-*/*') if path.is_file()}
    old_search = (benchmark.output/'tuning/search.json').read_bytes()
    recovery_calls = []
    runner = fake_runner(recovery_calls)
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', runner)
    monkeypatch.setattr('dasklearn.lr_tuning.execute_benchmark', runner)
    assert execute_tuned_benchmark(benchmark, resume_tuning=True) == 0
    assert [run.name for run in recovery_calls[0].runs] == ['tpe00005-s100', 'tpe00005-s101']
    assert recovery_calls[0].output.name == 'batch-0002-recovery-0001'
    assert len(recovery_calls) == 3  # retry two seeds, next batch, final evaluation
    restored = json.loads((benchmark.output/'tuning/search.json').read_text())
    expected = json.loads((reference.output/'tuning/search.json').read_text())
    assert restored['candidates'] == expected['candidates']
    assert restored['resumptions'][0]['retained_seed_runs'] == 16
    assert all(path.read_bytes() == content for path,content in original_files.items())
    snapshots = list(benchmark.output.glob('recovery-*'))
    assert len(snapshots) == 1 and (snapshots[0]/'tuning/search.json').read_bytes() == old_search


def test_recovery_rejects_a_changed_declaration(tmp_path, monkeypatch):
    benchmark = configuration(tmp_path)
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', fake_runner([], failure=True))
    assert execute_tuned_benchmark(benchmark) == 1
    changed = replace(benchmark, tuning=benchmark.tuning | dict(n_trials=8))
    before = (benchmark.output/'tuning/search.json').read_bytes()
    with pytest.raises(ValueError, match='original benchmark'):
        execute_tuned_benchmark(changed, resume_tuning=True)
    assert (benchmark.output/'tuning/search.json').read_bytes() == before


def test_progress_tracks_best_full_candidate_and_never_partial_seed_maxima(tmp_path):
    benchmark = configuration(tmp_path)
    candidates = two_candidates()
    candidates[0].update(mean_score=60, selected_time=60)
    candidates[1].update(status='running')
    report = completed(tuning_suite(benchmark, candidates))
    report['runs'][-1]['status'] = 'running'
    progress = search_progress(benchmark.tuning, dict(candidates=candidates), report)
    assert progress['completed']==1 and progress['proposed']==2 and progress['total']==4
    assert progress['running_seeds']==1 and progress['completed_seeds']==5
    assert progress['studies'][0]['phase']=='startup'
    assert progress['history']==[dict(study='joint', candidate_id='tpe00000', number=1, score=60,
                                     best_score=60, sampling_phase='startup')]
    assert progress['studies'][0]['best_candidate']['parameters']['local_steps']==1


def test_algorithm_details_show_optuna_progress_and_campaign_freezes_parameters_everywhere(tmp_path, monkeypatch):
    benchmark = configuration(tmp_path)
    registry = json.loads((ROOT/'experiments/leaderboard-algorithms.json').read_text())
    registry['algorithms'] = registry['algorithms'][:1]
    entry = registry['algorithms'][0]
    entry['_benchmark'] = benchmark
    detail = algorithm_document(entry, registry, 'iid-uniform')
    assert detail['tuning']['method']=='tpe' and detail['tuning']['total_trials']==12
    assert detail['tuning']['progress']['history']==[] and detail['tuning']['groups']==[]
    calls = []
    runner = fake_runner(calls)
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark', runner)
    monkeypatch.setattr('dasklearn.lr_tuning.execute_benchmark', runner)
    assert execute_campaign(registry, entry)==0
    detail = algorithm_document(entry, registry, 'iid-heterogeneous')
    progress = detail['tuning']['progress']
    assert progress['completed']==4 and progress['completed_seeds']==12 and progress['running_seeds']==0
    assert progress['studies'][0]['phase']=='completed' and len(progress['history'])==4
    assert [point['best_score'] for point in progress['history']]==sorted(point['best_score'] for point in progress['history'])
    assert sum(row['selected'] for row in detail['tuning']['groups'])==1
    assert len(detail['tuning']['trials'])==12 and detail['tuning']['selected_time'] in benchmark.times
    assert all(row['wall_seconds'] == 1 and row['timed_seeds'] == 3 for row in detail['tuning']['groups'])
    assert all(row['wall_seconds'] == 1 for row in detail['tuning']['trials'])
    selected = detail['tuning']['selected_parameters']
    assert len(calls)==6  # Two tuning batches, followed by all four scenarios.
    for stage in calls[2:]:
        for run in stage.runs:
            assert all(run.options[key]==value for key,value in selected.items())
            assert run.options['evaluation_split']=='test' and run.seed in [110,111,112]
    pending = pending_leaderboard(registry, 'iid-uniform')
    assert all(run['options']['local_steps']==selected['local_steps'] for run in pending['runs'])


def test_real_parallel_cpu_pipeline_evaluates_all_seeds_before_freezing(tmp_path):
    benchmark = configuration(tmp_path, budget=2, evaluation_period=1)
    document = json.loads(benchmark.path.read_text())
    document['benchmark'].update(budgets=[1,2], evaluation_period=1)
    benchmark.path.write_text(json.dumps(document))
    result = subprocess.run([sys.executable, str(ROOT/'scripts/run_benchmark.py'), str(benchmark.path)],
                            cwd=ROOT, text=True, capture_output=True, timeout=180)
    assert result.returncode==0, result.stdout+result.stderr
    selection = json.loads((benchmark.output/'selection.json').read_text())
    report = json.loads((benchmark.output/'tuning/leaderboard.json').read_text())
    search = json.loads((benchmark.output/'tuning/search.json').read_text())
    final = json.loads((benchmark.output/'evaluation/leaderboard.json').read_text())
    assert len(report['runs'])==12 and len(final['runs'])==3
    assert {row['executor_slot'] for row in report['runs']}=={0,1}
    assert all(row['status']=='completed' and row['curve'][-1]['time']==2 for row in report['runs'])
    assert select_search(load_benchmark(benchmark.path,create_parser()), report,search['candidates'])==selection
    for row in final['runs']:
        assert row['seed'] in [110,111,112] and row['options']['evaluation_split']=='test'
        assert all(row['options'][key]==value for key,value in selection['selected_parameters']['dpsgd'].items())
