"""Evidence, joint selection, and isolation for the fresh methodology dashboard."""
import copy
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from args import create_parser
from dasklearn.benchmark import load_benchmark, save_results
from dasklearn.leaderboard_details import algorithm_document, load_algorithms, pending_leaderboard
from dasklearn.leaderboard_server import create_server
from dasklearn.lr_tuning import select_learning_rates, tuning_benchmark
from scripts.run_methodology import execute_campaign

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def campaign(tmp_path):
    document = json.loads((ROOT / 'experiments/methodology/dpsgd-kreg.json').read_text())
    document['name'] = 'joint-test'
    document['output'] = 'fresh-results'
    document['settings'].update(dataset='synthetic', model='mlp', dataset_base_path=None, peers=4, device='cpu')
    document['benchmark'].update(budgets=[1, 2], seeds=[10, 11], execution_devices=['cpu'], evaluation_period=1)
    document['runs'] = [dict(name=f'local-h{h:02d}', settings=dict(algorithm='dpsgd', topology='kreg', k=2, local_steps=h)) for h in [1, 5]]
    document['tuning'].update(budget=2, seeds=[0, 1], evaluation_period=1, learning_rates={'local-h01':[.01,.1], 'local-h05':[.01,.1]})
    (tmp_path / 'config.json').write_text(json.dumps(document))
    registry = json.loads((ROOT / 'experiments/leaderboard-algorithms.json').read_text())
    registry['algorithms'] = registry['algorithms'][:1]
    registry['algorithms'][0]['config'] = 'config.json'
    registry['algorithms'][0]['tuning_setting'] = 'iid-uniform'
    (tmp_path / 'algorithms.json').write_text(json.dumps(registry))
    settings = json.loads((ROOT / 'experiments/leaderboard-settings.json').read_text())
    settings['algorithms'] = 'algorithms.json'
    for setting in settings['settings']:
        setting['output'] = f"fresh-results/{'evaluation' if setting['id']=='iid-uniform' else 'evaluation-'+setting['id']}"
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(settings))
    return path, load_algorithms(path)


def completed(benchmark):
    suite = benchmark.manifest() | dict(status='completed')
    for record in suite['runs']:
        score = 80 if record['options']['local_steps']==5 and record['options']['learning_rate']==.1 else 60
        record.update(status='completed', wall_seconds=1, curve=[dict(time=t, accuracy=score if t else 10,
            loss=1., worst_decile_accuracy=None, bytes_sent=t*100) for t in benchmark.times])
    return suite


def test_empty_campaign_shows_only_registered_algorithm_without_fake_scores(campaign):
    _, registry = campaign
    for setting in registry['scenarios']:
        result = pending_leaderboard(registry,setting)
        assert result['status']=='planned'
        assert {run['variant'] for run in result['runs']}=={'dpsgd-kreg-log2'}
        assert all(run['options']['learning_rate'] is None and run['options']['local_steps'] is None for run in result['runs'])
        assert all(table['provisional'] and table['rows'][0]['rank'] is None and table['rows'][0]['accuracy'] is None for table in result['tables'])
        assert result['network']['profile']==registry['scenarios'][setting]['profile']
    assert not registry['algorithms'][0]['_benchmark'].output.exists()


def test_detail_tables_only_publish_complete_endpoint_evidence(campaign):
    _, registry = campaign
    entry = registry['algorithms'][0]
    benchmark = entry['_benchmark']
    detail = algorithm_document(entry,registry,'iid-uniform')
    assert len(detail['tuning']['groups'])==4 and len(detail['tuning']['trials'])==8
    assert all(row['accuracy'] is None and not row['selected'] for row in detail['tuning']['groups'])
    assert all(row['wall_seconds'] is None and row['timed_seeds'] == 0 for row in detail['tuning']['groups'])
    assert all(row['wall_seconds'] is None for row in detail['tuning']['trials'])
    trials = tuning_benchmark(benchmark)
    suite = completed(trials)
    suite['runs'][0]['status']='running'
    trials.output.mkdir(parents=True)
    save_results(trials.output,suite)
    detail = algorithm_document(entry,registry,'iid-heterogeneous')
    assert detail['tuning']['groups'][0]['accuracy'] is None
    assert detail['tuning']['groups'][0]['completed_seeds']==1
    assert all(not row['selected'] for row in detail['tuning']['groups'])
    assert detail['tuning']['selected_learning_rate'] is None
    assert detail['tuning']['trials'][0]['accuracy'] is None
    assert dict((row['parameter'],row['value']) for row in detail['experiment_parameters'])['Displayed hardware profile']=='heterogeneous'



@pytest.mark.parametrize('states,times,expected', [
    (['completed', 'completed'], [20., 40.], 30.),
    (['completed', 'running'], [20., 10.], 15.),
    (['failed', 'interrupted'], [12., 18.], 15.),
    (['diverged', 'planned'], [36., None], 36.),
    (['running', 'planned'], [0., None], 0.),
    (['completed', 'completed'], [None, None], None),
])
def test_tuning_wall_time_uses_recorded_seed_durations(campaign, states, times, expected):
    _, registry = campaign
    entry = registry['algorithms'][0]
    benchmark = tuning_benchmark(entry['_benchmark'])
    report = completed(benchmark)
    for record, status, seconds in zip(report['runs'], states, times):
        record['status'] = status
        if seconds is None:
            record.pop('wall_seconds')
        else:
            record['wall_seconds'] = seconds
    benchmark.output.mkdir(parents=True)
    (benchmark.output / 'leaderboard.json').write_text(json.dumps(report))

    detail = algorithm_document(entry, registry, 'iid-uniform')['tuning']
    assert [trial['wall_seconds'] for trial in detail['trials'][:2]] == times
    group = detail['groups'][0]
    assert group['wall_seconds'] == expected
    assert group['timed_seeds'] == sum(value is not None for value in times)
    assert detail['groups'][1]['wall_seconds'] == 1
    assert detail['groups'][1]['timed_seeds'] == 2


def test_joint_selection_freezes_one_pair_across_all_settings(campaign,monkeypatch):
    _, registry = campaign
    entry = registry['algorithms'][0]
    calls=[]
    def execute(benchmark, on_update=None, **kwargs):
        calls.append(benchmark)
        benchmark.output.mkdir(parents=True)
        result=save_results(benchmark.output,completed(benchmark))
        if on_update:on_update(result)
        return 0
    monkeypatch.setattr('dasklearn.lr_tuning.execute_benchmark',execute)
    monkeypatch.setattr('dasklearn.benchmark.execute_benchmark',execute)
    assert execute_campaign(registry,entry)==0
    assert len(calls)==5 and len(calls[0].runs)==8
    for benchmark in calls[1:]:
        assert len(benchmark.runs)==2
        assert all(run.options['local_steps']==5 and run.options['learning_rate']==.1 for run in benchmark.runs)
        assert all(run.options['momentum']==0 and run.options['batch_size']==32 for run in benchmark.runs)
        assert benchmark.seeds==[10,11]
    selection=json.loads((entry['_benchmark'].output/'selection.json').read_text())
    assert selection['selected_variant']=='local-h05' and selection['selected_learning_rate']==.1
    assert all(benchmark.selection==selection for benchmark in calls[1:])
    detail=algorithm_document(entry,registry,'iid-uniform')
    assert detail['status']=='completed'
    assert sum(row['selected'] for row in detail['tuning']['groups'])==1
    assert detail['tuning']['boundary_winner']
    assert {benchmark.network['profile'] for benchmark in calls[1:]}=={'uniform','heterogeneous'}
    assert {benchmark.runs[0].options['partitioner'] for benchmark in calls[1:]}=={'iid','dirichlet'}


def test_joint_selection_keeps_divergence_and_cannot_use_partial_grids(campaign):
    _,registry=campaign
    benchmark=registry['algorithms'][0]['_benchmark']
    result=completed(tuning_benchmark(benchmark))
    for record in result['runs']:
        if record['options']['local_steps']==1:
            record.update(status='diverged',divergence_time=2,tuning_budget_completed=True)
    selected=select_learning_rates(benchmark,result)
    assert selected['selected_variant']=='local-h05'
    assert all(row['mean_score'] is None for row in selected['scores']['local-h01'])
    partial=copy.deepcopy(result);partial['runs'].pop()
    with pytest.raises(ValueError,match='Incomplete'):
        select_learning_rates(benchmark,partial)
    for record in result['runs']:
        record.update(status='diverged',divergence_time=2,tuning_budget_completed=True)
    with pytest.raises(ValueError,match='No finite joint'):
        select_learning_rates(benchmark,result)


def test_joint_selection_rejects_comparing_different_algorithms(campaign):
    catalog,registry=campaign
    path=registry['algorithms'][0]['_benchmark'].path
    document=json.loads(path.read_text())
    document['runs'][1]['settings']['algorithm']='qgm'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='one algorithm'):
        load_benchmark(path,create_parser())


def test_algorithm_routes_and_untrusted_query_isolation(campaign):
    catalog,_=campaign
    with create_server(catalog=catalog,port=0) as server:
        thread=threading.Thread(target=server.serve_forever);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            for route,content in [('/algorithm/dpsgd-kreg-log2',b'Joint learning-rate'),('/style.css',b'--bg'),('/algorithm.js',b'Authors'),('/implementation?algorithm=dpsgd-kreg-log2',b'class DPSGDClient')]:
                with urlopen(base+route) as response:assert content in response.read()
            for setting in ['iid-uniform','dirichlet01-heterogeneous']:
                with urlopen(base+'/leaderboard.json?setting='+setting) as response:
                    result=json.load(response);assert result['algorithms'][0]['id']=='dpsgd-kreg-log2'
                with urlopen(base+'/leaderboard.csv?setting='+setting) as response:
                    assert b'dpsgd-kreg-log2' in response.read()
            with urlopen(base+'/algorithm.json?algorithm=dpsgd-kreg-log2') as response:
                detail=json.load(response);assert detail['paper']['url'].startswith('https://papers.neurips.cc/')
            with urlopen(base+'/algorithm-config.json?algorithm=dpsgd-kreg-log2') as response:
                assert json.load(response)['tuning']['select_variant'] is True
            for route in ['/algorithm/unknown','/algorithm/../../etc/passwd','/algorithm.json?algorithm=dpsgd-kreg-log2&algorithm=other','/algorithm.json?algorithm=dpsgd-kreg-log2&setting=bad','/implementation?algorithm=dpsgd-kreg-log2&file=../../etc/passwd','/implementation?algorithm=dpsgd-kreg-log2&file=0&file=1']:
                with pytest.raises(HTTPError) as error:urlopen(base+route)
                assert error.value.code==404
        finally:
            server.shutdown();thread.join()


def test_loss_selection_minimizes_endpoint_and_accuracy_selection_maximizes_it(campaign):
    _,registry=campaign
    benchmark=registry['algorithms'][0]['_benchmark']
    result=completed(tuning_benchmark(benchmark))
    for record in result['runs']:
        # Highest accuracy is deliberately the worst NLL, with a misleading
        # earlier loss minimum on the wrong candidate.
        winner=record['options']['local_steps']==1 and record['options']['learning_rate']==.01
        record['curve'][-1]['loss']=.2 if winner else 2.
        record['curve'][0]['loss']=100. if winner else .001
    assert select_learning_rates(benchmark,result)['selected_variant']=='local-h05'
    benchmark.tuning['metric']='loss'
    selected=select_learning_rates(benchmark,result)
    assert selected['selected_variant']=='local-h01'
    assert selected['selected_learning_rate']==.01
    assert selected['rule'].startswith('Minimize mean loss')


def test_peak_selection_uses_maximum_of_seed_means_not_mean_of_seed_maxima(campaign):
    _,registry=campaign
    benchmark=registry['algorithms'][0]['_benchmark']
    benchmark.tuning['metric']='best_mean_accuracy'
    result=completed(tuning_benchmark(benchmark))
    for record in result['runs']:
        steps,rate,seed=record['options']['local_steps'],record['options']['learning_rate'],record['seed']
        values=[40.,40.]
        if steps==1 and rate==.01:values=[100.,0.] if seed==0 else [0.,100.]
        if steps==5 and rate==.1:values=[70.,65.]
        record['curve'][0]['accuracy']=99.  # Initialization is not a trained checkpoint.
        record['curve'][1]['accuracy'],record['curve'][2]['accuracy']=values
    selected=select_learning_rates(benchmark,result)
    assert selected['selected_variant']=='local-h05'
    assert selected['selected_learning_rate']==.1 and selected['selected_time']==1
    assert selected['scores']['local-h01'][0]['mean_score']==50.
    assert selected['scores']['local-h01'][0]['selected_time']==1
    assert selected['scores']['local-h05'][1]['mean_score']==70.
    result['runs'][0]['curve'].pop(1)
    with pytest.raises(ValueError,match='every declared finite checkpoint'):
        select_learning_rates(benchmark,result)
