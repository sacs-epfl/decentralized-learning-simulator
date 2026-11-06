"""Shatter graph declarations, independent results, and combined live comparisons."""
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import threading
from urllib.request import urlopen

import pytest

from args import create_parser
from dasklearn.benchmark import load_benchmark, save_results
from dasklearn.leaderboard_details import algorithm_document, combined_leaderboard, load_algorithms
from dasklearn.leaderboard_server import create_server
from scripts.run_methodology import graph_plan

ROOT = Path(__file__).resolve().parents[1]


def test_shatter_plan_uses_32_real_nodes_and_512_virtual_nodes():
    benchmark = load_benchmark(ROOT/'experiments/methodology/shatter-k16-rlog2-tpe-dirichlet01-h20.json', create_parser())
    assert benchmark.execution_devices==['cuda:0','cuda:1']*3
    assert benchmark.tuning['batch_size']*len(benchmark.tuning['seeds'])==6
    assert benchmark.tuning['budget']==900 and benchmark.tuning['n_trials']==40
    assert benchmark.tuning['search_spaces']['shatter-k16-rlog2']['local_steps']['choices']==[1,5,10,15,20]
    for record in graph_plan(benchmark).values():
        assert record['real_nodes']==32 and record['graph_nodes']==512
        assert record['virtual_nodes_per_real']==16 and record['degree']==5
        assert record['resampled_each_round'] and record['connected']
        assert len(record['edges'])==512*5//2
    invalid = replace(benchmark, runs=[replace(run, options=run.options | dict(r=9)) for run in benchmark.runs])
    with pytest.raises(ValueError, match='log2.real nodes'):
        graph_plan(invalid)


def complete(benchmark, accuracy):
    suite = benchmark.manifest() | dict(status='completed')
    for row in suite['runs']:
        row.update(status='completed',wall_seconds=1,curve=[dict(time=t,accuracy=accuracy if t else 10.,
            loss=1.,worst_decile_accuracy=None,bytes_sent=t*100) for t in benchmark.times])
    return suite


@pytest.fixture
def comparison(tmp_path):
    registry = json.loads((ROOT/'experiments/leaderboard-algorithms.json').read_text())
    registry['algorithms'] = registry['algorithms'][:2]
    for index, entry in enumerate(registry['algorithms']):
        document = json.loads((ROOT/'experiments'/entry['config']).read_text())
        document['output'] = f'results-{index}'
        document['settings'].update(dataset='synthetic',model='mlp',dataset_base_path=None,peers=4,device='cpu')
        document['benchmark'].update(budgets=[1,2],seeds=[10,11,12],evaluation_period=1,execution_devices=['cpu'])
        document['tuning'].update(budget=2,seeds=[0,1,2],evaluation_period=1,n_trials=4,n_startup_trials=2,batch_size=2)
        options = document['runs'][0]['settings']
        options.update(k=16,r=2) if options['algorithm']=='shatter' else options.update(k=2)
        entry['config'] = f'algorithm-{index}.json'
        (tmp_path/entry['config']).write_text(json.dumps(document))
    (tmp_path/'algorithms.json').write_text(json.dumps(registry))
    catalog = json.loads((ROOT/'experiments/leaderboard-settings.json').read_text())
    catalog['algorithms'] = 'algorithms.json'
    for setting in catalog['settings']:
        suffix = 'evaluation' if setting['id']=='dirichlet01-uniform' else 'evaluation-'+setting['id']
        setting['output'] = 'results-0/'+suffix
    path = tmp_path/'catalog.json'
    path.write_text(json.dumps(catalog))
    registry = load_algorithms(path)
    baseline = registry['algorithms'][0]['_benchmark']
    final = replace(baseline,output=baseline.output/'evaluation',tuning=None)
    final.output.mkdir(parents=True)
    save_results(final.output,complete(final,60.))
    return path, registry


def test_comparison_keeps_measured_baseline_and_adds_pending_shatter_without_writing(comparison):
    _, registry = comparison
    baseline = registry['algorithms'][0]['_benchmark'].output/'evaluation'
    before = {path.name:path.read_bytes() for path in baseline.iterdir()}
    document = combined_leaderboard(registry,'dirichlet01-uniform')
    assert len(document['runs'])==6
    rows = document['tables'][-1]['rows']
    assert rows[0]['variant']=='dpsgd-kreg-log2' and rows[0]['accuracy']['mean']==60.
    assert rows[1]['variant']=='shatter-k16-rlog2' and rows[1]['accuracy'] is None
    assert all(row['rank'] is None for row in rows) and document['tables'][-1]['provisional']
    assert {path.name:path.read_bytes() for path in baseline.iterdir()}==before
    assert not registry['algorithms'][1]['_benchmark'].output.exists()


def test_complete_comparison_and_http_routes_include_both_algorithms(comparison):
    catalog, registry = comparison
    entry = registry['algorithms'][1]
    benchmark = entry['_benchmark']
    final = replace(benchmark,output=benchmark.output/'evaluation',tuning=None)
    final.output.mkdir(parents=True)
    save_results(final.output,complete(final,70.))
    document = combined_leaderboard(registry,'dirichlet01-uniform')
    assert document['status']=='completed'
    assert [row['variant'] for row in document['tables'][-1]['rows']]==['shatter-k16-rlog2','dpsgd-kreg-log2']
    assert [row['rank'] for row in document['tables'][-1]['rows']]==[1,2]
    details = algorithm_document(entry,registry,'iid-uniform')
    params = {row['parameter']:row['value'] for row in details['experiment_parameters']}
    assert params['Virtual nodes per real node']==16 and params['Total virtual nodes']==64
    assert 'degree r' in details['experiment_parameters'][1]['parameter']
    assert details['tuning']['partitioner']=='dirichlet'
    with create_server(catalog=catalog,port=0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/leaderboard.json') as response:
                result=json.load(response)
            assert len(result['runs'])==6 and len(result['algorithms'])==2
            with urlopen(base+'/leaderboard.csv') as response:
                rows=list(csv.DictReader(io.StringIO(response.read().decode())))
            assert {row['variant'] for row in rows}=={'dpsgd-kreg-log2','shatter-k16-rlog2'}
            assert all(row['accuracy_mean'] for row in rows)
            with urlopen(base+'/algorithm/shatter-k16-rlog2') as response:
                assert response.status==200
            with urlopen(base+'/algorithm.json?algorithm=shatter-k16-rlog2') as response:
                assert json.load(response)['tuning']['total_trials']==12
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize('mutation', ['seeds','network','partitioner'])
def test_incompatible_saved_scenarios_cannot_be_combined(comparison, mutation):
    _, registry = comparison
    path=registry['algorithms'][0]['_benchmark'].output/'evaluation/manifest.json'
    data=json.loads(path.read_text())
    if mutation=='seeds':data['seeds']=[99]
    elif mutation=='network':data['network']['profile']='heterogeneous'
    else:data['runs'][0]['options']['partitioner']='iid'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='different'):
        combined_leaderboard(registry,'dirichlet01-uniform')
