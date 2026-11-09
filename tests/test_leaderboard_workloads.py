"""Dataset/model selectors isolate routes, scores, CSV and algorithm details."""
import json
from dataclasses import replace
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from dasklearn.leaderboard_details import load_algorithms
from dasklearn.benchmark import save_results
from dasklearn.leaderboard_server import create_server, load_workloads
from scripts.run_methodology import graph_plan

ROOT = Path(__file__).resolve().parents[1]


def test_cifar100_campaign_keeps_search_protocol_and_changes_workload_capacities():
    registry = load_algorithms(ROOT / 'experiments/leaderboard-cifar100-settings.json')
    benchmark = registry['algorithms'][0]['_benchmark']
    assert benchmark.runs[0].options['dataset'] == 'cifar100'
    assert benchmark.runs[0].options['model'] == 'resnet18_gn'
    assert benchmark.network == dict(profile='uniform', bandwidth_bytes_per_second=12500000, compute_step_ms=6)
    assert benchmark.tuning['n_trials'] == 40 and benchmark.tuning['budget'] == 3600
    assert benchmark.budgets == [3600, 7200, 10800, 14400, 18000]
    assert registry['algorithms'][0]['evaluation'] == dict(policy='best_accuracy_within_budget', primary_time=18000)
    assert benchmark.tuning['evaluation_period'] == 300
    assert benchmark.tuning['seeds'] == [100, 101, 102] and benchmark.seeds == [110, 111, 112]
    assert benchmark.execution_devices == ['cuda:0', 'cuda:1'] * 3
    assert benchmark.tuning['batch_size'] == 2
    assert len(graph_plan(benchmark)) == 6
    default, workloads = load_workloads(ROOT / 'experiments/leaderboard-workloads.json')
    assert default == 'cifar10-gnlenet' and len(workloads) == 2
    assert len(workloads['cifar10-gnlenet']['algorithms']['algorithms']) == 6
    assert len(workloads['cifar100-resnet18-gn']['algorithms']['algorithms']) == 1


@pytest.fixture
def served_workloads(tmp_path):
    workload_rows = []
    for index, source in enumerate(['leaderboard-settings.json', 'leaderboard-cifar100-settings.json']):
        catalog = json.loads((ROOT / 'experiments' / source).read_text())
        registry = json.loads((ROOT / 'experiments' / catalog['algorithms']).read_text())
        for entry in registry['algorithms']:
            config = json.loads((ROOT / 'experiments' / entry['config']).read_text())
            config['output'] = str(tmp_path / f'results-{index}' / entry['id'])
            config['settings']['dataset_base_path'] = str(ROOT / 'data' / config['settings']['dataset'])
            entry['config'] = str(tmp_path / f'config-{index}-{entry["id"]}.json')
            Path(entry['config']).write_text(json.dumps(config))
        catalog['algorithms'] = f'algorithms-{index}.json'
        (tmp_path / catalog['algorithms']).write_text(json.dumps(registry))
        catalog_path = tmp_path / f'catalog-{index}.json'
        catalog_path.write_text(json.dumps(catalog))
        if index == 0:
            # A measured score in one workload must never appear in the other,
            # even though both register the same D-PSGD algorithm identifier.
            benchmark = load_algorithms(catalog_path)['algorithms'][0]['_benchmark']
            final = replace(benchmark, output=benchmark.output / 'evaluation', tuning=None)
            final.output.mkdir(parents=True)
            suite = final.manifest() | dict(status='completed')
            for run in suite['runs']:
                run.update(status='completed', wall_seconds=1, curve=[dict(time=time,
                    accuracy=80. if time else 1., loss=1., bytes_sent=time * 100,
                    worst_decile_accuracy=None) for time in final.times])
            save_results(final.output, suite)
        workload_rows.append(dict(id=f'workload-{index}', label=f'Dataset/model {index}', catalog=catalog_path.name))
    outer = json.loads((tmp_path / 'catalog-0.json').read_text())
    outer.update(default_workload='workload-0', workloads=workload_rows)
    path = tmp_path / 'workloads.json'
    path.write_text(json.dumps(outer))
    with create_server(catalog=path, port=0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}'
        finally:
            server.shutdown()
            thread.join()


def test_workload_routes_do_not_mix_algorithms_or_data(served_workloads):
    for index, dataset, count in [(0, 'cifar10', 6), (1, 'cifar100', 1)]:
        suffix = f'?workload=workload-{index}'
        with urlopen(served_workloads + '/settings.json' + suffix) as response:
            settings = json.load(response)
        assert settings['workload'] == f'workload-{index}' and len(settings['workloads']) == 2
        assert len(settings['algorithms']) == count
        with urlopen(served_workloads + '/leaderboard.json' + suffix) as response:
            result = json.load(response)
        assert len(result['algorithms']) == count
        if index == 1:
            assert result['evaluation'] == dict(policy='best_accuracy_within_budget', primary_time=18000)
        assert all(run['options']['dataset'] == dataset for run in result['runs'])
        row = next(row for row in result['tables'][-1]['rows'] if row['variant'] == 'dpsgd-kreg-log2')
        if index == 0:
            assert row['accuracy']['mean'] == 80.
        else:
            assert row['accuracy'] is None
        with urlopen(served_workloads + '/leaderboard.csv' + suffix) as response:
            csv = response.read().decode()
        assert ('shatter-k16-rlog2' in csv) == (index == 0)
        with urlopen(served_workloads + '/algorithm.json' + suffix + '&algorithm=dpsgd-kreg-log2') as response:
            details = json.load(response)
        assert details['configuration']['settings']['dataset'] == dataset
        with urlopen(served_workloads + '/algorithm-config.json' + suffix + '&algorithm=dpsgd-kreg-log2') as response:
            assert json.load(response)['settings']['dataset'] == dataset


@pytest.mark.parametrize('query', ['workload=unknown', 'workload=../../etc',
    'workload=workload-0&workload=workload-1', 'workload=workload-1&algorithm=demo'])
def test_unknown_or_ambiguous_workload_and_algorithm_rejected(served_workloads, query):
    route = '/algorithm.json' if 'algorithm=' in query else '/leaderboard.json'
    with pytest.raises(HTTPError) as error:
        urlopen(served_workloads + route + '?' + query)
    assert error.value.code == 404
