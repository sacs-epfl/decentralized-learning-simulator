from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import threading
from urllib.request import urlopen

import pytest
import torch

from dasklearn.centralized_baseline import load_config, reusable_tuning, select_rate, train_worker
from dasklearn.leaderboard_reference import centralized_reference, reference_spec
from dasklearn.leaderboard_server import create_server

ROOT = Path(__file__).resolve().parents[1]


def config(tmp_path):
    path = tmp_path / 'central.json'
    path.write_text(json.dumps(dict(version=1, name='central-test', output='results',
        settings=dict(dataset='synthetic', model='mlp', device='cpu', torch_threads=1, batch_size=8),
        training=dict(steps=6, evaluation_period=2),
        tuning=dict(steps=2, seeds=[0], learning_rates=[.01,.03], evaluation_examples=100),
        evaluation=dict(seeds=[10]))))
    return load_config(path)


def test_real_plan_is_step_budgeted_and_disjoint_without_simulation():
    plan = load_config(ROOT / 'experiments/leaderboard-cifar10-centralized.yaml')
    assert plan['training'] == dict(steps=50000, evaluation_period=1000)
    assert plan['tuning']['steps'] == 5000 and len(plan['tuning']['learning_rates']) == 5
    assert plan['seeds'] == [10,11,12] and plan['tuning']['seeds'] == [0]
    assert plan['settings']['participants'] == 1 and plan['settings']['partitioner'] == 'iid'
    assert plan['time_basis'] == 'optimizer_steps'


@pytest.mark.parametrize('mutation', ['overlap', 'steps', 'unknown', 'lr', 'momentum'])
def test_invalid_config_rejected_before_training(tmp_path, mutation):
    plan = config(tmp_path)
    path = Path(plan['source']); document = json.loads(path.read_text())
    if mutation == 'overlap': document['evaluation']['seeds']=[0]
    elif mutation == 'steps': document['training']['steps']=True
    elif mutation == 'unknown': document['settings']['partitioner']='dirichlet'
    elif mutation == 'lr': document['tuning']['learning_rates']=[.01,.01]
    else: document['settings']['momentum']=.9
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError): load_config(path)
    assert not Path(plan['output']).exists()


@pytest.mark.parametrize('scores', [
    [(10,2),(70,.4),(90,.5),(80,.6)],
    [(10,2),(70,.4),(90,.4),(80,.6)],
])
def test_minimum_validation_loss_model_is_tested_once_and_earlier_step_breaks_loss_ties(tmp_path, monkeypatch, scores):
    from dasklearn.model_evaluator import ModelEvaluator
    plan = config(tmp_path)
    path = tmp_path / 'worker'; path.mkdir()
    settings = deepcopy(plan['settings']); settings.update(seed=10,work_dir=str(path))
    settings['learning']['learning_rate']=.03
    calls, validation_states, test_states = [], [], []
    scores = iter(scores)
    def evaluate(self, model, *args, **kwargs):
        calls.append(self.settings.evaluation_split)
        if self.settings.evaluation_split == 'validation':
            validation_states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
            return next(scores)
        test_states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        return 85,.7
    monkeypatch.setattr(ModelEvaluator,'evaluate_accuracy',evaluate)
    result = train_worker(dict(settings=settings,output=str(path),phase='evaluation',steps=6,
                               evaluation_period=2,validation_examples=0))
    assert result['completed_steps']==6 and result['samples_trained']==48
    assert result['best_validation']['step']==2 and result['test']['checkpoint_step']==2
    assert calls == ['validation']*4+['test']
    assert all(torch.equal(test_states[0][k],value) for k,value in validation_states[1].items())
    assert any(not torch.equal(validation_states[-1][k],v) for k,v in test_states[0].items())
    assert (path/'best-model.pt').exists() and (path/'validation-curve.csv').exists()
    checkpoint = torch.load(path/'best-model.pt', map_location='cpu', weights_only=True)
    assert checkpoint['validation']['step'] == 2
    assert all(torch.equal(checkpoint['state_dict'][k], value) for k,value in test_states[0].items())


def test_tuning_reads_validation_only_and_allocates_exact_update_count(tmp_path, monkeypatch):
    from dasklearn.model_evaluator import ModelEvaluator
    plan=config(tmp_path); path=tmp_path/'worker';path.mkdir()
    settings=deepcopy(plan['settings']);settings.update(seed=0,work_dir=str(path))
    calls=[]
    def evaluate(self,*args,**kwargs):
        calls.append(self.settings.evaluation_split);return 50,1
    monkeypatch.setattr(ModelEvaluator,'evaluate_accuracy',evaluate)
    result=train_worker(dict(settings=settings,output=str(path),phase='tuning',steps=2,
                             evaluation_period=2,validation_examples=100))
    assert result['completed_steps']==2 and result['test'] is None
    assert calls==['validation','validation'] and not (path/'best-model.pt').exists()


def test_lr_selection_uses_endpoint_not_best_checkpoint_and_rejects_incomplete_grid(tmp_path):
    spec=config(tmp_path)['tuning']
    records=[dict(rate_index=0,seed=0,status='completed',completed_steps=2,
                  curve=[dict(step=0,accuracy=99),dict(step=2,accuracy=50)]),
             dict(rate_index=1,seed=0,status='completed',completed_steps=2,
                  curve=[dict(step=0,accuracy=10),dict(step=2,accuracy=60)])]
    assert select_rate(records,spec)['learning_rate']==.03
    records[1]['curve'][-1]['accuracy']=50
    assert select_rate(records,spec)['learning_rate']==.01
    with pytest.raises(ValueError): select_rate(records[:1],spec)


def reference_fixture(tmp_path):
    plan=config(tmp_path)
    settings=plan['settings']
    raw=dict(kind='centralized-reference',time_basis='optimizer_steps',settings=settings,
        status='completed',phase='finished',steps=6,seeds=[10],learning_rate=.03,
        accuracy=dict(mean=80,sd=None),loss=dict(mean=.5,sd=None),
        runs=[dict(name='centralized-s10',seed=10,status='completed',completed_steps=6,total_steps=6,
                   test=dict(accuracy=80,loss=.5,checkpoint_step=4))])
    output=tmp_path/'reference';output.mkdir()
    (output/'reference.json').write_text(json.dumps(raw))
    options={k:v for k,v in settings.items() if k!='learning'} | settings['learning']
    target=dict(name='target',evaluation_split='test',evaluation_model='global',scenario=dict(peers=32),
                runs=[dict(seed=10,options=options)])
    return raw,output,target


def test_reference_requires_full_seed_set_and_matching_data_and_excludes_validation(tmp_path):
    raw,output,target=reference_fixture(tmp_path);spec=dict(output=output)
    result=centralized_reference(target,spec)
    assert result['status']=='completed' and result['accuracy']['mean']==80
    assert target.get('reference') is None
    for mutation in ('partial','nonfinite','data','seed','validation'):
        candidate=deepcopy(raw);view=deepcopy(target)
        if mutation=='partial':candidate['runs'][0]['status']='running'
        elif mutation=='nonfinite':candidate['loss']['mean']=float('inf')
        elif mutation=='data':view['runs'][0]['options']['model']='different'
        elif mutation=='seed':view['runs'][0]['seed']=11
        else:view['evaluation_split']='validation'
        (output/'reference.json').write_text(json.dumps(candidate))
        result=centralized_reference(view,spec)
        if result is not None:
            assert result['status']!='completed' and result['accuracy'] is None and result['loss'] is None


def test_reference_served_live_without_rewriting_experiment_or_csv(tmp_path):
    raw,output,target=reference_fixture(tmp_path)
    suite=tmp_path/'suite';suite.mkdir()
    encoded=json.dumps(target)
    (suite/'leaderboard.json').write_text(encoded)
    (suite/'leaderboard.csv').write_text('original\n')
    catalog=tmp_path/'catalog.json'
    catalog.write_text(json.dumps(dict(version=1,default='iid',settings=[dict(id='iid',label='IID',output='suite')],
                                      reference=dict(output='reference'))))
    assert reference_spec(catalog)['output']==output.resolve()
    with create_server(catalog=catalog,port=0) as server:
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            base=f'http://127.0.0.1:{server.server_port}'
            with urlopen(base+'/leaderboard.json') as response:
                assert json.load(response)['reference']['accuracy']['mean']==80
            raw.update(status='running',phase='evaluation')
            (output/'reference.json').write_text(json.dumps(raw))
            with urlopen(base+'/leaderboard.json') as response:
                result=json.load(response)['reference']
                assert result['status']=='running' and result['accuracy'] is None
            with urlopen(base+'/leaderboard.csv') as response:
                assert response.read()==b'original\n'
        finally:server.shutdown();thread.join()
    assert (suite/'leaderboard.json').read_text()==encoded


def test_single_cifar_partition_contains_all_training_examples():
    from datasets import Dataset
    from dasklearn.datasets.huggingface import allocate_indices
    from dasklearn.session_settings import SessionSettings
    settings=SessionSettings.from_dict(load_config(ROOT/'experiments/leaderboard-cifar10-centralized.yaml')['settings'])
    parts=allocate_indices(Dataset.from_dict(dict(label=[i%10 for i in range(45000)])),settings,min_size=8)
    assert len(parts)==1 and sorted(parts[0])==list(range(45000))


def previous_tuning(tmp_path, plan):
    previous=tmp_path/'previous';previous.mkdir()
    (previous/'resolved.json').write_text(json.dumps(plan))
    records=[]
    for index,rate in enumerate(plan['tuning']['learning_rates']):
        name=f'lr{index+1:02d}-s0';directory=previous/'tuning'/name;directory.mkdir(parents=True)
        settings=deepcopy(plan['settings']);settings.update(seed=0,torch_device_name='cpu',work_dir=str(directory))
        settings['learning']['learning_rate']=rate
        metrics=dict(status='completed',seed=0,learning_rate=rate,completed_steps=2,test=None,
                     curve=[dict(step=0,accuracy=10,loss=2),dict(step=2,accuracy=50+index,loss=1)])
        job=dict(settings=settings,phase='tuning',steps=2,evaluation_period=2,
                 validation_examples=plan['tuning']['evaluation_examples'])
        (directory/'resolved.json').write_text(json.dumps(job))
        (directory/'metrics.json').write_text(json.dumps(metrics))
        records.append(metrics|dict(name=name,rate_index=index))
    (previous/'tuning/manifest.json').write_text(json.dumps(dict(runs=records)))
    (previous/'selection.json').write_text(json.dumps(select_rate(records,plan['tuning'])))
    return previous


def test_reused_tuning_validates_raw_curves_and_keeps_original_files(tmp_path):
    plan=config(tmp_path);previous=previous_tuning(tmp_path,plan)
    manifest=previous/'tuning/manifest.json';original=manifest.read_bytes()
    records,selection=reusable_tuning(plan,previous)
    assert len(records)==2 and selection['learning_rate']==.03
    assert all(row.get('reused_from') for row in records) and manifest.read_bytes()==original
    metrics=previous/'tuning/lr01-s0/metrics.json';raw=json.loads(metrics.read_text())
    raw['curve'][-1]['accuracy']=99;metrics.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='raw tuning trial differs'):reusable_tuning(plan,previous)
    assert not Path(plan['output']).exists()


def test_reused_tuning_executes_only_final_seeds(tmp_path,monkeypatch):
    import dasklearn.centralized_baseline as module
    plan=config(tmp_path);previous=previous_tuning(tmp_path,plan);jobs=[]
    class Process:
        def __init__(self,command,**kwargs):
            job=json.loads(Path(command[-1]).read_text());jobs.append(job)
            assert job['phase']=='evaluation'
            train_worker(job)
            self.pid=123
        def poll(self):return 0
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    monkeypatch.setattr(module,'provenance',lambda root:{})
    assert module.execute(plan,devices=['cpu'],reuse_tuning=previous)==0
    assert len(jobs)==1 and jobs[0]['settings']['learning']['learning_rate']==.03
    state=json.loads((Path(plan['output'])/'reference.json').read_text())
    assert state['status']=='completed' and state['reused_tuning_trials']==2
    assert state['checkpoint_rule']=='Lowest validation loss; earlier step breaks ties'
    run=state['runs'][0]
    expected=min(run['curve'],key=lambda point:(point['loss'],point['step']))
    assert run['test']['checkpoint_step']==expected['step']
