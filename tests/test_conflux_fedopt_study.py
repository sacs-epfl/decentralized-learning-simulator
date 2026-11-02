import json
from pathlib import Path
from scripts.benchmark_conflux_fedopt import tuning_jobs,confirmation_jobs,selection

ROOT=Path(__file__).resolve().parents[1]


def test_fedopt_study_has_equal_validation_budgets_and_paired_confirmations():
    jobs=tuning_jobs(ROOT/'experiments/conflux-v2/fedopt/tuning.json')
    assert len(jobs)==36
    selected={}
    for family in ['server-sgd','server-momentum','server-adam']:
        group=[j for j in jobs if j['family']==family]
        assert len(group)==12
        assert all(j['settings']['evaluation_split']=='validation' and not j['settings']['final_heldout'] for j in group)
        selected[family]=dict(settings=group[0]['settings'])
    confirm=confirmation_jobs(selected)
    assert len(confirm)==27
    for seed in [0,1,2]:
        paired=[j for j in confirm if j['seed']==seed]
        assert len(paired)==9 and all(j['settings']['final_heldout'] for j in paired)
        for key in ['peers','sample_size','chunks_in_sample','relay_count','partitioner','capacity_profile',
                    'bandwidth_bytes_per_s','compute_step_ms','network_latency_ms','model_size_bytes','local_steps']:
            assert len({j['settings'][key] for j in paired})==1,key
        adam=[j for j in paired if j['family']=='server-adam']
        assert {(j['settings']['reducer_epoch_rounds'],j['settings']['state_bits']) for j in adam}=={(0,32),(1,32),(1,8),(8,8)}
        assert len({(j['settings']['learning_rate'],j['settings']['server_learning_rate'],j['settings']['server_tau']) for j in adam})==1


def test_rate_selection_uses_validation_not_test_score(tmp_path):
    for family in ['server-sgd','server-momentum','server-adam']:
        for index,auc,test in [(0,10,99),(1,20,1)]:
            name=f'tune-{family}-{index}';directory=tmp_path/'runs'/name;directory.mkdir(parents=True)
            record=dict(name=name,stage='tune',family=family,status='completed',settings=dict(server_learning_rate=.1+index),
                        metrics=dict(validation_auc=auc,test_accuracy=test))
            (directory/'record.json').write_text(json.dumps(record))
    chosen=selection(tmp_path)
    assert all(v['run'].endswith('-1') for v in chosen.values())
