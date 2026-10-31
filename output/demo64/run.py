"""Server coordinator: two validation workers, then two evaluation profiles."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd

root=Path('/mnt/data/mdevos/conflux/results/demo64')
code=Path('/mnt/data/mdevos/conflux/demo-code')
source='948e33c642a2dea2732f0c9dbcd84c48235e0efe5018944aedb88799258638d8'
python=sys.executable
spec=importlib.util.spec_from_file_location('benchmark',code/'scripts/benchmark_demo.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
h=hashlib.sha256()
for path in sorted(list((code/'dasklearn').rglob('*.py'))+list((code/'scripts').glob('*.py'))):
    h.update(str(path.relative_to(code)).encode());h.update(path.read_bytes())
assert h.hexdigest()==source
if (root/'plan.json').exists():raise RuntimeError('Existing plan; do not launch duplicates')
# Old V2 execution files must match except for the appended new task functions
# and baseline registration. Fresh full test replays additionally verify this.
old=Path('/mnt/data/mdevos/conflux/v2-uploads-code');unchanged=[]
for path in sorted((old/'dasklearn').rglob('*.py')):
    relative=path.relative_to(old)
    if str(relative) in ['dasklearn/functions.py','dasklearn/simulation/peer_baselines.py']:continue
    assert path.read_bytes()==(code/relative).read_bytes(),relative
    unchanged.append(str(relative))
assert (code/'dasklearn/functions.py').read_text().startswith((old/'dasklearn/functions.py').read_text())
plan={'status':'validation','created_utc':datetime.now(timezone.utc).isoformat(),'host':'sacs006',
    'code_directory':str(code),'execution_commit':'73cba38','source_sha256':source,'python':python,
    'profiles':['uniform','heterogeneous'],'partitioners':['iid','dirichlet'],'seeds':[0,1,2],
    'local_steps':[5,1],'new_test_runs':24,'validation_runs':16,'v2_replays':4,'peers':64,
    'duration_simulated_s':900,'test_period_simulated_s':150,'dataset':'cifar10','model':'gnlenet',
    'batch_size':32,'alpha':.1,'payload_bytes':346251,'sparse_packet_bytes':20472,
    'mean_bandwidth_bytes_s':100000,'nominal_step_ms':20,'latency_ms':20,
    'compression_decay':.999,'compression_chunk':64,'compression_topk':32,
    'cached_comparators':'/mnt/data/mdevos/conflux/results/v2-uploads64',
    'validation_rule':'Independently for H=5 and H=1 maximize mean uniform IID/Dirichlet validation accuracy, seed 0, 900s; lower LR breaks ties.',
    'lr_candidates':[.0003,.001,.003,.01],
    'cache_validation':{'unchanged_files':unchanged,'functions':'Previous contents preserved, only DeMo functions appended.','replays':'All 384 per-node metrics/statistics/network points for seed-0 IID and Dirichlet in both profiles must match cached V2 exactly.'},
    'tests':{'macos_cpu':'111 passed, 6 skipped','linux_cuda':'see tests-linux.txt','official_reference':'3 workers, 7 steps, parameter error 0; residual error <=4.19e-9'},
    'limitations':['CNN matrix-unfolding adapter; not paper language-model reproduction.',
        'Ring all-gather, stable membership, no churn recovery.',
        'H=5 is gradient accumulation at fixed weights; H=1 is canonical.',
        'Reference sparse collision averaging and Signum optimizer; different update rule from V2 SGD.',
        'Compression/aggregation take zero simulated time as with other baselines; physical execution reported separately.',
        'Fixed reference compression settings and limited validation-only LR search; no guarantee of optimal tuning.'],
    'jobs':[]}

def save(): (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
def launch(command,gpu,label):
    log_path=root/f'{label}.log'
    with log_path.open('wb') as log:
        job=subprocess.Popen(command,cwd=code,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    plan['jobs'].append({'label':label,'gpu':gpu,'pid':job.pid,'log':str(log_path),'command':command});save()
    return job
save()
# Each validation worker owns one H and one GPU. No shared output mutation.
workers=[]
for gpu,steps in enumerate([5,1]):
    command=[python,str(root/'tune_worker.py'),'--steps',str(steps)]
    workers.append(launch(command,gpu,f'validation-h{steps}'))
for job in workers:
    if job.wait()!=0:raise RuntimeError('Validation failed; inspect logs')
scores=pd.concat([pd.read_csv(root/f'validation-h{steps}.csv') for steps in [5,1]],ignore_index=True)
scores.to_csv(root/'validation-scores.csv',index=False)
selected={str(steps):float(g.sort_values(['mean_validation_accuracy','learning_rate'],ascending=[False,True]).iloc[0].learning_rate) for steps,g in scores.groupby('local_steps')}
(root/'selection.json').write_text(json.dumps({'selected':selected,'rule':plan['validation_rule'],'candidates':plan['lr_candidates'],'split':'5000 validation images, test excluded'},indent=2)+'\n')
print(json.dumps({'validation_selection':selected,'scores':scores.to_dict('records')}),flush=True)
plan['status']='test';save();workers=[]
for gpu,profile in enumerate(plan['profiles']):
    command=[python,str(code/'scripts/benchmark_demo.py'),'--output',str(root),'--dataset-base-path','/mnt/data/mdevos/conflux/datasets','--mode','evaluate','--profile',profile,'--device','cuda:0']
    workers.append(launch(command,gpu,profile))
for job in workers:
    if job.wait()!=0:raise RuntimeError('Evaluation failed; inspect logs')
plan['status']='ready-for-audit';save()
print('All runs finished; ready for audit',flush=True)
