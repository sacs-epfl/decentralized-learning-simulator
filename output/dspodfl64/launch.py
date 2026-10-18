import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

root=Path('/mnt/data/mdevos/conflux/results/dspodfl64')
code=Path('/mnt/data/mdevos/conflux/dspodfl-code')
old_code=Path('/mnt/data/mdevos/conflux/v2-uploads-code')
source='f3d42e5a990a7014745dc4355aad8925bb77fe7cd6959883441b4ab903d3d33a'
h=hashlib.sha256()
for path in sorted(list((code/'dasklearn').rglob('*.py'))+list((code/'scripts').glob('*.py'))):
    h.update(str(path.relative_to(code)).encode());h.update(path.read_bytes())
assert h.hexdigest()==source
unchanged=[]
for previous in sorted((old_code/'dasklearn').rglob('*.py')):
    relative=previous.relative_to(old_code)
    if str(relative)=='dasklearn/simulation/peer_baselines.py':continue
    assert previous.read_bytes()==(code/relative).read_bytes(),str(relative)
    unchanged.append(str(relative))
if (root/'plan.json').exists():raise RuntimeError('Existing plan; do not launch duplicate experiments')
plan={'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),'host':'sacs006',
    'execution_commit':'a71720fc7d31c054040bdb9a0178727b09f41ed7','source_sha256':source,
    'code_directory':str(code),'python':sys.executable,'dataset':'cifar10','model':'gnlenet',
    'peers':64,'profiles':['uniform','heterogeneous'],'partitioners':['iid','dirichlet'],'seeds':[0,1,2],
    'local_steps':[5,1],'runs':24,'duration_simulated_s':900,'test_period_simulated_s':150,
    'learning_rate':.01,'batch_size':32,'momentum':0,'weight_decay':0,'alpha':.1,
    'payload_bytes':346251,'mean_bandwidth_bytes_s':100000,'nominal_step_ms':20,'latency_ms':20,
    'evaluation_examples':10000,'evaluation_split':'test','degree':4,'probability_mode':'capacity',
    'sgd_probability':.5,'communication_probability':.5,'minimum_round_ms':20,
    'cached_comparators':'/mnt/data/mdevos/conflux/results/v2-uploads64',
    'cache_validation':{'unchanged_execution_files':unchanged,'exception':'peer_baselines.py only adds the new DSpodFL registration; V2 protocol and learning/execution code are unchanged.'},
    'tests':{'macos_cpu':'98 passed, 6 skipped','linux_cuda':'104 passed'},
    'readout':'All physical nodes, stored completed checkpoints; no free global averaging. V2 latest aggregate is separate.',
    'limitations':['One-step equation and five-step block-delta extension reported separately.',
        'Fixed capacity-derived policy and degree-4 graph; not individually tuned reproduction of paper VGG11/RGG experiments.',
        'Stable membership; sporadic work does not implement offline-node recovery.'],
    'jobs':[]}
for gpu,profile in enumerate(plan['profiles']):
    command=[sys.executable,str(code/'scripts/benchmark_dspodfl.py'),'--output',str(root),
        '--dataset-base-path','/mnt/data/mdevos/conflux/datasets','--profile',profile,'--device','cuda:0']
    log_path=root/f'{profile}.log'
    with log_path.open('wb') as log:
        job=subprocess.Popen(command,cwd=code,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    plan['jobs'].append({'profile':profile,'gpu':gpu,'pid':job.pid,'command':command,'log':str(log_path)})
(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'jobs':plan['jobs'],'unchanged_runtime_files':len(unchanged)},indent=2))
