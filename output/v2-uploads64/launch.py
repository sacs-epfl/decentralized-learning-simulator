"""Launch the predeclared factorial comparison on the authorized GPU server."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/mnt/data/mdevos/conflux/results/v2-uploads64')
CODE = Path('/mnt/data/mdevos/conflux/v2-uploads-code')
SOURCE = 'c0281ddd84a9367dffd81239d3aa235f863aaaae1b0aa7ba793d382214a08305'


if __name__ == '__main__':
    h = hashlib.sha256()
    for path in sorted(list((CODE/'dasklearn').rglob('*.py'))+list((CODE/'scripts').glob('*.py'))):
        h.update(str(path.relative_to(CODE)).encode());h.update(path.read_bytes())
    assert h.hexdigest() == SOURCE
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'plan.json').exists():
        raise RuntimeError('Plan already exists; inspect existing jobs instead of starting duplicates')
    plan = {'status':'running', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'host':'sacs006', 'execution_commit':'e11ac439cd7076f1557c909f8929806bfa2b9b43',
        'source_sha256':SOURCE, 'code_directory':str(CODE), 'python':sys.executable,
        'cases':['reference','immediate','wide','immediate-wide','moshpit'],
        'profiles':['uniform','heterogeneous'], 'partitioners':['iid','dirichlet'], 'seeds':[0,1,2],
        'runs':60, 'peers':64, 'sample_size':16, 'reference_chunks_and_relays':8,
        'wide_chunks_and_relays':16, 'success_fraction':1,
        'duration_simulated_s':900, 'test_period_simulated_s':150,
        'learning':{'dataset':'cifar10','model':'gnlenet','local_steps':5,'batch_size':32,
                    'learning_rate':.01,'momentum':0,'weight_decay':0,'alpha':.1},
        'mean_bandwidth_bytes_s':100000, 'nominal_step_ms':20, 'latency_ms':20,
        'payload_bytes':346251, 'evaluation_split':'test', 'evaluation_examples':10000,
        'selection':'Report all fixed configurations; no test-set winner selection.',
        'tests':{'macos_cpu':'85 passed, 6 CUDA tests skipped','linux_cuda':'91 passed'},
        'jobs':[]}
    for gpu, profile in enumerate(plan['profiles']):
        command = [sys.executable,str(CODE/'scripts/benchmark_conflux_v2_uploads.py'),
            '--output',str(ROOT),'--dataset-base-path','/mnt/data/mdevos/conflux/datasets',
            '--profile',profile,'--device','cuda:0']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log_path = ROOT/f'{profile}.log'
        with log_path.open('wb') as log:
            process = subprocess.Popen(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,
                                       start_new_session=True)
        plan['jobs'].append({'profile':profile,'gpu':gpu,'pid':process.pid,
                             'log':str(log_path),'command':command})
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2))
