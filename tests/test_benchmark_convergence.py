"""Exercise tuning, heldout evaluation, reporting, resume, and provenance guard."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_small_convergence_study_executes_and_resumes(tmp_path):
    study=dict(version=1,name='smoke',device='cpu',peers=4,sample_size=2,chunks=2,
        algorithms=['conflux-v2','dpsgd'],profiles=['uniform'],partitioners=['iid'],
        seeds=[0],tuning_seed=10001,tuning_profile='heterogeneous',alpha=.1,local_steps=1,batch_size=16,
        workloads=[dict(name='digits',dataset='digits',model='mlp',duration=2,test_period=1,
            bandwidth_bytes_per_s=100000,compute_step_ms=100,evaluation_examples=0,
            learning_rates=[.03,.1],targets=[5,100])])
    config=tmp_path/'study.json';config.write_text(json.dumps(study))
    output=tmp_path/'output'
    cmd=[sys.executable,str(ROOT/'scripts/benchmark_convergence.py'),'--study',str(config),
         '--output',str(output),'--data',str(tmp_path),'--gpus','0']
    env=dict(os.environ,HF_HOME=str(tmp_path/'hf'),MPLCONFIGDIR=str(tmp_path/'mpl'),OMP_NUM_THREADS='1')
    result=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    assert (output/'COMPLETE').exists()
    records=[json.loads(p.read_text()) for p in output.glob('runs/*/record.json')]
    assert len(records)==6 and all(r['status']=='completed' for r in records)
    confirm=[r for r in records if r['stage']=='confirm']
    assert len(confirm)==2 and all('test_accuracy' in r['metrics'] for r in confirm)
    assert len(json.loads((output/'selection.json').read_text()))==2
    tta=(output/'time-to-accuracy.csv').read_text()
    assert 'False' in tta  # Impossible 100% target is censored, not fabricated.
    before={p:p.stat().st_mtime_ns for p in output.glob('runs/*/record.json')}
    resumed=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=30)
    assert resumed.returncode==0,resumed.stdout+resumed.stderr
    assert before=={p:p.stat().st_mtime_ns for p in before}
    import fcntl
    with (output/'driver.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        concurrent=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=30)
        assert concurrent.returncode!=0 and 'Study is already running' in concurrent.stderr
    study['alpha']=.2;config.write_text(json.dumps(study))
    changed=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=30)
    assert changed.returncode!=0 and 'Refusing to change source/study' in changed.stderr
