"""Split independent remaining uniform runs without changing protocol code."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pandas as pd

ROOT=Path('/mnt/data/mdevos/conflux/results/peer-baselines64')
CODE=Path('/mnt/data/mdevos/conflux/peer-baselines-code')
LOGS=Path('/mnt/data/mdevos/conflux/logs')
ORIGINAL_PID=477068
ABLATION_PID=479679


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def run_cohort(gpu,seeds):
    if gpu==0:
        # Preserve every completed IID run. The original driver may have begun
        # its next independent run; only that uncommitted work is discarded.
        while True:
            frame=pd.read_csv(ROOT/'uniform/summary.csv')
            if len(frame[frame.partitioner.eq('iid')])==18:break
            if not alive(ORIGINAL_PID):raise RuntimeError('Original uniform job exited early')
            time.sleep(1)
        command=Path(f'/proc/{ORIGINAL_PID}/cmdline').read_bytes()
        assert b'/peer-baselines64/uniform\0' in command
        os.kill(ORIGINAL_PID,signal.SIGTERM)
        for _ in range(150):
            if not alive(ORIGINAL_PID):break
            time.sleep(.1)
        else:raise RuntimeError('Original uniform process did not stop')
        frame=pd.read_csv(ROOT/'uniform/summary.csv')
        frame[frame.partitioner.eq('iid')].to_csv(ROOT/'uniform-iid-completed-summary.csv',index=False)
        curves=pd.read_csv(ROOT/'uniform/curves.csv')
        curves[curves.partitioner.eq('iid')].to_csv(ROOT/'uniform-iid-completed-curves.csv',index=False)
    else:
        while not (ROOT/'moshpit-full/COMPLETE').exists():
            if not alive(ABLATION_PID):raise RuntimeError('Ablation exited without completing')
            time.sleep(2)
    name='uniform-dir-s'+''.join(map(str,seeds));out=ROOT/name
    assert not out.exists(),'Cohort output already exists'
    variants=['conflux-v2','shatter-r2','teleportation-ring','teleportation-hypercube','d-local-sgd-ring','moshpit']
    command=[sys.executable,str(CODE/'scripts/run_cifar.py'),'--dataset-base-path',
        '/mnt/data/mdevos/conflux/datasets','--output',str(out),'--variants',*variants,
        '--profiles','uniform','--partitioners','dirichlet','--seeds',*map(str,seeds),
        '--peers','64','--sample-size','16','--chunks','8','--group-size','8','--duration','900',
        '--test-period','150','--bandwidth','100000','--step-ms','20','--latency-ms','20',
        '--learning-rate','.01','--alpha','.1','--momentum','0','--model-bytes','346251']
    record={'gpu':gpu,'seeds':seeds,'partitioner':'dirichlet','command':command,'output':str(out)}
    (ROOT/(name+'-dispatch.json')).write_text(json.dumps(record,indent=2)+'\n')
    with (LOGS/(name+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=CODE,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}')
    assert (out/'COMPLETE').exists()
    return out


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run_cohort,0,[2]),pool.submit(run_cohort,1,[0,1])]
        cohorts=[job.result() for job in jobs]
    summary=[pd.read_csv(ROOT/'uniform-iid-completed-summary.csv')]
    curves=[pd.read_csv(ROOT/'uniform-iid-completed-curves.csv')]
    for cohort in cohorts:
        summary.append(pd.read_csv(cohort/'summary.csv'));curves.append(pd.read_csv(cohort/'curves.csv'))
        for folder in (cohort/'runs').iterdir():
            canonical=ROOT/'uniform/runs'/folder.name
            if canonical.exists():canonical.rename(canonical.with_name(canonical.name+'-partial-before-split'))
            canonical.symlink_to(folder,target_is_directory=True)
    summary=pd.concat(summary,ignore_index=True);curves=pd.concat(curves,ignore_index=True)
    assert len(summary)==36 and not summary.duplicated(['variant','profile','partitioner','seed']).any()
    summary.to_csv(ROOT/'uniform/summary.csv',index=False);curves.to_csv(ROOT/'uniform/curves.csv',index=False)
    manifest=json.loads((ROOT/'uniform/manifest.json').read_text())
    manifest['cuda_visible_devices']='0 and 1; see uniform-dir-s*-dispatch.json'
    (ROOT/'uniform/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'uniform/COMPLETE').write_text('All uniform IID and Dirichlet cohorts completed\n')
    (ROOT/'uniform-split-COMPLETE').write_text('Independent cohorts merged; protocol source and learning settings unchanged\n')
    print('Uniform split complete',flush=True)


if __name__=='__main__':main()
