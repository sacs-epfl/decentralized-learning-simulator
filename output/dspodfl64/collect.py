import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/mnt/data/mdevos/conflux/results/dspodfl64')
plan=json.loads((root/'plan.json').read_text());previous=None
while True:
    counts={}
    for job in plan['jobs']:
        p=job['profile']
        counts[p]=sum(max(0,len(f.read_text().splitlines())-1) for f in (root/p).glob('*/summary.csv'))
        if not (root/p/'COMPLETE').exists():
            try:os.kill(job['pid'],0)
            except ProcessLookupError:raise RuntimeError(f'{p} driver exited without COMPLETE; inspect {job["log"]}')
    if counts!=previous:print(json.dumps({'completed_runs':counts}),flush=True);previous=counts
    if all((root/p/'COMPLETE').exists() for p in plan['profiles']):break
    time.sleep(15)
subprocess.run([sys.executable,str(root/'report.py')],check=True)
(root/'COMPLETE').write_text('All 24 new runs and 36 cached comparison runs audited\n')
