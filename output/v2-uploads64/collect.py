"""Wait for the existing jobs, then audit without changing their frozen code."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/mnt/data/mdevos/conflux/results/v2-uploads64')
plan=json.loads((root/'plan.json').read_text())
previous=None
while True:
    counts={}
    for job in plan['jobs']:
        profile=job['profile']
        count=0
        for path in (root/profile).glob('*/summary.csv'):
            count+=max(0,len(path.read_text().splitlines())-1)
        counts[profile]=count
        if not (root/profile/'COMPLETE').exists():
            try: os.kill(job['pid'],0)
            except ProcessLookupError:
                raise RuntimeError(f'{profile} driver exited without COMPLETE; inspect {job["log"]}')
    if counts!=previous:
        print(json.dumps({'completed_runs':counts}),flush=True);previous=counts
    if all((root/p/'COMPLETE').exists() for p in plan['profiles']):break
    time.sleep(15)
subprocess.run([sys.executable,str(root/'report.py')],check=True)
(root/'COMPLETE').write_text('All 60 runs completed and audited\n')
