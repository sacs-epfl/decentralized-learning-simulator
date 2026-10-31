"""Wait for the existing coordinator, then audit; never launches experiments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/mnt/data/mdevos/conflux/results/demo64')
last=None
while True:
    plan=json.loads((root/'plan.json').read_text())
    counts={profile:sum(max(0,len(f.read_text().splitlines())-1) for f in (root/profile).glob('*/summary.csv')) for profile in ['uniform','heterogeneous']}
    counts['validation']=sum(max(0,len(f.read_text().splitlines())-1) for f in (root/'validation').glob('*/summary.csv'))
    current=plan['status'],counts
    if current!=last:print(json.dumps({'status':plan['status'],'completed':counts}),flush=True);last=current
    if plan['status']=='ready-for-audit':break
    if plan['status']=='complete':sys.exit(0)
    try:os.kill(489437,0)
    except ProcessLookupError:raise RuntimeError('Coordinator exited before completion; inspect coordinator.log')
    time.sleep(15)
subprocess.run([sys.executable,str(root/'report.py')],check=True)
