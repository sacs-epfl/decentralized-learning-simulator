import json,os,subprocess,sys,time
from pathlib import Path
import psutil,pandas as pd
root=Path('/mnt/data/mdevos/conflux'); output=root/'results/main128'
variants=['shatter-r2','shatter-r4','conflux-v2']
plan_path=output/'plan.json'; plan=json.loads(plan_path.read_text())
(root/'results/excluded-v1/plan-original48.json').write_text(json.dumps(plan,indent=2)+'\n')
plan['selection']=[x for x in plan['selection'] if x['selected']['variant'] in variants]
plan.update(variants=variants,expected_runs=36,scope_note='User requested Shatter vs Conflux V2; original Conflux V1 excluded.')
plan_path.write_text(json.dumps(plan,indent=2)+'\n')
for path in output.glob('status-gpu*.json'):
 path.rename(root/'results/excluded-v1'/path.name)
jobs=[{'variant':v,'profile':p,'output':str(output/f'{v}-{p}'),'gpu':0 if v=='shatter-r2' else 1,'restarts':0} for p in ['uniform','heterogeneous'] for v in variants]
handles=[]
while True:
 for job in jobs:
  target=Path(job['output'])
  if (target/'COMPLETE').exists():
   job['status']='complete'; job.pop('pid',None); continue
  found=None
  for process in psutil.process_iter(['pid','cmdline','username','status']):
   cmd=process.info['cmdline'] or []
   if '--output' in cmd and cmd[cmd.index('--output')+1]==str(target) and any(x.endswith('run_cifar.py') for x in cmd) and process.info['status']!='zombie':
    found=process.pid; break
  if found:
   job.update(status='running',pid=found); continue
  if job['restarts']>=2: raise RuntimeError(f'Repeated failure: {job}')
  cmd=[sys.executable,str(root/'code/scripts/run_cifar.py'),'--dataset-base-path',str(root/'datasets'),'--output',str(target),'--variants',job['variant'],'--profiles',job['profile'],'--peers','128','--sample-size','32','--chunks','16','--duration','900','--test-period','90','--learning-rate','0.01','--step-ms','20','--bandwidth','100000']
  log=(output/f"{job['variant']}-{job['profile']}.log").open('a'); handles.append(log)
  process=subprocess.Popen(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(job['gpu'])},stdout=log,stderr=subprocess.STDOUT)
  job.update(status='running',pid=process.pid,restarts=job['restarts']+1)
 status={'expected_runs':36,'jobs':jobs,'updated':time.time()}
 tmp=output/'status-core.tmp'; tmp.write_text(json.dumps(status,indent=2)+'\n'); tmp.replace(output/'status-core.json')
 if all(j['status']=='complete' for j in jobs): break
 time.sleep(5)
rows=pd.concat([pd.read_csv(Path(j['output'])/'summary.csv') for j in jobs],ignore_index=True)
curves=pd.concat([pd.read_csv(Path(j['output'])/'curves.csv') for j in jobs],ignore_index=True)
assert len(rows)==36 and not rows.duplicated(['variant','profile','partitioner','seed']).any()
rows.to_csv(output/'summary.csv',index=False); curves.to_csv(output/'curves.csv',index=False)
sys.path.insert(0,str(root/'code/scripts'))
from validate_conflux import plot
plot(curves,output)
subprocess.run([sys.executable,str(root/'code/scripts/report_cifar.py'),'--input',str(output)],check=True)
(output/'COMPLETE').write_text('36 Shatter / Conflux V2 paired runs completed; V1 excluded at user request.\n')
print('CORE_COMPLETE',flush=True)
