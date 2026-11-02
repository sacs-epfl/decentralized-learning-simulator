"""Compare unchanged default protocols with the prior chunk/reducer sweep."""
from pathlib import Path
import json,tarfile
import pandas as pd
root=Path(__file__).resolve().parent
old_path=root.parent/'conflux-v2-chunk-reducer-64-20261102/measurements.tgz'
checks=[]
with tarfile.open(old_path,'r:gz') as old,tarfile.open(root/'measurements.tgz','r:gz') as new:
 old_names=old.getnames();new_names=new.getnames()
 for member in old.getmembers():
  if not member.name.endswith('/record.json'):continue
  r=json.load(old.extractfile(member));s=r['settings']
  if s['partitioner']!='dirichlet' or s['capacity_profile']!='heterogeneous':continue
  algorithm=s['algorithm']
  if algorithm=='conflux-v2' and (s['chunks_in_sample'],s['relay_count'])!=(128,16):continue
  if algorithm not in ['conflux-v2','moshpit']:continue
  prefix=member.name.rsplit('/',1)[0]
  accuracy=next(n for n in old_names if n.startswith(prefix+'/data/') and n.endswith('/accuracies.csv'))
  frame=pd.read_csv(old.extractfile(accuracy));frame=frame[frame.time==900].sort_values('peer')
  label='original-fedavg-rotating' if algorithm=='conflux-v2' else 'moshpit'
  newprefix=f'confirm-{label}-s{s["seed"]}/data/'
  heldout=[n for n in new_names if n.startswith(newprefix) and n.rsplit('/',1)[-1].startswith('heldout_') and n.endswith('.csv')]
  result=pd.concat([pd.read_csv(new.extractfile(n)) for n in heldout]).sort_values('peer')
  assert len(frame)==len(result)==64
  assert frame['round'].tolist()==result['round'].tolist()
  error=(frame[['accuracy','loss']].reset_index(drop=True)-result[['accuracy','loss']].reset_index(drop=True)).abs().max()
  checks.append(dict(algorithm=algorithm,seed=s['seed'],max_accuracy_difference=float(error.accuracy),max_loss_difference=float(error.loss)))
assert len(checks)==6
(root/'baseline-replay-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
assert all(c['max_accuracy_difference']==0 and c['max_loss_difference']==0 for c in checks)
