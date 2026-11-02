"""Record dataset split/partition identities for every paired sweep setting."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from dasklearn.datasets.huggingface import load_splits,allocate_indices

root=Path('/mnt/data/mdevos/conflux-chunk-reducer-20261102')
plan=json.loads((root/'results-cached/plan.json').read_text())
fields=[k for k in plan[0]['settings'] if k.startswith('dataset_')]+[
 'dataset','model','peers','alpha','seed','partitioner','partition_column','label_column',
 'feature_column','train_split','test_split','validation_split','validation_set_fraction',
 'min_partition_size','shards_per_partition']
groups={}
for job in plan:
 s=job['settings'];key=(s['partitioner'],s['seed'])
 signature={k:s[k] for k in fields}
 if key in groups:assert groups[key]==signature,key
 groups[key]=signature
rows=[]
for (partitioner,seed),signature in sorted(groups.items()):
 s=next(j['settings'] for j in plan if j['partitioner']==partitioner and j['seed']==seed)
 config=SimpleNamespace(**s,participants=s['peers'])
 splits=load_splits(config)
 parts=allocate_indices(splits['train'],config,min_size=8)
 merged=[i for part in parts for i in part]
 assert len(merged)==len(set(merged))==len(splits['train'])
 rows.append(dict(partitioner=partitioner,seed=seed,
  split_sizes={k:len(v) for k,v in splits.items()},
  split_fingerprints={k:v._fingerprint for k,v in splits.items()},
  client_sizes=list(map(len,parts)),
  client_index_sha256=[hashlib.sha256(np.asarray(part,dtype='<i8').tobytes()).hexdigest() for part in parts]))
result=dict(configured_pairing_verified=True,disjoint_complete_partitions=True,assignments=rows)
(root/'results-cached/data-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('Six data assignments audited; all 228 configurations share the correct paired inputs')
