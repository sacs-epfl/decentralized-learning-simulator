"""Supplementary one-seed model-size check on the otherwise idle second GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

root=Path('/mnt/data/mdevos/conflux/results/v2-uploads64/tiny-check')
code=Path('/mnt/data/mdevos/conflux/v2-uploads-code')
old=Path('/mnt/data/mdevos/conflux/results/tiny128-gpu-training/conflux-v2')
require=lambda ok,message: None if ok else (_ for _ in ()).throw(RuntimeError(message))
require((root.parent/'heterogeneous'/'COMPLETE').exists(),'Wait for the existing GPU-1 cohort')
root.mkdir(parents=True,exist_ok=True)
if (root/'plan.json').exists():raise RuntimeError('Existing Tiny check; inspect before resuming')
h=hashlib.sha256()
for path in sorted(list((code/'dasklearn').rglob('*.py'))+list((code/'scripts').glob('*.py'))):
    h.update(str(path.relative_to(code)).encode());h.update(path.read_bytes())
require(h.hexdigest()=='c0281ddd84a9367dffd81239d3aa235f863aaaae1b0aa7ba793d382214a08305','Source changed')
commands=[]
for case,chunks,eager in [('reference',16,False),('immediate-wide',32,True)]:
    commands.append([sys.executable,str(code/'scripts/run_tiny.py'),'--output',str(root/case),
        '--data','/mnt/data/mdevos/conflux/tiny-datasets','--variant','conflux-v2',
        '--peers','128','--sample-size','32','--chunks',str(chunks),'--relay-count',str(chunks),
        '--duration','1800','--test-period','300','--seed','0','--device','cuda:0',
        '--immediate-uploads' if eager else '--no-immediate-uploads'])
plan={'status':'running','source_sha256':h.hexdigest(),'gpu':1,'seed':0,'commands':commands,
      'purpose':'Fixed one-seed reference replay and combined-change cross-check on Tiny ImageNet/ResNet18.',
      'curve_examples':2000,'final_examples':10000,'limitation':'Not a Moshpit comparison or a convergence conclusion.'}
(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for case,command in zip(['reference','immediate-wide'],commands):
    print('Starting Tiny '+case,flush=True)
    with (root/f'{case}.log').open('wb') as log:
        subprocess.run(command,cwd=code,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'),
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print((root/case/'summary.csv').read_text(),flush=True)

summaries=[];fresh=[];common=None;capacities=None
for case in ['reference','immediate-wide']:
    folder=root/case;run=folder/'runs'/'conflux-v2'
    state=json.loads((run/'settings.json').read_text())
    excluded={'work_dir','data_dir','chunks_in_sample','relay_count','immediate_uploads'}
    settings={k:v for k,v in state.items() if k not in excluded}
    if common is None:common=settings
    require(settings==common,'Different learning/capacity configuration')
    stats=pd.read_csv(run/'client_statistics.csv').sort_values('client').reset_index(drop=True)
    capacity=stats[['client','training_speed','bandwidth']]
    if capacities is None:capacities=capacity
    pd.testing.assert_frame_equal(capacity,capacities,check_exact=True)
    summary=pd.read_csv(folder/'summary.csv');summary['case']=case;summaries.append(summary)
    require(stats.bytes_sent.sum()==stats.bytes_received.sum()==summary.bytes_sent.iloc[0],'Byte imbalance')
    nodes=pd.read_csv(run/'accuracies.csv')
    require(len(nodes)==768 and not nodes.duplicated(['time','peer']).any(),'Wrong snapshot coverage')
    counts=nodes.groupby('time').peer.nunique()
    require(np.array_equal(counts.index,np.arange(300,1801,300)) and counts.eq(128).all(),'Missing nodes/times')
    final=pd.read_csv(folder/'final.csv')
    require(len(final)==128 and final.peer.nunique()==128,'Wrong final coverage')
    for label,frame in [('curve_2000',nodes),('final_10000',final)]:
        require(np.isfinite(frame[['accuracy','loss']]).all().all(),'Nonfinite metrics')
        for time,g in frame.groupby('time'):
            r=int(g['round'].max());latest=g[g['round']==r]
            require(latest.accuracy.nunique()==1 and latest.loss.nunique()==1,'Inconsistent aggregate')
            fresh.append({'case':case,'metric':label,'time':time,'round':r,
                'accuracy':latest.accuracy.iloc[0],'loss':latest.loss.iloc[0],'recipients':len(latest),
                'mean_staleness_rounds':float((r-g['round']).mean()),'max_staleness_rounds':int((r-g['round']).max())})
    if case=='reference':
        for name,keys in [('accuracies.csv',['time','peer']),('client_statistics.csv',['client'])]:
            pd.testing.assert_frame_equal(pd.read_csv(run/name).sort_values(keys).reset_index(drop=True),
                pd.read_csv(old/'runs'/'conflux-v2'/name).sort_values(keys).reset_index(drop=True),check_exact=True)
        pd.testing.assert_frame_equal(final.sort_values('peer').reset_index(drop=True),
            pd.read_csv(old/'final.csv').sort_values('peer').reset_index(drop=True),check_exact=True)
pd.concat(summaries,ignore_index=True).to_csv(root/'summary.csv',index=False)
pd.DataFrame(fresh).to_csv(root/'latest-v2-aggregate.csv',index=False)
(root/'audit.json').write_text(json.dumps({'passed':True,'source_sha256':h.hexdigest(),
    'runs':2,'curve_node_rows':1536,'full_validation_rows':256,
    'reference_exact_to_previous_gpu_run':True,'same_learning_and_capacities':True,
    'latest_aggregate_recipients_identical':True,'byte_conservation':True,
    'limitation':'One seed and a short low-accuracy horizon; no Moshpit/Tiny comparison.'},indent=2)+'\n')
plan['status']='complete';(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(root/'COMPLETE').write_text('Both Tiny model-size checks completed and audited\n')
print('Tiny checks audited',flush=True)
