"""Validation-only LR selection and a one-step CIFAR-10 comparison.

Run tune workers with distinct --slot values, then select, then evaluate workers.
Every job has a source/configuration manifest and is independently resumable.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
from validate_algorithms import run_one


def digest():
    root=Path(__file__).resolve().parents[1];h=hashlib.sha256()
    for path in sorted(list((root/'dasklearn').rglob('*.py'))+list((root/'scripts').glob('*.py'))):
        h.update(str(path.relative_to(root)).encode());h.update(path.read_bytes())
    return h.hexdigest()


def candidates(args):
    return [(v,'uniform',p,0,rate) for v in args.variants for rate in args.rates for p in ['iid','dirichlet']]


def folder(root,phase,variant,profile,partitioner,seed,rate):
    return root/phase/f'{variant}-{profile}-{partitioner}-s{seed}-lr{rate:g}'


def configuration(args,output,rate,tune):
    return SimpleNamespace(output=output,dataset='cifar10',model='gnlenet',dataset_base_path=args.dataset_base_path,
        device=args.device,peers=args.peers,sample_size=args.peers//4,chunks=16,group_size=8,averaging_steps=1,
        success_fraction=1.,relay_count=16,immediate_uploads=True,alpha=.1,
        learning_rate=rate,momentum=0.,weight_decay=0.,batch_size=32,local_steps=1,
        duration=args.duration,test_period=args.duration if tune else args.duration/6,
        bandwidth=100000,step_ms=20,latency_ms=20,torch_threads=4,model_bytes=346251,
        equal_capacity=True,streaming=True,gpu_residency=True,training_batching=True,
        training_batch_size=2,gpu_state_cache_mb=256,evaluation_split='validation',
        evaluation_examples=0 if tune else 1000,final_test=not tune,
        qgm_beta=.9,qgm_mu=.9,qgm_nesterov=args.nesterov,qgm_topology='ring',qgm_degree=4,qgm_mode='quasi')


def manifest(args,source):
    return {'source_sha256':source,'variants':args.variants,'rates':args.rates,'seeds':args.seeds,
        'peers':args.peers,'duration':args.duration,'nesterov':args.nesterov,
        'dataset_base_path':str(Path(args.dataset_base_path).resolve()),
        'torch':torch.__version__,'capacity':'same mean 20ms/SGD step and 100000 B/s; independent compute/BW heterogeneity',
        'data':'45000 training, disjoint 5000 validation, disjoint 10000 test; GN-LeNet, no augmentation',
        'learning':'one gradient step per exchange; beta=mu=.9; no weight decay; V2 S=N/4, K=R=16, immediate uploads',
        'selection_rule':'Per variant, highest mean uniform IID/Dirichlet full validation accuracy at final time, seed 0; lower LR breaks ties.',
        'curves':'Balanced fixed 1000-image validation subset; final full 10000-image test, all nodes',
        'limitation':'Small validation LR grid; does not reproduce the paper model, epoch schedule, or tuned recipe'}


async def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    source=digest();expected=manifest(args,source);path=root/'plan.json'
    # Identical content across workers; atomically publishing distinct temp files
    # makes simultaneous initial startup safe.
    if path.exists():
        if json.loads(path.read_text())!=expected:raise ValueError('Source/configuration differs from frozen plan')
    else:
        temporary=path.with_name(f'plan-slot{args.slot}.tmp');temporary.write_text(json.dumps(expected,indent=2)+'\n');temporary.replace(path)
    if args.phase=='select':
        rows=[]
        for v,profile,p,seed,rate in candidates(args):
            file=folder(root,'tune',v,profile,p,seed,rate)/'summary.json'
            record=json.loads(file.read_text())
            rows.append({'variant':v,'partitioner':p,'learning_rate':rate,
                         'validation_accuracy':record['final_accuracy'] if record.get('status')!='diverged' else -float('inf')})
        scores=pd.DataFrame(rows).groupby(['variant','learning_rate'],as_index=False).validation_accuracy.mean()
        if not scores.groupby('variant').validation_accuracy.max().map(np.isfinite).all():
            raise ValueError('A variant has no finite validation candidate')
        selected={v:float(g.sort_values(['validation_accuracy','learning_rate'],ascending=[False,True]).iloc[0].learning_rate)
                  for v,g in scores.groupby('variant')}
        scores.to_csv(root/'validation-scores.csv',index=False)
        (root/'selection.json').write_text(json.dumps({'selected':selected,'rule':expected['selection_rule'],
            'rates':args.rates,'source_sha256':source},indent=2)+'\n')
        print(json.dumps({'selected':selected}),flush=True);return
    if args.phase=='tune':jobs=candidates(args)
    else:
        selection=json.loads((root/'selection.json').read_text())
        if selection['source_sha256']!=source:raise ValueError('Selection source differs')
        jobs=[(v,profile,p,seed,selection['selected'][v]) for profile in ['uniform','heterogeneous']
              for p in ['iid','dirichlet'] for seed in args.seeds for v in args.variants]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for index,(variant,profile,partitioner,seed,rate) in enumerate(jobs):
        if index%args.slots!=args.slot:continue
        output=folder(root,args.phase,variant,profile,partitioner,seed,rate)
        output.mkdir(parents=True,exist_ok=True)
        if (output/'COMPLETE').exists():
            print(json.dumps({'skip':str(output)}),flush=True);continue
        config=configuration(args,output,rate,args.phase=='tune')
        print(json.dumps({'starting':str(output),'job_index':index,'total_jobs':len(jobs)}),flush=True)
        row,curve=await run_one(config,variant,profile,partitioner,seed)
        finite=np.isfinite([row['final_accuracy'],row['final_loss']]).all()
        row['status']='passed' if finite else 'diverged'
        if not finite:
            if args.phase!='tune':raise ValueError('Nonfinite evaluation metrics')
            row['final_accuracy']=None;row['final_loss']=None
        if args.phase=='evaluate':
            if not np.isfinite([row['final_test_accuracy'],row['final_test_loss']]).all():raise ValueError('Nonfinite test metrics')
            run_dir=output/'runs'/f'{variant}-{profile}-{partitioner}-s{seed}'
            heldout=pd.read_csv(run_dir/'heldout.csv')
            if len(heldout)!=args.peers or not heldout.peer.nunique()==args.peers:raise ValueError('Incomplete heldout models')
            if variant=='conflux-v2':
                latest_round=int(heldout['round'].max());latest=heldout[heldout['round']==latest_round]
                if latest.accuracy.nunique()!=1 or latest.loss.nunique()!=1:raise ValueError('Inconsistent final V2 aggregate recipients')
                row.update(final_test_latest_aggregate_accuracy=float(latest.accuracy.iloc[0]),
                    final_test_latest_aggregate_loss=float(latest.loss.iloc[0]),
                    mean_staleness_rounds=float((latest_round-heldout['round']).mean()),
                    max_staleness_rounds=int((latest_round-heldout['round']).max()))
        row.update(source_sha256=source,phase=args.phase,nesterov=args.nesterov)
        (output/'summary.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
        curve.to_csv(output/'curves.csv',index=False)
        (output/'COMPLETE').write_text('All requested node metrics validated\n')
    (root/f'{args.phase}-slot{args.slot}-COMPLETE').write_text('All assigned jobs complete\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset-base-path',required=True)
    p.add_argument('--phase',choices=['tune','select','evaluate'],required=True)
    p.add_argument('--slot',type=int,default=0);p.add_argument('--slots',type=int,default=2)
    p.add_argument('--variants',nargs='+',default=['qgm','dsgdm','dsgd'])
    p.add_argument('--rates',nargs='+',type=float,default=[.001,.003,.01,.03,.1])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--peers',type=int,default=64);p.add_argument('--duration',type=float,default=900)
    p.add_argument('--device',default='cuda:0');p.add_argument('--nesterov',action='store_true')
    args=p.parse_args()
    if not 0<=args.slot<args.slots:p.error('slot must be in [0, slots)')
    asyncio.run(main(args))
