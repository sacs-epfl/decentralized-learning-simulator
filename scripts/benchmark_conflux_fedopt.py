"""Validation-only tuning then paired reducer FedOpt / handoff confirmation.

Every numerical run is a normal version-1 main.py configuration. The generic
resumable executor supplies source/configuration guards and full final tests.
No source changes or historical-run pooling are allowed on resume.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import os
from pathlib import Path
import queue
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from args import create_parser
from dasklearn.experiments import resolve_experiment
from dasklearn.experiment_runner import provenance,write_json
from benchmark_convergence import execute_job,csv_write


def tuning_jobs(config):
    exp=resolve_experiment(config,create_parser())
    jobs=[]
    for run in exp.runs:
        family=next(name for name in ['server-sgd','server-momentum','server-adam'] if name in run.name)
        s=run.options
        jobs.append(dict(name='tune-'+run.name,stage='tune',family=family,variant=family,
            workload='cifar10-gnlenet',algorithm=s['algorithm'],profile=s['capacity_profile'],
            partitioner=s['partitioner'],seed=s['seed'],learning_rate=s['learning_rate'],settings=s))
    return jobs


def execute(jobs,output,gpus,origin,allow_divergence=False):
    work=queue.Queue()
    for job in jobs:work.put(job)
    def worker(gpu):
        while True:
            try:job=work.get_nowait()
            except queue.Empty:return
            try:execute_job(job,output,gpu,origin)
            except RuntimeError:
                path=output/'runs'/job['name']
                log=(path/'run.log').read_text()
                record=json.loads((path/'record.json').read_text())
                numerical='Nonfinite' in log or 'Nonfinite' in record.get('error','')
                if not (allow_divergence and numerical):raise
                record['numerical_divergence']=True
                write_json(path/'record.json',record)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in gpus]
        for future in futures:future.result()


def records(output):
    result=[]
    for path in sorted((output/'runs').glob('*/record.json')):
        r=json.loads(path.read_text())
        if path.parent.name==r['name']:result.append(r)
    return result


def selection(output):
    selected={}
    for family in ['server-sgd','server-momentum','server-adam']:
        candidates=[r for r in records(output) if r['stage']=='tune' and r['family']==family and r['status']=='completed']
        if not candidates:raise ValueError(f'No completed validation candidate for {family}')
        best=max(candidates,key=lambda r:(r['metrics']['validation_auc'],-r['settings']['server_learning_rate']))
        selected[family]=dict(run=best['name'],validation_auc=best['metrics']['validation_auc'],settings=best['settings'])
    write_json(output/'selection.json',selected)
    return selected


def confirmation_jobs(selected):
    variants=[]
    for family,item in selected.items():variants.append((family+'-fixed',family,dict(item['settings'])))
    adam=selected['server-adam']['settings']
    for suffix,epoch,bits in [('rotate-fp32',1,32),('rotate-int8',1,8),('epoch8-int8',8,8)]:
        variants.append(('server-adam-'+suffix,'server-adam',dict(adam,reducer_epoch_rounds=epoch,state_bits=bits)))
    base=selected['server-sgd']['settings']
    for suffix,epoch in [('fixed',0),('rotating',1)]:
        variants.append(('original-fedavg-'+suffix,'original-fedavg',dict(base,server_optimizer='fedavg',
            server_learning_rate=1.,learning_rate=.01,reducer_epoch_rounds=epoch)))
    variants.append(('moshpit','moshpit',dict(base,algorithm='moshpit',learning_rate=.01,
        server_optimizer='fedavg',server_learning_rate=1.)))
    jobs=[]
    for seed in [0,1,2]:
        for variant,family,s in variants:
            s=dict(s,seed=seed,final_heldout=True)
            jobs.append(dict(name=f'confirm-{variant}-s{seed}',stage='confirm',family=family,variant=variant,
                workload='cifar10-gnlenet',algorithm=s['algorithm'],profile=s['capacity_profile'],
                partitioner=s['partitioner'],seed=seed,learning_rate=s['learning_rate'],settings=s))
    return jobs


def report(output,complete=False):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_records=records(output)
    rows=[];curves=[];tta=[];signatures={}
    for r in all_records:
        s=r['settings']
        row={k:r[k] for k in ['name','stage','family','variant','seed','status']}
        row.update(client_lr=s['learning_rate'],server_lr=s['server_learning_rate'],
            server_optimizer=s['server_optimizer'],epoch_rounds=s['reducer_epoch_rounds'],state_bits=s['state_bits'])
        row.update(r.get('metrics',{}));row['wall_seconds']=r.get('wall_seconds')
        if r['status']=='completed':
            directory=output/'runs'/r['name']
            artifacts=list(directory.glob('data/*/accuracies.csv'))
            data=pd.read_csv(artifacts[0])
            if len(data)!=64*18 or data.duplicated(['time','peer']).any():raise ValueError('Incomplete snapshots')
            if not data.groupby('time').peer.nunique().eq(64).all():raise ValueError('Missing evaluated peers')
            if not np.isfinite(data[['accuracy','loss']]).all().all():raise ValueError('Nonfinite snapshots')
            stats=pd.read_csv(artifacts[0].parent/'client_statistics.csv').sort_values('client')
            sig=stats[['client','training_speed','bandwidth']].to_csv(index=False)
            if r['seed'] in signatures and signatures[r['seed']]!=sig:raise ValueError('Unpaired hardware')
            signatures[r['seed']]=sig
            if stats.bytes_sent.sum()!=stats.bytes_received.sum():raise ValueError('Wire bytes disagree')
            handoffs=artifacts[0].parent/'conflux_optimizer_handoffs.csv'
            row['state_handoffs']=0;row['state_payload_bytes_scheduled']=0
            if handoffs.exists():
                h=pd.read_csv(handoffs);row['state_handoffs']=len(h);row['state_payload_bytes_scheduled']=int(h.payload_bytes.sum())
            if r['stage']=='confirm':
                frame=pd.read_csv(directory/'curve.csv')
                for k in ['variant','seed']:frame[k]=r[k]
                curves.extend(frame.to_dict('records'))
                for target in [25,30,35,40,45]:
                    hit=frame[frame.accuracy>=target]
                    upper=float(hit.time.iloc[0]) if len(hit) else None
                    tta.append(dict(variant=r['variant'],seed=r['seed'],target=target,reached=upper is not None,
                        lower_s=upper-s['test_period'] if upper is not None else s['duration'],
                        upper_s=upper,restricted_s=upper if upper is not None else s['duration']))
        rows.append(row)
    csv_write(output/'all-runs.csv',rows)
    csv_write(output/'curves.csv',curves);csv_write(output/'time-to-accuracy.csv',tta)
    df=pd.DataFrame(rows)
    confirm=df[(df.stage=='confirm')&(df.status=='completed')]
    if len(confirm):
        grouped=confirm.groupby('variant').agg(n=('seed','nunique'),test_mean=('test_accuracy','mean'),
            test_sd=('test_accuracy','std'),validation_auc_mean=('validation_auc','mean'),
            bytes_mean=('bytes_sent','mean'),training_tasks_mean=('training_tasks','mean'),
            handoff_payload_mean=('state_payload_bytes_scheduled','mean'))
        grouped.to_csv(output/'comparison.csv')
        fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
        frame=pd.DataFrame(curves)
        for variant,g in frame.groupby('variant'):
            v=g.groupby('time').accuracy.agg(['mean','std'])
            axes[0].plot(v.index,v['mean'],label=variant)
            axes[0].fill_between(v.index,v['mean']-v['std'],v['mean']+v['std'],alpha=.1)
        axes[0].set(xlabel='Simulated elapsed seconds',ylabel='All-node validation accuracy (%)',title='Validation convergence (1,000 fixed images)')
        axes[0].grid(alpha=.2);axes[0].legend(fontsize=7)
        ordered=grouped.sort_values('test_mean')
        axes[1].barh(ordered.index,ordered.test_mean,xerr=ordered.test_sd,capsize=3)
        axes[1].set(xlabel='Mean full-test accuracy (%)',title='Final 900-second checkpoints (10,000 images)')
        axes[1].grid(axis='x',alpha=.2)
        fig.suptitle('Reducer-side FedOpt · CIFAR-10/GNLeNet · 64 nodes / 16 trainers\nNon-IID α=0.1 · heterogeneous hardware · three paired seeds',fontsize=13)
        fig.savefig(output/'fedopt-comparison.png',dpi=170);fig.savefig(output/'fedopt-comparison.pdf');plt.close(fig)
    if complete:
        if len(confirm)!=27 or not confirm.groupby('variant').seed.nunique().eq(3).all():raise ValueError('Incomplete confirmation')
        if any(r['status']!='completed' and not r.get('numerical_divergence') for r in all_records):raise ValueError('Unexpected run failure')
        write_json(output/'audit.json',dict(tuning_runs=36,confirmation_runs=27,paired_hardware=True,
            complete_node_snapshots=True,finite_metrics=True,byte_conservation=True,
            selection='Validation AUC on seed 10011, then frozen settings on seeds 0/1/2; test scores never used for selection',
            server_compute_ms_per_chunk=.1,state_pack_compute_ms_per_migrated_chunk=.05,
            handoff_accounting='FP32 base + FP32 or blockwise 8-bit moments + uint64 step + normal header/metadata'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'experiments/conflux-v2/fedopt/tuning.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,default=[0,1])
    p.add_argument('--stage',choices=['all','tune','confirm','report','status','plan'],default='all')
    args=p.parse_args();output=args.output.resolve()
    if args.stage=='status':
        r=records(output);print(json.dumps(dict(counts=dict(Counter(x['stage']+'/'+x['status'] for x in r)),
            running=[dict(name=x['name'],gpu=x['gpu']) for x in r if x['status']=='running'],complete=(output/'COMPLETE').exists()),indent=2));return
    if args.stage=='report':report(output);return
    jobs=tuning_jobs(args.config)
    if args.stage=='plan':print(json.dumps(dict(tuning_runs=len(jobs),confirmation_runs=27)));return
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'driver.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    origin=provenance(ROOT)
    exported=ROOT/'source-export.json'
    if exported.exists():origin['source_export']=json.loads(exported.read_text())
    if (output/'provenance.json').exists():
        if json.loads((output/'provenance.json').read_text())['source_sha256']!=origin['source_sha256'] or json.loads((output/'tuning-plan.json').read_text())!=jobs:
            raise ValueError('Refusing changed source/plan on resume')
    else:write_json(output/'provenance.json',origin);write_json(output/'tuning-plan.json',jobs)
    if args.stage in ['all','tune']:
        execute(jobs,output,args.gpus,origin,allow_divergence=True);selected=selection(output)
    else:selected=json.loads((output/'selection.json').read_text())
    if args.stage in ['all','confirm']:
        confirmed=confirmation_jobs(selected);write_json(output/'confirmation-plan.json',confirmed)
        execute(confirmed,output,args.gpus,origin)
        report(output,complete=True);(output/'COMPLETE').write_text('All selected paired confirmations completed and audited\n')
    else:report(output)


if __name__=='__main__':main()
