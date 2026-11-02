"""Paired fixed-owner V2, matched sharded PS and automatic direct-only study.

No accuracy-based parameter search. Previously selected client/FedAdam rates are
frozen. Startup probes, profile exchange and controlled planner/dispatch costs
are inside the simulated horizon. All algorithms execute freshly on one host.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from dasklearn.experiment_runner import provenance, write_json
from benchmark_convergence import execute_jobs, csv_write


def jobs_for(data, duration=900, seeds=(0,1,2)):
    base = json.loads((ROOT/'experiments/conflux-v3/direct-study.json').read_text())['settings']
    base.update(dataset_base_path=str(data), device='cpu', capacity_profile='heterogeneous',
                partitioner='dirichlet', duration=duration, test_period=min(50,max(1,duration//2)), final_heldout=True)
    jobs=[]
    for seed in seeds:
        for algorithm in ['conflux-v2', 'sharded-ps', 'conflux-v3']:
            s=dict(base, algorithm=algorithm, seed=seed)
            jobs.append(dict(name=f'{algorithm}-s{seed}', stage='confirm', variant=algorithm,
                workload='cifar10-gnlenet', algorithm=algorithm, profile='heterogeneous',
                partitioner='dirichlet', seed=seed, learning_rate=s['learning_rate'], settings=s))
    return jobs


def report(output, complete=False):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=json.loads((output/'plan.json').read_text()); duration=plan[0]['settings']['duration']
    records=[json.loads(p.read_text()) for p in sorted((output/'runs').glob('*/record.json'))]
    records=[r for r in records if r['status']=='completed']
    rows=[]; curves=[]; targets=[]; tables={}; hardware={}; profiles=[]
    for r in records:
        directory=output/'runs'/r['name']; data=next(directory.glob('data/*/accuracies.csv')).parent
        frame=pd.read_csv(data/'accuracies.csv').sort_values(['time','peer'])
        if len(frame)!=64*(duration//r['settings']['test_period']) or frame.duplicated(['time','peer']).any():
            raise ValueError('Incomplete/duplicate snapshots')
        if not np.isfinite(frame[['accuracy','loss']]).all().all(): raise ValueError('Nonfinite metrics')
        if not frame.groupby('time').peer.nunique().eq(64).all(): raise ValueError('Missing peers')
        tables[(r['algorithm'], r['seed'])]=frame
        stats=pd.read_csv(data/'client_statistics.csv').sort_values('client')
        signature=stats[['client','training_speed','bandwidth']].to_csv(index=False)
        if r['seed'] in hardware and hardware[r['seed']]!=signature: raise ValueError('Unpaired hardware')
        hardware[r['seed']]=signature
        if stats.bytes_sent.sum()!=stats.bytes_received.sum(): raise ValueError('Wire byte mismatch')
        row=dict(algorithm=r['algorithm'], seed=r['seed'], **r['metrics'], wall_seconds=r['wall_seconds'])
        saved=json.loads((data/'settings.json').read_text())
        row.update(chunks=saved['chunks_in_sample'], reducers=saved['relay_count'],
                   startup_seconds=0., scheduler_cpu_seconds=0., profile_training_tasks=0)
        scheduler=data/'scheduler.json'
        if scheduler.exists():
            s=json.loads(scheduler.read_text())
            row.update(chunks=s['chunks'], reducers=s['reducers'], startup_seconds=s['startup_last_plan_us']/1e6,
                       scheduler_cpu_seconds=s['scheduler_cpu_us']/1e6, profile_training_tasks=64,
                       planner_wall_seconds=s['planner_wall_seconds'])
            profiles.append(dict(seed=r['seed'], **s))
        row['sample_training_tasks']=row['training_tasks']-row['profile_training_tasks']
        rounds=pd.read_csv(data/'conflux_v2_rounds.csv')
        row['completed_recipient_models']=len(rounds)
        row['latest_round']=int(rounds['round'].max()) if len(rounds) else 0
        latest=[]
        for t,g in frame.groupby('time'):
            recipients=g[g['round']==g['round'].max()]
            if recipients.accuracy.nunique()!=1 or recipients.loss.nunique()!=1:
                raise ValueError('Aggregate recipients disagree')
            latest.append(dict(time=t,round=int(recipients['round'].iloc[0]),
                               accuracy=float(recipients.accuracy.iloc[0]),loss=float(recipients.loss.iloc[0])))
        pd.DataFrame(latest).to_csv(directory/'latest-aggregate.csv',index=False)
        curve=pd.read_csv(directory/'curve.csv')
        for t in [30,35,40,45]:
            hit=curve[curve.accuracy>=t]
            upper=float(hit.time.iloc[0]) if len(hit) else None
            targets.append(dict(algorithm=r['algorithm'],seed=r['seed'],target=t,reached=upper is not None,
                lower_seconds=upper-r['settings']['test_period'] if upper is not None else duration,
                upper_seconds=upper))
        curve['algorithm']=r['algorithm'];curve['seed']=r['seed'];curves.extend(curve.to_dict('records'))
        rows.append(row)
    if not rows: return
    matched=[]
    for seed in sorted(hardware):
        if ('conflux-v2',seed) in tables and ('sharded-ps',seed) in tables:
            a=tables[('conflux-v2',seed)]; b=tables[('sharded-ps',seed)]
            cols=['time','peer','round','accuracy','loss']
            equal=a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True))
            matched.append(dict(seed=seed, all_checkpoint_rows_equal=equal))
            if not equal: raise ValueError('Matched baseline differs from V2')
    csv_write(output/'summary.csv',rows);csv_write(output/'curves.csv',curves)
    csv_write(output/'time-to-accuracy.csv',targets)
    write_json(output/'scheduler-details.json',profiles)
    write_json(output/'audit.json',dict(completed_runs=len(rows), matched_baseline_checks=matched,
        paired_hardware=True,byte_conservation=True,common_recipient_models=True,
        frozen_optimizer='client=.01, server Adam=.01, beta1=.9, beta2=.99, tau=.001',
        selection='No accuracy tuning; automatic geometry uses only charged startup measurements',
        hardware='Local CPU execution, fresh runs for all variants; simulated capacities unchanged'))
    frame=pd.DataFrame(rows)
    grouped=frame.groupby('algorithm').agg(n=('seed','nunique'),test_mean=('test_accuracy','mean'),
        test_sd=('test_accuracy','std'),validation_auc=('validation_auc','mean'),
        latest_round=('latest_round','mean'),bytes_mean=('bytes_sent','mean'),
        startup_seconds=('startup_seconds','mean'),scheduler_cpu_seconds=('scheduler_cpu_seconds','mean'))
    grouped.to_csv(output/'comparison.csv')
    names={'conflux-v2':'Conflux V2 (128 chunks / 16 reducers)',
           'sharded-ps':'Matched sharded parameter server', 'conflux-v3':'Automatic direct-only'}
    colors={'conflux-v2':'#0072B2','sharded-ps':'#D55E00','conflux-v3':'#009E73'}
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    cv=pd.DataFrame(curves)
    for alg,g in cv.groupby('algorithm'):
        stats=g.groupby('time').accuracy.agg(['mean','std'])
        axes[0].plot(stats.index,stats['mean'],label=names[alg],color=colors[alg],
                     linestyle='--' if alg=='sharded-ps' else '-')
        axes[0].fill_between(stats.index,stats['mean']-stats['std'].fillna(0),stats['mean']+stats['std'].fillna(0),alpha=.12,color=colors[alg])
    axes[0].set(xlabel='Simulated seconds (startup included)',ylabel='All-node validation accuracy (%)')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    labels=list(grouped.index)
    axes[1].barh([names[a] for a in labels],grouped.test_mean,
                 xerr=grouped.test_sd.fillna(0),color=[colors[a] for a in labels],capsize=4)
    axes[1].set(xlabel='Final full-test accuracy (%)',title=f'{duration}s · mean ± sample SD')
    axes[1].grid(axis='x',alpha=.2)
    fig.suptitle('CIFAR-10 / GNLeNet · 64 nodes / 16 changing trainers\nNon-IID α=0.1 · heterogeneous network and compute · identical FedAdam')
    fig.savefig(output/'comparison.png',dpi=170);fig.savefig(output/'comparison.pdf');plt.close(fig)
    if complete:
        if len(records)!=len(plan) or len(matched)!=len({j['seed'] for j in plan}):
            raise ValueError('Incomplete study')
        (output/'COMPLETE').write_text('All paired runs completed and audited.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['all','plan','report','status'],default='all')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--duration',type=int,default=900)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=p.parse_args(); output=args.output.resolve()
    jobs=jobs_for(args.data.resolve(),args.duration,args.seeds)
    if args.stage=='plan': print(json.dumps(jobs,indent=2));return
    if args.stage=='status':
        print(json.dumps(dict(Counter(json.loads(f.read_text())['status'] for f in (output/'runs').glob('*/record.json')))));return
    if args.stage=='report':report(output);return
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'driver.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    origin=provenance(ROOT)
    if (output/'provenance.json').exists():
        if json.loads((output/'provenance.json').read_text())['source_sha256']!=origin['source_sha256'] or json.loads((output/'plan.json').read_text())!=jobs:
            raise ValueError('Refusing changed source/plan on resume')
    else:
        write_json(output/'provenance.json',origin);write_json(output/'plan.json',jobs)
    execute_jobs(jobs,output,list(range(args.workers)),origin)
    report(output,complete=True)


if __name__=='__main__':main()
