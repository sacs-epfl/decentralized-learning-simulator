"""Run ordinary experiment configs on two GPUs and report chunk/reducer effects.

Uses the shared config resolver and resumable job executor. Every run is a fresh
main.py --config process. Full test curves describe a fixed sweep, without
selecting or tuning on test results. Historical runs are never pooled in.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from args import create_parser
from dasklearn.experiments import resolve_experiment
from dasklearn.experiment_runner import provenance, write_json
from benchmark_convergence import csv_write, execute_jobs


def jobs_for(configs):
    jobs = []
    for config in configs:
        experiment = resolve_experiment(config, create_parser())
        for run in experiment.runs:
            s = run.options
            if run.execution != 'local' or s['algorithm'] not in {'conflux-v2', 'moshpit'}:
                raise ValueError('Expected local Conflux V2/Moshpit configurations')
            if s['evaluation_split'] != 'test' or s['evaluation_examples'] != 0 or s['final_heldout']:
                raise ValueError('This fixed sweep requires full test snapshots')
            jobs.append(dict(name=f'{experiment.name}-{run.name}', stage='sweep',
                workload='cifar10-gnlenet', algorithm=s['algorithm'],
                profile=s['capacity_profile'], partitioner=s['partitioner'],
                seed=s['seed'], learning_rate=s['learning_rate'], settings=s))
    if len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate run names')
    # Start the comparator and reference cell first in each scenario.
    return sorted(jobs, key=lambda j: (j['seed'], j['partitioner'], j['profile'],
        j['algorithm'] != 'moshpit',
        (j['settings']['chunks_in_sample'],j['settings']['relay_count']) != (16,16),
        j['settings']['chunks_in_sample'], j['settings']['relay_count']))


def status(output):
    records = []
    for path in sorted((output/'runs').glob('*/record.json')):
        record = json.loads(path.read_text())
        if path.parent.name == record['name']:
            records.append(record)
    return dict(counts=dict(Counter(r['status'] for r in records)),
        running=[dict(name=r['name'],gpu=r['gpu'],elapsed_s=round(time.time()-r['started_at']))
                 for r in records if r['status']=='running'],
        failed=[r['name'] for r in records if r['status']=='failed'],
        complete=(output/'COMPLETE').exists())


def report(output, require_complete=False):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plan = json.loads((output/'plan.json').read_text())
    expected = {j['name']:j for j in plan}
    rows, curves, latest, signatures = [], [], [], {}
    source = json.loads((output/'provenance.json').read_text())['source_sha256']
    for name, job in expected.items():
        directory = output/'runs'/name
        path = directory/'record.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if r['status'] != 'completed':
            continue
        if r['settings'] != job['settings'] or r['source_sha256'] != source:
            raise ValueError(f'Run provenance mismatch: {name}')
        s = r['settings']
        artifacts = list(directory.glob('data/*/accuracies.csv'))
        if len(artifacts) != 1:
            raise ValueError(f'Missing or ambiguous accuracy file: {name}')
        data = pd.read_csv(artifacts[0])
        keys = ['time','peer']
        if data.duplicated(keys).any() or len(data) != s['peers']*(s['duration']//s['test_period']):
            raise ValueError(f'Incomplete or duplicate checkpoint rows: {name}')
        if not set(data.peer) == set(range(s['peers'])) or not np.isfinite(data[['accuracy','loss']]).all().all():
            raise ValueError(f'Invalid node metrics: {name}')
        frame = pd.read_csv(directory/'curve.csv')
        if list(frame.time) != list(range(s['test_period'], s['duration']+1,s['test_period'])):
            raise ValueError(f'Wrong checkpoints: {name}')
        stats = pd.read_csv(artifacts[0].parent/'client_statistics.csv').sort_values('client')
        signature = stats[['client','training_speed','bandwidth']].to_csv(index=False)
        key = (s['capacity_profile'],s['seed'])
        if key in signatures and signature != signatures[key]:
            raise ValueError(f'Unpaired physical capacities: {name}')
        signatures[key] = signature
        if int(stats.bytes_sent.sum()) != int(stats.bytes_received.sum()):
            raise ValueError(f'Byte-accounting mismatch: {name}')
        saved = json.loads((artifacts[0].parent/'settings.json').read_text())
        if saved['duration'] != s['duration']*1000000 or saved['model_size_bytes'] != 346251:
            raise ValueError(f'Wrong duration/payload: {name}')
        row = dict(name=name, algorithm=s['algorithm'], profile=s['capacity_profile'],
            partitioner=s['partitioner'], seed=s['seed'],
            chunks=s['chunks_in_sample'] if s['algorithm']=='conflux-v2' else 0,
            reducers=s['relay_count'] if s['algorithm']=='conflux-v2' else 0,
            final_accuracy=float(frame.accuracy.iloc[-1]),final_loss=float(frame.loss.iloc[-1]),
            bytes_sent=int(stats.bytes_sent.sum()),training_tasks=r['metrics']['training_tasks'],
            wall_seconds=r['wall_seconds'])
        if s['algorithm']=='conflux-v2':
            from dasklearn.simulation.conflux_v2.client import _layout
            completions = pd.read_csv(artifacts[0].parent/'conflux_v2_rounds.csv')
            first = completions.groupby('round').time_us.min().sort_index()/1e6
            capacities = tuple(zip(stats.client.astype(int),stats.bandwidth.astype(int)))
            active_reducers = [len(set(_layout(int(rn),s['chunks_in_sample'],s['relay_count'],capacities)[1]))
                               for rn in first.index]
            row.update(latest_round=int(completions['round'].max()),
                mean_aggregate_interval_s=float(first.diff().dropna().mean()),
                effective_reducers_mean=float(np.mean(active_reducers)),
                effective_reducers_min=min(active_reducers),effective_reducers_max=max(active_reducers))
            diagnostic = pd.read_csv(directory/'latest-aggregate.csv')
            diagnostic['name'] = name
            latest.extend(diagnostic.to_dict('records'))
        rows.append(row)
        for key in ['name','algorithm','profile','partitioner','seed','chunks','reducers']:
            frame[key] = row[key]
        curves.extend(frame.to_dict('records'))
    if require_complete and len(rows) != len(expected):
        raise ValueError(f'Only {len(rows)}/{len(expected)} completed runs')
    if not rows:
        return
    csv_write(output/'summary.csv', rows)
    csv_write(output/'curves.csv', curves)
    csv_write(output/'latest-aggregate.csv',latest)
    df = pd.DataFrame(rows)
    baseline = df[df.algorithm=='moshpit'][['profile','partitioner','seed','final_accuracy']].rename(
        columns={'final_accuracy':'moshpit_accuracy'})
    v2 = df[df.algorithm=='conflux-v2'].merge(baseline,on=['profile','partitioner','seed'],validate='many_to_one')
    v2['difference_pp'] = v2.final_accuracy-v2.moshpit_accuracy
    csv_write(output/'paired-moshpit.csv',v2.to_dict('records'))
    summary = v2.groupby(['profile','partitioner','chunks','reducers']).agg(
        accuracy_mean=('final_accuracy','mean'),accuracy_sd=('final_accuracy','std'),
        difference_mean_pp=('difference_pp','mean'),difference_sd_pp=('difference_pp','std'),
        seeds=('seed','nunique'),training_tasks_mean=('training_tasks','mean'),
        aggregate_interval_s=('mean_aggregate_interval_s','mean'),
        effective_reducers_mean=('effective_reducers_mean','mean')).reset_index()
    summary.to_csv(output/'cells.csv',index=False)
    baseline.groupby(['profile','partitioner']).moshpit_accuracy.agg(['mean','std','count']).to_csv(output/'moshpit.csv')
    chunks = sorted({j['settings']['chunks_in_sample'] for j in plan if j['algorithm']=='conflux-v2'})
    reducers = sorted({j['settings']['relay_count'] for j in plan if j['algorithm']=='conflux-v2'})
    for delta in [False,True]:
        fig,axes = plt.subplots(2,2,figsize=(14,10),layout='constrained')
        values = summary.difference_mean_pp if delta else summary.accuracy_mean
        limit = max(1,float(values.abs().max())) if delta else None
        norm = TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit) if delta else None
        for row,partitioner in enumerate(['iid','dirichlet']):
            for col,profile in enumerate(['uniform','heterogeneous']):
                ax = axes[row,col]
                group = summary[(summary.profile==profile)&(summary.partitioner==partitioner)]
                matrix = np.full((len(reducers),len(chunks)),np.nan)
                for _,cell in group.iterrows():
                    matrix[reducers.index(int(cell.reducers)),chunks.index(int(cell.chunks))] = cell.difference_mean_pp if delta else cell.accuracy_mean
                im = ax.imshow(np.ma.masked_invalid(matrix),origin='lower',aspect='auto',
                    cmap='RdBu_r' if delta else 'viridis',norm=norm,
                    **({} if delta else dict(vmin=float(summary.accuracy_mean.min()),vmax=float(summary.accuracy_mean.max()))))
                for i,rn in enumerate(reducers):
                    for k,cn in enumerate(chunks):
                        cell = group[(group.reducers==rn)&(group.chunks==cn)]
                        if rn>cn:
                            label='—'
                        elif len(cell):
                            c=cell.iloc[0]
                            label=(f'{c.difference_mean_pp:+.2f}\n±{c.difference_sd_pp:.2f}' if delta
                                   else f'{c.accuracy_mean:.2f}\n±{c.accuracy_sd:.2f}')
                            if c.seeds != 3:
                                label+=f'\nn={int(c.seeds)}'
                        else:
                            label='pending'
                        rgba = im.cmap(im.norm(matrix[i,k])) if np.isfinite(matrix[i,k]) else (1,1,1,1)
                        luminance = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                        ax.text(k,i,label,ha='center',va='center',fontsize=9,color='black' if luminance>.5 else 'white')
                b = baseline[(baseline.profile==profile)&(baseline.partitioner==partitioner)].moshpit_accuracy
                ax.set(title=f'{profile.capitalize()} hardware · {"IID" if partitioner=="iid" else "Non-IID (α=0.1)"}\nMoshpit: {b.mean():.2f} ± {b.std():.2f}% (n={len(b)})',
                    xlabel='Number of chunks',ylabel='Reducer cap',xticks=range(len(chunks)),xticklabels=chunks,
                    yticks=range(len(reducers)),yticklabels=reducers)
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Accuracy difference from Moshpit (percentage points)' if delta else 'All-node test accuracy (%)')
        fig.suptitle('Conflux V2 − Moshpit at 900 simulated seconds' if delta else 'Conflux V2 final test accuracy at 900 simulated seconds',fontsize=17)
        fig.supxlabel('Cell: mean ± sample SD across three paired seeds. All 64 stored node checkpoints; full 10,000-image test set.\nReducer caps above chunk count are omitted. A cap may yield fewer active reducers on heterogeneous hardware.',fontsize=10)
        stem = 'moshpit-difference-heatmaps' if delta else 'accuracy-heatmaps'
        fig.savefig(output/f'{stem}.png',dpi=180)
        fig.savefig(output/f'{stem}.pdf')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for row,partitioner in enumerate(['iid','dirichlet']):
        for col,profile in enumerate(['uniform','heterogeneous']):
            ax=axes[row,col]
            group=summary[(summary.profile==profile)&(summary.partitioner==partitioner)]
            for rn,g in group.groupby('reducers'):
                ax.errorbar(g.chunks,g.accuracy_mean,yerr=g.accuracy_sd,marker='o',capsize=3,label=f'{rn} reducers')
            b=baseline[(baseline.profile==profile)&(baseline.partitioner==partitioner)].moshpit_accuracy
            ax.axhline(b.mean(),color='black',linestyle='--',label='Moshpit')
            ax.axhspan(b.mean()-b.std(),b.mean()+b.std(),color='black',alpha=.08)
            ax.set_xscale('log',base=2)
            ax.set(title=f'{profile.capitalize()} · {partitioner}',xlabel='Number of chunks',ylabel='Final all-node test accuracy (%)',xticks=chunks,xticklabels=chunks)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Chunk size and reducer count at a fixed 900-second budget',fontsize=16)
    fig.savefig(output/'accuracy-by-chunks.png',dpi=180)
    fig.savefig(output/'accuracy-by-chunks.pdf')
    plt.close(fig)
    write_json(output/'audit.json',dict(completed_runs=len(rows),expected_runs=len(expected),
        per_node_checkpoint_rows=len(rows)*64*6,paired_capacities=True,finite_metrics=True,
        complete_snapshots=True,byte_conservation=True,source_sha256=source,
        metric='Mean full-test accuracy across all 64 stored completed node checkpoints at 900 simulated seconds',
        selection='Fixed grid; report all cells. Descriptive test-set comparisons, no independent confirmation of the best cell.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,default=[0,1])
    p.add_argument('--stage',choices=['plan','run','status','report'],default='run')
    args=p.parse_args();output=args.output.resolve()
    if args.stage=='status':
        print(json.dumps(status(output),indent=2));return
    if args.stage=='report':
        report(output);return
    jobs=jobs_for(args.configs)
    if args.stage=='plan':
        print(json.dumps(dict(runs=len(jobs),by_algorithm=dict(Counter(j['algorithm'] for j in jobs)),jobs=jobs),indent=2));return
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'driver.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    origin=provenance(ROOT)
    export=ROOT/'source-export.json'
    if export.exists():origin['source_export']=json.loads(export.read_text())
    config_hashes={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in args.configs}
    if (output/'plan.json').exists():
        if json.loads((output/'plan.json').read_text()) != jobs or json.loads((output/'provenance.json').read_text())['source_sha256'] != origin['source_sha256']:
            raise ValueError('Refusing changed plan/source on resume')
    else:
        write_json(output/'plan.json',jobs)
        write_json(output/'provenance.json',origin|dict(config_hashes=config_hashes))
    execute_jobs(jobs,output,args.gpus,origin)
    report(output,require_complete=True)
    (output/'COMPLETE').write_text(f'All {len(jobs)} paired sweep/baseline runs completed and audited\n')


if __name__=='__main__':
    main()
