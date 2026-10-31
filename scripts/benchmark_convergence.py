"""Run a preregistered, resumable two-GPU convergence study via shared configs.

Each training run is a new main.py process with a normal version-1 configuration.
Tuning uses validation curves and a disjoint seed. Confirmation has paired seeds,
all four capacity profiles, fixed validation checkpoints and full final test sets.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dasklearn.experiment_runner import provenance, write_json


def csv_write(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def settings_for(study, workload, algorithm, profile, partitioner, seed, lr, data):
    settings = dict(algorithm=algorithm, dataset=workload['dataset'], model=workload['model'],
        device=study.get('device','cuda:0'), peers=study['peers'], seed=seed, partitioner=partitioner, alpha=study['alpha'],
        stop='duration', duration=workload['duration'], rounds=0, test_method='snapshot',
        test_interval=0, test_period=workload['test_period'], evaluation_split='validation',
        evaluation_examples=workload['evaluation_examples'], learning_rate=lr, momentum=0,
        local_steps=study['local_steps'], batch_size=study['batch_size'], torch_threads=2,
        capacity_profile=profile, bandwidth_bytes_per_s=workload['bandwidth_bytes_per_s'],
        compute_step_ms=workload['compute_step_ms'], model_size_bytes=workload.get('model_size_bytes'), network_latency_ms=20,
        sample_size=study['sample_size'], chunks_in_sample=study['chunks'],
        relay_count=study['peers'], immediate_uploads=True, success_fraction=1,
        group_size=study['sample_size'], averaging_steps=1, k=study['chunks'] if algorithm=='shatter' else 3,
        r=2, topology='kreg', qgm_topology='kreg', qgm_degree=3,
        qgm_beta=.9, qgm_mu=.9, training_batch_size=2, log_level='WARNING',
        final_heldout=seed != study['tuning_seed'])
    if workload['dataset'] != 'digits':
        settings['dataset_base_path'] = str(data / workload['dataset'])
    # QG momentum is the published one-local-step method, with its own beta.
    if algorithm == 'qgm':
        settings['local_steps'] = 1
    return settings


def run_name(stage, workload, algorithm, profile, partitioner, seed, lr):
    return f'{stage}-{workload["name"]}-{algorithm}-{profile}-{partitioner}-s{seed}-lr{lr:g}'


def run_metrics(directory, peers, duration):
    import pandas as pd
    artifacts = list(directory.glob('data/*/accuracies.csv'))
    if len(artifacts) != 1:
        raise ValueError(f'Expected one accuracy file: {directory}')
    frame = pd.read_csv(artifacts[0])
    curve = frame.groupby('time', as_index=False).agg(accuracy=('accuracy','mean'), loss=('loss','mean'), nodes=('peer','nunique'))
    if not curve.nodes.eq(peers).all() or curve.time.iloc[-1] != duration:
        raise ValueError('Incomplete all-node snapshots or missing final checkpoint')
    import numpy as np
    if not np.isfinite(curve[['accuracy','loss']].to_numpy()).all():
        raise ValueError('Nonfinite training metrics')
    curve.to_csv(directory/'curve.csv', index=False)
    # Exclude the unobserved interval before the first scheduled snapshot.
    area = float(np.trapezoid(curve.accuracy, curve.time) / (duration - curve.time.iloc[0]))
    result = dict(final_accuracy=float(curve.accuracy.iloc[-1]), final_loss=float(curve.loss.iloc[-1]), validation_auc=area)
    heldout = sorted(artifacts[0].parent.glob('heldout_[0-9]*.csv'))
    if heldout:
        test = pd.concat([pd.read_csv(path) for path in heldout])
        if len(test) != peers or test.peer.nunique() != peers:
            raise ValueError('Incomplete final test evaluation')
        result.update(test_accuracy=float(test.accuracy.mean()), test_loss=float(test.loss.mean()))
    saved = json.loads((artifacts[0].parent/'settings.json').read_text())
    if saved['algorithm']=='conflux-v2':
        latest=[]
        for checkpoint,group in frame.groupby('time'):
            round_nr=int(group['round'].max())
            recipients=group[group['round']==round_nr]
            if recipients.accuracy.nunique()!=1 or recipients.loss.nunique()!=1:
                raise ValueError('V2 aggregate recipients disagree')
            latest.append(dict(time=checkpoint,round=round_nr,accuracy=float(recipients.accuracy.iloc[0]),
                loss=float(recipients.loss.iloc[0]),recipients=len(recipients),
                mean_staleness_rounds=float((round_nr-group['round']).mean())))
        pd.DataFrame(latest).to_csv(directory/'latest-aggregate.csv',index=False)
    result['model_bytes'] = saved.get('model_size_bytes')
    # The true serialized payload is recorded by the simulator's output, not an override.
    stats = artifacts[0].parent/'execution-stats.json'
    if stats.exists():
        result.update(json.loads(stats.read_text()))
    return result


def execute_job(job, output, gpu, origin):
    directory = output/'runs'/job['name']
    record_path = directory/'record.json'
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if old['settings'] != job['settings'] or old['source_sha256'] != origin['source_sha256']:
            raise ValueError(f'Resume source/config mismatch: {directory}')
        if old['status'] == 'completed':
            return old
        # Do not overwrite the evidence from a failed/interrupted attempt.
        attempt = 1
        while directory.with_name(directory.name+f'.attempt{attempt}').exists():
            attempt += 1
        directory.rename(directory.with_name(directory.name+f'.attempt{attempt}'))
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(version=1, name=job['name'], execution='local', output=str(directory), settings=job['settings'])
    write_json(directory/'config.json', config)
    record = dict(**job, status='running', source_sha256=origin['source_sha256'], gpu=gpu, started_at=time.time())
    write_json(record_path, record)
    print(json.dumps(dict(starting=job['name'],gpu=gpu)), flush=True)
    started=time.monotonic()
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    with (directory/'run.log').open('w') as stream:
        process=subprocess.run([sys.executable,str(ROOT/'main.py'),'--config',str(directory/'config.json')],
                               cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    record.update(returncode=process.returncode,wall_seconds=time.monotonic()-started,finished_at=time.time())
    record['status']='completed' if process.returncode==0 else 'failed'
    if process.returncode==0:
        try:
            record['metrics']=run_metrics(directory,job['settings']['peers'],job['settings']['duration'])
            if job['settings']['final_heldout'] and 'test_accuracy' not in record['metrics']:
                raise ValueError('Missing heldout metrics')
        except Exception as exc:
            record.update(status='failed',error=str(exc))
    write_json(record_path,record)
    print(json.dumps(dict(finished=job['name'],status=record['status'],wall_seconds=record['wall_seconds'],metrics=record.get('metrics'))),flush=True)
    if record['status']!='completed':
        raise RuntimeError(f'Run failed: {directory}; {record.get("error", "see run.log")}')
    return record


def execute_jobs(jobs, output, gpus, origin):
    work=queue.Queue()
    for job in jobs:
        work.put(job)
    def worker(gpu):
        while True:
            try:
                job=work.get_nowait()
            except queue.Empty:
                break
            execute_job(job,output,gpu,origin)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in gpus]
        for future in futures:
            future.result()


def build_jobs(study, data, selection=None, pilot=False):
    jobs=[]
    for workload in study['workloads']:
        stage='tune' if selection is None else 'confirm'
        profiles=[study['tuning_profile']] if selection is None else study['profiles']
        seeds=[study['tuning_seed']] if selection is None else study['seeds']
        for algorithm,partitioner,profile,seed in itertools.product(study['algorithms'],study['partitioners'],profiles,seeds):
            key=f'{workload["name"]}/{algorithm}/{partitioner}'
            rates=workload['learning_rates'] if selection is None else [selection[key]['learning_rate']]
            if pilot:
                rates=rates[:1]
            for lr in rates:
                settings=settings_for(study,workload,algorithm,profile,partitioner,seed,lr,data)
                if pilot:
                    settings.update(duration=workload['test_period']*2,test_period=workload['test_period'])
                    settings['final_heldout']=False
                name=run_name('pilot' if pilot else stage,workload,algorithm,profile,partitioner,seed,lr)
                jobs.append(dict(name=name,stage='pilot' if pilot else stage,workload=workload['name'],algorithm=algorithm,
                                 partitioner=partitioner,profile=profile,seed=seed,learning_rate=lr,settings=settings))
    # Interleave workloads to expose integration failures early.
    return sorted(jobs,key=lambda job:(job['seed'],job['partitioner'],job['algorithm'],job['workload'],job['profile'],job['learning_rate']))


def select(output):
    groups={}
    for path in (output/'runs').glob('tune-*/record.json'):
        record=json.loads(path.read_text())
        if path.parent.name!=record['name']:
            continue
        if record['status']!='completed':
            continue
        key=f'{record["workload"]}/{record["algorithm"]}/{record["partitioner"]}'
        groups.setdefault(key,[]).append(record)
    selection={key:dict(learning_rate=(best:=max(runs,key=lambda r:(r['metrics']['validation_auc'],-r['learning_rate'])))['learning_rate'],
                        validation_auc=best['metrics']['validation_auc'],run=best['name']) for key,runs in groups.items()}
    write_json(output/'selection.json',selection)
    return selection


def report(output):
    import numpy as np
    import pandas as pd
    records=[]
    for path in sorted((output/'runs').glob('*/record.json')):
        record=json.loads(path.read_text())
        if path.parent.name!=record['name']:
            continue
        if record['status']=='completed':
            records.append(record)
    rows=[{key:record[key] for key in ('name','stage','workload','algorithm','profile','partitioner','seed','learning_rate','wall_seconds')} |
          record['metrics'] for record in records]
    csv_write(output/'summary.csv',rows)
    study=json.loads((output/'study.json').read_text())
    workloads={w['name']:w for w in study['workloads']}
    tta=[]
    curves=[]
    for record in records:
        if record['stage']!='confirm':
            continue
        frame=pd.read_csv(output/'runs'/record['name']/'curve.csv')
        for key in ('workload','algorithm','profile','partitioner','seed'):
            frame[key]=record[key]
        curves.append(frame)
        for target in workloads[record['workload']]['targets']:
            hit=frame.loc[frame.accuracy>=target,'time']
            upper=float(hit.iloc[0]) if len(hit) else None
            lower=max(0,upper-record['settings']['test_period']) if upper is not None else record['settings']['duration']
            tta.append({key:record[key] for key in ('workload','algorithm','profile','partitioner','seed')} |
                       dict(target=target,reached=upper is not None,lower_seconds=lower,upper_seconds=upper,
                            restricted_seconds=upper if upper is not None else record['settings']['duration']))
    csv_write(output/'time-to-accuracy.csv',tta)
    comparisons=[]
    if tta:
        frame=pd.DataFrame(tta)
        keys=['workload','profile','partitioner','seed','target']
        v2=frame[frame.algorithm=='conflux-v2'].set_index(keys)
        for _,baseline in frame[frame.algorithm!='conflux-v2'].iterrows():
            key=tuple(baseline[k] for k in keys)
            if key not in v2.index:
                continue
            conflux=v2.loc[key]
            # Conservative first-passage comparison uses checkpoint intervals.
            win=bool(conflux.reached and conflux.upper_seconds < baseline.lower_seconds)
            loss=bool(baseline.reached and baseline.upper_seconds < conflux.lower_seconds)
            comparisons.append({k:baseline[k] for k in keys}|dict(baseline=baseline.algorithm,
                outcome='v2-faster' if win else 'baseline-faster' if loss else 'unresolved',
                v2_reached=bool(conflux.reached),baseline_reached=bool(baseline.reached),
                speedup=float(baseline.upper_seconds/conflux.upper_seconds) if baseline.reached and conflux.reached else None))
    csv_write(output/'paired-comparisons.csv',comparisons)
    lines=['# Conflux V2 expanded convergence study','',
        'Times are simulated elapsed seconds with charged network/compute costs. GPU execution wall seconds are separate.',
        'All-node validation snapshots include idle nodes. Final test scores use all final node checkpoints and the full test split.',
        f'Learning rates are selected per workload/algorithm/partitioner by validation accuracy area on tuning seed {study["tuning_seed"]} under {study["tuning_profile"]} capacities.',
        f'Confirmation seeds: {study["seeds"]}. Network and compute assignments use independent RNG streams and preserve arithmetic mean capacities.',
        'Unreached thresholds are right-censored. A strict win requires nonoverlapping first-crossing checkpoint intervals.',
        '',f'Completed runs: {len(records)}. Confirmation runs: {sum(r["stage"]=="confirm" for r in records)}.','']
    if comparisons:
        counts=pd.DataFrame(comparisons).groupby(['baseline','outcome']).size().unstack(fill_value=0)
        lines+=['| Baseline | V2 faster | Baseline faster | Unresolved |','|---|---:|---:|---:|']
        for baseline,row in counts.iterrows():
            lines.append(f'| {baseline} | {row.get("v2-faster",0)} | {row.get("baseline-faster",0)} | {row.get("unresolved",0)} |')
        lines+=['','Counts include multiple correlated targets; they are descriptive, not independent statistical tests.','']
        aggregate=[]
        paired=pd.DataFrame(comparisons)
        for key,group in paired.groupby(['workload','profile','partitioner','baseline','target']):
            values=group.speedup.dropna().to_numpy()
            estimate=dict(zip(['workload','profile','partitioner','baseline','target'],key))
            estimate.update(paired_seeds=len(group),both_reached=len(values))
            if len(values)==len(study['seeds']):
                rng=np.random.default_rng(1729)
                samples=np.median(rng.choice(values,(5000,len(values))),axis=1)
                estimate.update(median_speedup=float(np.median(values)),ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)))
            aggregate.append(estimate)
        csv_write(output/'speedup-summary.csv',aggregate)
        lines+=['Paired speedups and 95% seed-bootstrap intervals are in speedup-summary.csv, only where all confirmation seeds reached both targets.',
                'With three seeds, these intervals are coarse. Both censored and overlapping intervals remain unresolved.','']
    if curves:
        combined=pd.concat(curves,ignore_index=True)
        combined.to_csv(output/'curves.csv',index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for workload,data in combined.groupby('workload'):
            fig,axes=plt.subplots(2,len(study['profiles']),figsize=(5*len(study['profiles']),7),squeeze=False)
            for row,partitioner in enumerate(study['partitioners']):
                for column,profile in enumerate(study['profiles']):
                    ax=axes[row,column]
                    for algorithm,group in data[(data.partitioner==partitioner)&(data.profile==profile)].groupby('algorithm'):
                        stats=group.groupby('time').accuracy.agg(['mean','std']).fillna(0)
                        ax.plot(stats.index,stats['mean'],label=algorithm)
                        ax.fill_between(stats.index,stats['mean']-stats['std'],stats['mean']+stats['std'],alpha=.12)
                    ax.set(title=f'{profile}, {partitioner}',xlabel='Simulated elapsed seconds',ylabel='All-node validation accuracy (%)')
                    ax.grid(alpha=.2);ax.legend(fontsize=7)
            fig.suptitle(workload);fig.tight_layout();fig.savefig(output/f'{workload}.png',dpi=130);plt.close(fig)
    lines+=['Artifacts: study.json, provenance.json, selection.json, per-run config.json/record.json/run.log, raw DAGs and node metrics.',
            'Raw checkpoints/DAGs remain on sacs006 under /mnt/data/mdevos. Historical experiments were not reused.','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'experiments/conflux-v2/study.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1])
    parser.add_argument('--stage',choices=['all','pilot','tune','confirm','report','plan','status'],default='all')
    args=parser.parse_args();args.output=args.output.resolve();args.data=args.data.resolve()
    study=json.loads(args.study.read_text())
    if args.stage=='status':
        counts={};running=[];failed=[]
        for path in sorted((args.output/'runs').glob('*/record.json')):
            record=json.loads(path.read_text())
            if path.parent.name!=record['name']:
                continue
            key=record['stage']+'/'+record['status']
            counts[key]=counts.get(key,0)+1
            if record['status']=='running':
                running.append(dict(name=record['name'],gpu=record['gpu'],elapsed_seconds=round(time.time()-record['started_at'])))
            elif record['status']=='failed':
                failed.append(record['name'])
        print(json.dumps(dict(counts=counts,running=running,failed=failed,complete=(args.output/'COMPLETE').exists()),indent=2))
        return
    if args.stage=='plan':
        tuning=build_jobs(study,args.data)
        confirmation=len(study['workloads'])*len(study['algorithms'])*len(study['profiles'])*len(study['partitioners'])*len(study['seeds'])
        print(json.dumps(dict(tuning_runs=len(tuning),confirmation_runs=confirmation,study=study),indent=2));return
    if args.stage=='report':
        report(args.output);return
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.output/'driver.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError('Study is already running; use --stage status or report') from exc
    lock.seek(0);lock.truncate();lock.write(str(os.getpid())+'\n');lock.flush()
    origin=provenance(ROOT)
    exported=ROOT/'source-export.json'
    if exported.exists():
        origin['source_export']=json.loads(exported.read_text())
    path=args.output/'provenance.json'
    if path.exists():
        if json.loads(path.read_text())['source_sha256'] != origin['source_sha256'] or json.loads((args.output/'study.json').read_text())!=study:
            raise ValueError('Refusing to change source/study while resuming')
    else:
        write_json(path,origin);write_json(args.output/'study.json',study)
    if args.stage=='pilot':
        execute_jobs(build_jobs(study,args.data,pilot=True),args.output,args.gpus,origin);report(args.output);return
    if args.stage in {'all','tune'}:
        execute_jobs(build_jobs(study,args.data),args.output,args.gpus,origin)
        selection=select(args.output)
    else:
        selection=json.loads((args.output/'selection.json').read_text())
    if args.stage in {'all','confirm'}:
        execute_jobs(build_jobs(study,args.data,selection),args.output,args.gpus,origin)
    report(args.output)
    if args.stage in {'all','confirm'}:
        (args.output/'COMPLETE').write_text('All requested tuning and confirmation runs completed\n')


if __name__=='__main__':
    main()
