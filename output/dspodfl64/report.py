"""Audit the DSpodFL matrix and its unchanged, cached comparison runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def report(root,code):
    plan=json.loads((root/'plan.json').read_text());cache=Path(plan['cached_comparators'])
    require=lambda ok,msg:None if ok else (_ for _ in ()).throw(RuntimeError(msg))
    h=hashlib.sha256()
    for path in sorted(list((code/'dasklearn').rglob('*.py'))+list((code/'scripts').glob('*.py'))):
        h.update(str(path.relative_to(code)).encode());h.update(path.read_bytes())
    require(h.hexdigest()==plan['source_sha256'],'Execution source changed')
    require(json.loads((cache/'audit.json').read_text())['passed'],'Cached comparison not audited')
    specs=[]
    for profile in plan['profiles']:
        require((root/profile/'COMPLETE').exists(),'Profile incomplete: '+profile)
        for steps in [5,1]:
            for seed in [0,1,2]:
                specs.append((root/profile/f'h{steps}-s{seed}',f'DSpodFL H={steps}',steps,False))
        for case,label in [('reference','Conflux V2 original'),('immediate-wide','Conflux V2 optimized'),('moshpit','Moshpit')]:
            specs.append((cache/profile/case,label,5,True))
    rows=[];curves=[];checks=[];fresh=[];capacities={};learning={};probabilities={};runtime=None
    common_fields=['dataset','model','seed','participants','alpha','partitioner','dataset_base_path',
        'model_size_bytes','evaluation_split','evaluation_examples','duration','test_period','test_method',
        'network_latency_ms','control_message_bytes','torch_device_name','torch_threads','churn',
        'gpu_residency','training_batching','training_batch_size','gpu_state_cache_mb']
    for folder,label,steps,cached in specs:
        require((folder/'COMPLETE').exists(),'Incomplete cohort')
        manifest=json.loads((folder/'manifest.json').read_text())
        provenance={k:manifest[k] for k in ['python','torch','numpy','dataset_provenance','split']}
        if runtime is None:runtime=provenance
        require(provenance==runtime,'Different dataset provenance or learning runtime')
        summary=pd.read_csv(folder/'summary.csv')
        for row in summary.to_dict('records'):
            profile,part,seed=row['profile'],row['partitioner'],int(row['seed'])
            run=folder/'runs'/f'{row["variant"]}-{profile}-{part}-s{seed}'
            state=json.loads((run/'settings.json').read_text())
            common={k:state[k] for k in common_fields}
            hyper=state['learning'].copy();require(hyper.pop('local_steps')==steps,'Unexpected local steps')
            key=(profile,part,seed)
            if key not in learning:learning[key]=(common,hyper)
            require(learning[key]==(common,hyper),'Different common settings')
            require(state['model_size_bytes']==346251 and state['participants']==64,'Wrong payload/node count')
            require(state['duration']==900000000 and state['test_period']==150000000,'Wrong time budget')
            stats=pd.read_csv(run/'client_statistics.csv').sort_values('client').reset_index(drop=True)
            capacity=stats[['client','training_speed','bandwidth']]
            ckey=(profile,seed)
            if ckey not in capacities:capacities[ckey]=capacity
            pd.testing.assert_frame_equal(capacity,capacities[ckey],check_exact=True)
            require(len(stats)==64 and stats.client.tolist()==list(range(64)),'Missing client statistics')
            require(stats.bytes_sent.sum()==stats.bytes_received.sum()==row['bytes_sent'],'Byte conservation failure')
            nodes=pd.read_csv(run/'accuracies.csv')
            require(len(nodes)==384 and not nodes.duplicated(['peer','time']).any(),'Wrong per-node coverage')
            require(nodes.peer.between(0,63).all() and np.isfinite(nodes[['accuracy','loss']]).all().all(),'Invalid node metrics')
            require(nodes.accuracy.between(0,100).all(),'Invalid accuracy')
            counts=nodes.groupby('time').peer.nunique()
            require(np.array_equal(counts.index,np.arange(150,901,150)) and counts.eq(64).all(),'Wrong checkpoint coverage')
            curve=nodes.groupby('time',as_index=False).agg(accuracy=('accuracy','mean'),loss=('loss','mean'),nodes=('peer','nunique'))
            network=pd.read_csv(run/'network-snapshots.csv')
            require(network.bytes_sent.is_monotonic_increasing and network.bytes_sent.iloc[-1]==row['bytes_sent'],'Traffic checkpoint mismatch')
            curve=curve.merge(network,on='time',validate='one_to_one')
            for name,value in [('variant',label),('profile',profile),('partitioner',part),('seed',seed)]:curve[name]=value
            curves.append(curve)
            require(abs(curve.accuracy.iloc[-1]-row['final_accuracy'])<1e-10,'Summary mismatch')
            if not cached:
                require(manifest['source_sha256']==plan['source_sha256'],'Mixed new execution source')
                require(state['degree']==4 and state['probability_mode']=='capacity','Wrong DSpodFL configuration')
                probability=pd.read_csv(run/'dspodfl_probabilities.csv')
                nodes_p=probability[probability.kind=='sgd'];links=probability[probability.kind=='link']
                require(len(nodes_p)==64 and len(links)==128,'Invalid physical graph')
                require(probability.probability.between(0,1).all() and np.allclose(links.weight,.2),'Invalid probabilities/weights')
                pkey=(profile,steps,seed)
                if pkey not in probabilities:probabilities[pkey]=probability
                pd.testing.assert_frame_equal(probability,probabilities[pkey],check_exact=True)
                iterations=pd.read_csv(run/'dspodfl_iterations.csv')
                require(not iterations.duplicated(['iteration','client']).any(),'Repeated update')
                require(iterations.time_us.between(0,900000000).all(),'Invalid completion time')
                row['completed_local_iterations']=len(iterations)
                row['sgd_steps_scheduled']=row['training_tasks']*steps
            else:
                row['completed_local_iterations']=None
                row['sgd_steps_scheduled']=row['training_tasks']*5
                if row['variant']=='conflux-v2':
                    latest=pd.read_csv(run/'latest-aggregate.csv')
                    for time,g in nodes.groupby('time'):
                        recipients=g[g['round']==g['round'].max()]
                        require(recipients.accuracy.nunique()==1 and recipients.loss.nunique()==1,'Inconsistent V2 aggregate')
                    latest=latest.assign(variant=label,profile=profile,partitioner=part,seed=seed);fresh.append(latest)
            row.update(protocol_variant=row['variant'],variant=label,cached=cached,local_steps=steps)
            rows.append(row)
            checks.append({'variant':label,'profile':profile,'partitioner':part,'seed':seed,'cached':cached,
                'per_node_rows':384,'finite':True,'byte_conservation':True,'run_directory':str(run),
                'source_sha256':manifest['source_sha256']})
    summary=pd.DataFrame(rows);curve=pd.concat(curves,ignore_index=True)
    require(len(summary)==60 and len(curve)==360,'Wrong total matrix size')
    require(len(summary[~summary.cached])==24,'Wrong new run count')
    summary.to_csv(root/'summary.csv',index=False);curve.to_csv(root/'curves.csv',index=False)
    pd.concat(fresh,ignore_index=True).to_csv(root/'latest-v2-aggregate.csv',index=False)
    comparison=summary.groupby(['profile','partitioner','variant']).agg(seeds=('seed','nunique'),
        accuracy_mean=('final_accuracy','mean'),accuracy_sd=('final_accuracy','std'),
        loss_mean=('final_loss','mean'),bytes_mean=('bytes_sent','mean'),
        training_tasks_mean=('training_tasks','mean'),sgd_steps_mean=('sgd_steps_scheduled','mean'),
        execution_s_mean=('actual_execution_s','mean')).reset_index()
    comparison.to_csv(root/'comparison.csv',index=False)
    reference=summary[summary.variant=='Conflux V2 optimized'][['profile','partitioner','seed','final_accuracy','bytes_sent']]
    paired=summary.merge(reference,on=['profile','partitioner','seed'],suffixes=('','_v2'),validate='many_to_one')
    paired['accuracy_difference_pp']=paired.final_accuracy-paired.final_accuracy_v2
    paired['traffic_ratio']=paired.bytes_sent/paired.bytes_sent_v2
    paired.to_csv(root/'paired-v2.csv',index=False)
    targets=[]
    for key,g in curve.groupby(['variant','profile','partitioner','seed']):
        for target in [20,30,40,50,60]:
            reached=g[g.accuracy>=target].sort_values('time')
            targets.append(dict(zip(['variant','profile','partitioner','seed'],key))|{
                'target_accuracy':target,'reached':bool(len(reached)),
                'time_s':None if reached.empty else float(reached.time.iloc[0]),
                'bytes_at_checkpoint':None if reached.empty else int(reached.bytes_sent.iloc[0]),'censor_time_s':900})
    pd.DataFrame(targets).to_csv(root/'time-to-accuracy.csv',index=False)
    audit={'passed':True,'new_runs':24,'cached_runs':36,'per_node_rows':23040,'new_per_node_rows':9216,
        'source_sha256':h.hexdigest(),'checks':checks,'same_capacities':True,
        'learning_exception':'H=1 explicitly changes local_steps only; H=5 matches the matrix learning settings.',
        'cached_source_compatibility':plan['cache_validation'],
        'limitations':plan['limitations']+['Three-seed common-setting screening, no test-set tuning.',
            'Physical bandwidth contention and causal elapsed time replace the paper normalized average-delay metric.',
            'GPU execution seconds are separate and cached timings come from previous runs.',
            'Minimum iteration opportunity is 20 ms, including otherwise empty iterations.',
            'DSpodFL has different local models; no shared aggregate or free global model is reported.']}
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    sys.path.insert(0,str(code/'scripts'))
    from report_baselines import plot
    plot(curve,root,64,3)
    for path in root.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plan.update(status='complete',audited_new_runs=24,audited_cached_runs=36)
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(comparison.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/data/mdevos/conflux/results/dspodfl64'))
    parser.add_argument('--code',type=Path,default=Path('/mnt/data/mdevos/conflux/dspodfl-code'))
    args=parser.parse_args();report(args.root,args.code)
