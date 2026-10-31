"""Audit all new DeMo runs, validation selection, and paired cached V2 runs."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

root=Path('/mnt/data/mdevos/conflux/results/demo64');plan=json.loads((root/'plan.json').read_text())
code=Path(plan['code_directory']);cache=Path(plan['cached_comparators'])
def require(value,message):
    if not value:raise RuntimeError(message)
h=hashlib.sha256()
for path in sorted(list((code/'dasklearn').rglob('*.py'))+list((code/'scripts').glob('*.py'))):
    h.update(str(path.relative_to(code)).encode());h.update(path.read_bytes())
require(h.hexdigest()==plan['source_sha256'],'Frozen execution source changed')
require(json.loads((cache/'audit.json').read_text())['passed'],'Cached V2 audit failed')
selection=json.loads((root/'selection.json').read_text());scores=pd.read_csv(root/'validation-scores.csv')
require(len(scores)==8,'Wrong validation score count')
for steps,g in scores.groupby('local_steps'):
    require(set(g.learning_rate)==set(plan['lr_candidates']),'Wrong tuning candidates')
    expected=float(g.sort_values(['mean_validation_accuracy','learning_rate'],ascending=[False,True]).iloc[0].learning_rate)
    require(expected==selection['selected'][str(steps)],'Selection differs from validation rule')

common_fields=['dataset','model','seed','participants','alpha','partitioner','dataset_base_path',
    'model_size_bytes','evaluation_examples','duration','test_period','test_method','network_latency_ms',
    'control_message_bytes','torch_device_name','torch_threads','churn','gpu_residency','training_batching',
    'training_batch_size','gpu_state_cache_mb']
checks=[];capacities={};common_settings={};runtime=None

def inspect(folder,row,cached=False,validation=False):
    global runtime
    profile,part,seed=row['profile'],row['partitioner'],int(row['seed'])
    manifest=json.loads((folder/'manifest.json').read_text())
    provenance={k:manifest[k] for k in ['python','torch','numpy','dataset_provenance','split']}
    if runtime is None:runtime=provenance
    require(provenance==runtime,'Runtime/dataset provenance mismatch')
    run=folder/'runs'/f'{row["variant"]}-{profile}-{part}-s{seed}'
    state=json.loads((run/'settings.json').read_text())
    require(state['evaluation_split']==('validation' if validation else 'test'),'Wrong evaluation split')
    require(state['participants']==64 and state['model_size_bytes']==346251,'Wrong nodes/model bytes')
    require(state['duration']==900000000 and state['test_period']==150000000,'Wrong simulated budget')
    require(state['learning']['momentum']==0 and state['learning']['weight_decay']==0 and state['learning']['batch_size']==32,'Different common learning settings')
    key=profile,part,seed
    common={k:state[k] for k in common_fields}
    if key not in common_settings:common_settings[key]=common
    require(common_settings[key]==common,'Different common settings')
    stats=pd.read_csv(run/'client_statistics.csv').sort_values('client').reset_index(drop=True)
    require(len(stats)==64 and stats.client.tolist()==list(range(64)),'Missing physical clients')
    require(stats.bytes_sent.sum()==stats.bytes_received.sum()==row['bytes_sent'],'Byte conservation failure')
    capacity=stats[['client','training_speed','bandwidth']]
    ckey=profile,seed
    if ckey not in capacities:capacities[ckey]=capacity
    pd.testing.assert_frame_equal(capacity,capacities[ckey],check_exact=True)
    nodes=pd.read_csv(run/'accuracies.csv')
    require(len(nodes)==384 and not nodes.duplicated(['time','peer']).any(),'Wrong per-node rows')
    require(nodes.peer.between(0,63).all() and np.isfinite(nodes[['accuracy','loss']]).all().all(),'Nonfinite metric')
    require(nodes.accuracy.between(0,100).all(),'Invalid accuracy')
    require(nodes.learning_rate.eq(state['learning']['learning_rate']).all() and row['learning_rate']==state['learning']['learning_rate'],'Learning-rate record mismatch')
    counts=nodes.groupby('time').peer.nunique()
    require(np.array_equal(counts.index,np.arange(150,901,150)) and counts.eq(64).all(),'Wrong checkpoint coverage')
    curve=nodes.groupby('time',as_index=False).agg(accuracy=('accuracy','mean'),loss=('loss','mean'),nodes=('peer','nunique'))
    network=pd.read_csv(run/'network-snapshots.csv')
    require(np.array_equal(network.time,np.arange(150,901,150)),'Wrong traffic checkpoint coverage')
    require(network.bytes_sent.is_monotonic_increasing and network.bytes_sent.iloc[-1]==row['bytes_sent'],'Traffic mismatch')
    curve=curve.merge(network,on='time',validate='one_to_one')
    require(abs(curve.accuracy.iloc[-1]-row['final_accuracy'])<1e-10,'Incorrect summary')
    if not cached:require(manifest['source_sha256']==plan['source_sha256'],'Mixed execution source')
    fresh=[]
    for time,g in nodes.groupby('time'):
        r=int(g['round'].max());recipients=g[g['round']==r]
        require(recipients.accuracy.nunique()==1 and recipients.loss.nunique()==1,'Different metrics for same shared round')
        fresh.append({'time':time,'round':r,'accuracy':float(recipients.accuracy.iloc[0]),
            'loss':float(recipients.loss.iloc[0]),'recipients':len(recipients),
            'mean_staleness_rounds':float((r-g['round']).mean()),'max_staleness_rounds':int((r-g['round']).max())})
    if row['variant']=='demo':
        require(state['compression_decay']==.999 and state['compression_chunk']==64 and state['compression_topk']==32,'Wrong compression')
        rounds=pd.read_csv(run/'demo-rounds.csv')
        require(not rounds.duplicated(['peer','round']).any() and rounds.time.between(0,900).all(),'Invalid completed rounds')
        require(state['learning']['local_steps'] in [1,5],'Wrong accumulation')
        if not validation:require(state['learning']['learning_rate']==selection['selected'][str(state['learning']['local_steps'])],'Wrong selected LR')
        row['completed_local_iterations']=len(rounds)
        row['sparse_packet_bytes']=20472
    else:
        require(state['learning']['learning_rate']==.01 and state['learning']['local_steps']==5,'Wrong V2 learning')
        require(state['sample_size']==state['chunks_in_sample']==state['relay_count']==16 and state['immediate_uploads'],'Wrong optimized V2 configuration')
        row['completed_local_iterations']=None;row['sparse_packet_bytes']=None
    row['local_steps']=state['learning']['local_steps']
    row['microbatches_scheduled']=row['training_tasks']*row['local_steps']
    checks.append({'run_directory':str(run),'variant':row['variant'],'profile':profile,'partitioner':part,
        'seed':seed,'validation':validation,'cached':cached,'per_node_rows':384,
        'source_sha256':manifest['source_sha256'],'finite':True,'byte_conservation':True})
    return curve,pd.DataFrame(fresh),nodes,stats,network

# Audit validation data and score aggregation before looking at test comparisons.
for steps in [5,1]:
    for rate in plan['lr_candidates']:
        folder=root/'validation'/f'h{steps}-lr{rate:g}'
        require((folder/'COMPLETE').exists(),'Incomplete validation cohort')
        rows=pd.read_csv(folder/'summary.csv');require(len(rows)==2,'Missing validation partition')
        for row in rows.to_dict('records'):inspect(folder,row,validation=True)
        recorded=scores[(scores.local_steps==steps)&(scores.learning_rate==rate)].iloc[0]
        require(abs(rows.final_accuracy.mean()-recorded.mean_validation_accuracy)<1e-10,'Tuning score mismatch')

# Replays compare every saved metric and physical capacity/traffic statistic.
replays=[]
for profile in plan['profiles']:
    folder=root/profile/'v2-replay';require((folder/'COMPLETE').exists(),'Missing V2 replay')
    for row in pd.read_csv(folder/'summary.csv').to_dict('records'):
        _,_,nodes,stats,network=inspect(folder,row)
        old=cache/profile/'immediate-wide'/'runs'/f'conflux-v2-{profile}-{row["partitioner"]}-s0'
        for current,name,sort in [(nodes,'accuracies.csv',['time','peer']),(stats,'client_statistics.csv',['client']),
                                  (network,'network-snapshots.csv',['time'])]:
            previous=pd.read_csv(old/name)
            pd.testing.assert_frame_equal(current.sort_values(sort).reset_index(drop=True),previous.sort_values(sort).reset_index(drop=True),check_exact=True)
        replays.append({'profile':profile,'partitioner':row['partitioner'],'seed':0,'exact':True,'per_node_rows':384})

rows=[];curves=[];latest=[]
for profile in plan['profiles']:
    require((root/profile/'COMPLETE').exists(),'Incomplete profile')
    specs=[(root/profile/f'h{steps}-s{seed}',f'DeMo H={steps}',False) for seed in [0,1,2] for steps in [5,1]]
    specs.append((cache/profile/'immediate-wide','Conflux V2 optimized',True))
    for folder,label,cached in specs:
        require((folder/'COMPLETE').exists(),'Incomplete cohort')
        for row in pd.read_csv(folder/'summary.csv').to_dict('records'):
            curve,fresh,_,_,_=inspect(folder,row,cached=cached)
            for frame in [curve,fresh]:
                for name,value in [('variant',label),('profile',profile),('partitioner',row['partitioner']),('seed',int(row['seed']))]:frame[name]=value
            curves.append(curve);latest.append(fresh)
            row.update(protocol_variant=row['variant'],variant=label,cached=cached);rows.append(row)
summary=pd.DataFrame(rows);curve=pd.concat(curves,ignore_index=True);fresh=pd.concat(latest,ignore_index=True)
require(len(summary)==36 and len(curve)==216,'Wrong comparison matrix size')
require(not summary.duplicated(['variant','profile','partitioner','seed']).any(),'Duplicate matrix case')
require(summary.groupby(['variant','profile','partitioner']).seed.apply(lambda x:set(x)=={0,1,2}).all(),'Wrong seeds')
summary.to_csv(root/'summary.csv',index=False);curve.to_csv(root/'curves.csv',index=False)
fresh.to_csv(root/'latest-aggregate.csv',index=False)
comparison=summary.groupby(['profile','partitioner','variant']).agg(seeds=('seed','nunique'),
    accuracy_mean=('final_accuracy','mean'),accuracy_sd=('final_accuracy','std'),loss_mean=('final_loss','mean'),
    bytes_mean=('bytes_sent','mean'),training_tasks_mean=('training_tasks','mean'),
    microbatches_mean=('microbatches_scheduled','mean'),execution_s_mean=('actual_execution_s','mean')).reset_index()
comparison.to_csv(root/'comparison.csv',index=False)
v2=summary[summary.variant=='Conflux V2 optimized'][['profile','partitioner','seed','final_accuracy','bytes_sent']]
paired=summary.merge(v2,on=['profile','partitioner','seed'],suffixes=('','_v2'),validate='many_to_one')
paired['accuracy_difference_pp']=paired.final_accuracy-paired.final_accuracy_v2
paired['traffic_ratio']=paired.bytes_sent/paired.bytes_sent_v2;paired.to_csv(root/'paired-v2.csv',index=False)
targets=[]
for key,g in curve.groupby(['variant','profile','partitioner','seed']):
    for target in [20,30,40,50,60]:
        reached=g[g.accuracy>=target].sort_values('time')
        targets.append(dict(zip(['variant','profile','partitioner','seed'],key))|{'target_accuracy':target,
            'reached':bool(len(reached)),'time_s':None if reached.empty else float(reached.time.iloc[0]),
            'bytes_at_checkpoint':None if reached.empty else int(reached.bytes_sent.iloc[0]),'censor_time_s':900})
pd.DataFrame(targets).to_csv(root/'time-to-accuracy.csv',index=False)
(root/'audit.json').write_text(json.dumps({'passed':True,'source_sha256':h.hexdigest(),'test_runs':24,
    'cached_v2_runs':12,'v2_replays':replays,'validation_runs':16,'per_node_rows':len(checks)*384,
    'same_capacities':True,'checks':checks,'cache_validation':plan['cache_validation'],
    'readout':'All-node mean stored completed model; latest shared aggregate and staleness reported separately for both protocols.',
    'learning_exception':'DeMo validation-selected Signum LR; H is gradient accumulation, V2 H=5 is local SGD.',
    'limitations':plan['limitations']+['Three training/capacity seeds; short CIFAR screen, not universal ranking.',
        'Targets use first observed fixed checkpoint; unreached targets are right-censored at 900s.',
        'Execution timings include real training/evaluation; cached V2 timings are from prior jobs.']},indent=2)+'\n')
sys.path.insert(0,str(code/'scripts'))
from report_baselines import plot
plot(curve,root,64,3)
# Explicitly distinguish the newest V2 aggregate from its network-wide mean.
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
for ax,((profile,part),data) in zip(axes.flat,curve.groupby(['profile','partitioner'],sort=True)):
    for label,g in data.groupby('variant',sort=False):
        values=g.groupby('time').accuracy
        mean,sd=values.mean(),values.std()
        ax.plot(mean.index,mean,label=label+' — all nodes')
        ax.fill_between(mean.index,mean-sd,mean+sd,alpha=.12)
    v=fresh[(fresh.profile==profile)&(fresh.partitioner==part)&(fresh.variant=='Conflux V2 optimized')].groupby('time').accuracy.mean()
    ax.plot(v.index,v,linestyle='--',color='black',label='V2 latest aggregate (distinct metric)')
    ax.set(title=f'{profile}, {part}',xlabel='Simulated protocol time [s]',ylabel='Heldout test accuracy [%]');ax.grid(alpha=.25)
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9)
fig.suptitle('Network-wide checkpoints and the distinct freshest V2 aggregate')
fig.tight_layout(rect=(0,.09,1,.96));fig.savefig(root/'aggregate-readout.png',dpi=180);fig.savefig(root/'aggregate-readout.svg');plt.close(fig)
for path in root.glob('*.svg'):path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
plan.update(status='complete',audited_new_test_runs=24,audited_validation_runs=16,audited_cached_runs=12,audited_v2_replays=4)
(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(root/'COMPLETE').write_text('All experiments audited\n')
print(comparison.to_string(index=False))
