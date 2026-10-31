"""Audit raw server results, verify old baselines, and export compact figures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CASES = ['reference', 'immediate', 'wide', 'immediate-wide', 'moshpit']
LABELS = {'reference':'V2 original', 'immediate':'V2 immediate uploads',
          'wide':'V2 16 reducers', 'immediate-wide':'V2 both changes', 'moshpit':'Moshpit (one phase)'}
COMMON_SETTINGS = ['dataset','model','learning','seed','participants','partitioner','alpha',
    'dataset_base_path','torch_device_name','model_size_bytes','duration','test_period','test_method',
    'test_interval','evaluation_split','evaluation_examples','network_latency_ms','control_message_bytes',
    'torch_threads','churn','gpu_residency','training_batching','training_batch_size','gpu_state_cache_mb']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit(root, old, code):
    plan = json.loads((root/'plan.json').read_text())
    h = hashlib.sha256()
    for path in sorted(list((code/'dasklearn').rglob('*.py'))+list((code/'scripts').glob('*.py'))):
        h.update(str(path.relative_to(code)).encode()); h.update(path.read_bytes())
    require(h.hexdigest() == plan['source_sha256'], 'Frozen execution code changed')
    summaries, curves, fresh, rounds, checks, replays = [], [], [], [], [], []
    reference_settings, reference_capacities, common_manifest = {}, {}, None
    for profile in plan['profiles']:
        require((root/profile/'COMPLETE').exists(), 'Incomplete profile: '+profile)
        for case in CASES:
            folder = root/profile/case
            require((folder/'COMPLETE').exists(), 'Incomplete case: '+str(folder))
            manifest = json.loads((folder/'manifest.json').read_text())
            variant = 'moshpit' if case == 'moshpit' else 'conflux-v2'
            require(manifest['source_sha256'] == plan['source_sha256'], 'Mixed protocol sources')
            require(manifest['variants'] == [variant], 'Unexpected method')
            require(manifest['profiles'] == [profile] and manifest['seeds'] == [0,1,2], 'Unexpected run axes')
            require(manifest['partitioners'] == ['iid','dirichlet'], 'Unexpected partitioners')
            expected_chunks = 16 if case in ['wide','immediate-wide'] else 8
            require(manifest['chunks'] == expected_chunks and manifest['relay_count'] == expected_chunks, 'Wrong placement')
            require(manifest['immediate_uploads'] == (case in ['immediate','immediate-wide']), 'Wrong upload schedule')
            excluded = {'output','profiles','variants','chunks','relay_count','immediate_uploads','cuda_visible_devices'}
            common = {k:v for k,v in manifest.items() if k not in excluded}
            if common_manifest is None: common_manifest = common
            require(common == common_manifest, 'Different common manifest settings')
            summary = pd.read_csv(folder/'summary.csv')
            require(len(summary)==6 and not summary.duplicated(['partitioner','seed']).any(), 'Missing/duplicate runs')
            curve = pd.read_csv(folder/'curves.csv')
            require(len(curve)==36, 'Wrong aggregate checkpoint count')
            for row in summary.to_dict('records'):
                partitioner, seed = row['partitioner'], int(row['seed'])
                key = (profile, partitioner, seed)
                run = folder/'runs'/f'{variant}-{profile}-{partitioner}-s{seed}'
                state = json.loads((run/'settings.json').read_text())
                common_state = {k:state[k] for k in COMMON_SETTINGS}
                if key not in reference_settings: reference_settings[key] = common_state
                require(common_state == reference_settings[key], 'Different learning/evaluation settings')
                stats = pd.read_csv(run/'client_statistics.csv').sort_values('client').reset_index(drop=True)
                capacity = stats[['client','training_speed','bandwidth']]
                capacity_key = (profile, seed)
                if capacity_key not in reference_capacities: reference_capacities[capacity_key] = capacity
                pd.testing.assert_frame_equal(capacity, reference_capacities[capacity_key],check_exact=True)
                require(len(stats)==64 and stats.client.tolist()==list(range(64)), 'Missing clients')
                require(int(stats.bytes_sent.sum())==int(stats.bytes_received.sum())==int(row['bytes_sent']), 'Unbalanced bytes')
                nodes = pd.read_csv(run/'accuracies.csv')
                expected_times = np.arange(150,901,150)
                require(len(nodes)==384 and not nodes.duplicated(['peer','time']).any(), 'Missing/duplicate node checkpoint')
                counts = nodes.groupby('time').peer.nunique()
                require(np.array_equal(counts.index, expected_times) and counts.eq(64).all(), 'Wrong checkpoint coverage')
                require(np.isfinite(nodes[['accuracy','loss']]).all().all(), 'Nonfinite node metrics')
                require(nodes.accuracy.between(0,100).all(), 'Invalid accuracy')
                network = pd.read_csv(run/'network-snapshots.csv')
                require(np.array_equal(network.time,expected_times) and network.bytes_sent.is_monotonic_increasing, 'Invalid traffic curve')
                require(int(network.bytes_sent.iloc[-1])==int(row['bytes_sent']), 'Final traffic mismatch')
                measured = nodes.groupby('time',as_index=False).agg(accuracy=('accuracy','mean'),loss=('loss','mean'),nodes=('peer','nunique'))
                saved = curve[(curve.partitioner==partitioner)&(curve.seed==seed)].sort_values('time')
                np.testing.assert_allclose(measured[['time','accuracy','loss','nodes']],saved[['time','accuracy','loss','nodes']],rtol=1e-13,atol=1e-13)
                measured = measured.merge(network,on='time',validate='one_to_one')
                for name,value in [('case',case),('profile',profile),('partitioner',partitioner),('seed',seed)]: measured[name]=value
                curves.append(measured)
                if case != 'moshpit':
                    latest = pd.read_csv(run/'latest-aggregate.csv')
                    require(len(latest)==6, 'Missing aggregate readout')
                    for index, (time,g) in enumerate(nodes.groupby('time')):
                        r = int(g['round'].max()); recipients = g[g['round']==r]
                        require(recipients.accuracy.nunique()==1 and recipients.loss.nunique()==1, 'Inconsistent aggregate recipients')
                        got = latest.iloc[index]
                        np.testing.assert_allclose([got.time,got['round'],got.accuracy,got.loss,got.recipients,got.mean_staleness_rounds,got.max_staleness_rounds],
                            [time,r,recipients.accuracy.iloc[0],recipients.loss.iloc[0],len(recipients),(r-g['round']).mean(),(r-g['round']).max()],rtol=1e-13,atol=1e-13)
                    latest = latest.assign(case=case,profile=profile,partitioner=partitioner,seed=seed)
                    fresh.append(latest)
                    completions = pd.read_csv(run/'conflux_v2_rounds.csv')
                    counts_by_round = completions.groupby('round').client.nunique()
                    first_completion = completions.groupby('round').time_us.min()/1e6
                    # At the time horizon, the last aggregate may have reached only some recipients.
                    rounds.append({'case':case,'profile':profile,'partitioner':partitioner,'seed':seed,
                        'latest_completed_round':int(completions['round'].max()),
                        'fully_disseminated_rounds':int(counts_by_round.eq(16).sum()),
                        'mean_first_completion_interval_s':float(first_completion.diff().dropna().mean()),
                        'first_complete_aggregate_s':float(first_completion.iloc[0])})
                if case in ['reference','moshpit']:
                    old_run = old/profile/'runs'/f'{variant}-{profile}-{partitioner}-s{seed}'
                    previous = pd.read_csv(old_run/'accuracies.csv').sort_values(['time','peer']).reset_index(drop=True)
                    current = nodes.sort_values(['time','peer']).reset_index(drop=True)
                    pd.testing.assert_frame_equal(current,previous,check_exact=True)
                    pd.testing.assert_frame_equal(stats,pd.read_csv(old_run/'client_statistics.csv').sort_values('client').reset_index(drop=True),check_exact=True)
                    pd.testing.assert_frame_equal(network,pd.read_csv(old_run/'network-snapshots.csv'),check_exact=True)
                    replays.append({'case':case,'profile':profile,'partitioner':partitioner,'seed':seed,'per_node_rows_exact':384,'traffic_and_capacities_exact':True})
                checks.append({'case':case,'profile':profile,'partitioner':partitioner,'seed':seed,'node_rows':384,'finite':True,'byte_conservation':True})
            summary['case'] = case
            summaries.append(summary)
    summary = pd.concat(summaries,ignore_index=True)
    curves = pd.concat(curves,ignore_index=True)
    require(len(summary)==60 and len(curves)==360, 'Incomplete study')
    summary.to_csv(root/'summary.csv',index=False);curves.to_csv(root/'curves.csv',index=False)
    pd.concat(fresh,ignore_index=True).to_csv(root/'latest-v2-aggregate.csv',index=False)
    pd.DataFrame(rounds).to_csv(root/'round-progress.csv',index=False)
    comparison = summary.groupby(['profile','partitioner','case']).agg(seeds=('seed','nunique'),
        accuracy_mean=('final_accuracy','mean'),accuracy_sd=('final_accuracy','std'),
        loss_mean=('final_loss','mean'),bytes_mean=('bytes_sent','mean'),
        training_tasks_mean=('training_tasks','mean'),execution_s_mean=('actual_execution_s','mean')).reset_index()
    comparison.to_csv(root/'comparison.csv',index=False)
    paired = summary.merge(summary[summary['case']=='reference'][['profile','partitioner','seed','final_accuracy','final_loss','bytes_sent','training_tasks']],
        on=['profile','partitioner','seed'],suffixes=('','_reference'),validate='many_to_one')
    paired['accuracy_gain_pp'] = paired.final_accuracy-paired.final_accuracy_reference
    paired['traffic_ratio'] = paired.bytes_sent/paired.bytes_sent_reference
    paired['training_ratio'] = paired.training_tasks/paired.training_tasks_reference
    paired.to_csv(root/'paired-reference.csv',index=False)
    targets=[]
    for key,g in curves.groupby(['case','profile','partitioner','seed']):
        for target in [20,30,40,50,60]:
            reached=g[g.accuracy>=target].sort_values('time')
            targets.append(dict(zip(['case','profile','partitioner','seed'],key))|{
                'target_accuracy':target,'reached':bool(len(reached)),
                'time_s':None if reached.empty else float(reached.time.iloc[0]),
                'bytes_at_checkpoint':None if reached.empty else int(reached.bytes_sent.iloc[0]),'censor_time_s':900})
    pd.DataFrame(targets).to_csv(root/'time-to-accuracy.csv',index=False)
    audit_report={'passed':True,'runs':60,'aggregate_checkpoint_rows':360,'per_node_rows':23040,
        'source_sha256':h.hexdigest(),'checks':checks,'old_baseline_replays':replays,
        'old_replay_node_rows':24*384,'learning_and_capacity_assignments_exact':True,
        'limitations':['Three seeds; short CIFAR/GNLeNet experiment, stable membership.',
            'Common hyperparameters; Moshpit is the stable-membership specialization.',
            'Targets use fixed 150-second checkpoint resolution; unreached targets are censored.',
            'All-node checkpoints and latest V2 aggregates are distinct quantities.',
            'More completed updates can increase endpoint traffic; bytes-to-target reported separately.',
            'Chunking and aggregation instantaneous in simulated time for all methods; local training explicitly timed.',
            'GPU execution seconds are separate from simulated protocol time.']}
    (root/'audit.json').write_text(json.dumps(audit_report,indent=2)+'\n')
    plot(curves,root)
    plan.update(status='complete',audited_runs=60,per_node_rows=23040,baseline_replay_node_rows=9216)
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(comparison.to_string(index=False))


def plot(curves,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors = dict(zip(CASES,['#777777','#0072B2','#009E73','#D55E00','#CC79A7']))
    for metric,xfield,name in [('accuracy','time','accuracy'),('loss','time','loss'),('accuracy','bytes_sent','traffic-accuracy')]:
        fig,axes=plt.subplots(2,2,figsize=(11.5,7.7),sharey=True)
        for axis,((profile,partitioner),data) in zip(axes.flat,curves.groupby(['profile','partitioner'],sort=True)):
            for case in CASES:
                g=data[data['case']==case].groupby('time').agg(mean=(metric,'mean'),sd=(metric,'std'),x=(xfield,'mean'))
                x=g.x/1e9 if xfield=='bytes_sent' else g.x
                axis.plot(x,g['mean'],color=colors[case],label=LABELS[case],linewidth=1.8)
                axis.fill_between(x,g['mean']-g.sd,g['mean']+g.sd,color=colors[case],alpha=.13)
            axis.set_title(f"{profile.capitalize()} / {'IID' if partitioner=='iid' else 'Dirichlet α = 0.1'}")
            axis.set_xlabel('Simulated protocol time [s]' if xfield=='time' else 'Communicated bytes [GB, decimal]')
            axis.set_ylabel('All-node mean accuracy [%]' if metric=='accuracy' else 'All-node mean loss')
            axis.grid(alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
        fig.suptitle('CIFAR-10 / GNLeNet · 64 nodes · 3 seeds · mean ± sample SD')
        fig.tight_layout(rect=[0,.09,1,.95])
        for extension in ['png','svg']:
            path=root/f'{name}.{extension}'
            fig.savefig(path,dpi=180)
            if extension=='svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/data/mdevos/conflux/results/v2-uploads64'))
    parser.add_argument('--old',type=Path,default=Path('/mnt/data/mdevos/conflux/results/peer-baselines64'))
    parser.add_argument('--code',type=Path,default=Path('/mnt/data/mdevos/conflux/v2-uploads-code'))
    args=parser.parse_args()
    audit(args.root,args.old,args.code)
