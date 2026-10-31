"""Run on sacs006; preserve the main 72 runs and report 12 extra T=2 runs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SOURCE=Path('/mnt/data/mdevos/conflux/peer-baselines-code')
ROOT=Path('/mnt/data/mdevos/conflux/results/peer-baselines64')
SHA='9be351fee9cdac4fc49816d87665cabfcef722ba0d0c1161cdd2e22513cbe96c'


def main():
    digest=hashlib.sha256()
    for p in sorted([*(SOURCE/'dasklearn').rglob('*.py'),*(SOURCE/'scripts').glob('*.py')]):
        digest.update(str(p.relative_to(SOURCE)).encode());digest.update(p.read_bytes())
    assert digest.hexdigest()==SHA
    sys.path.insert(0,str(SOURCE/'scripts'))
    from report_baselines import report,plot
    report(ROOT)
    # The split driver preserves the original profile manifest. Verify every
    # actual per-run configuration and capacity assignment, including cohorts.
    common_learning={'learning_rate':.01,'momentum':0,'weight_decay':0,'batch_size':32,'local_steps':5}
    capacities={}
    for row in pd.read_csv(ROOT/'summary.csv').to_dict('records'):
        run=ROOT/row['profile']/'runs'/f'{row["variant"]}-{row["profile"]}-{row["partitioner"]}-s{row["seed"]}'
        state=json.loads((run/'settings.json').read_text())
        assert state['learning']==common_learning
        for key,value in {'seed':row['seed'],'partitioner':row['partitioner'],'alpha':.1,
            'participants':64,'duration':900000000,'test_period':150000000,
            'network_latency_ms':20,'model_size_bytes':346251,'evaluation_split':'test',
            'evaluation_examples':0,'dataset':'cifar10','model':'gnlenet'}.items():
            assert state[key]==value,(row['variant'],key)
        statistics=pd.read_csv(run/'client_statistics.csv')
        assert int(statistics.bytes_sent.sum())==int(statistics.bytes_received.sum())==int(row['bytes_sent'])
        actual=statistics[['client','training_speed','bandwidth']]
        key=(row['profile'],row['seed'])
        if key in capacities:pd.testing.assert_frame_equal(actual,capacities[key],check_exact=True)
        else:capacities[key]=actual
    from validate_conflux import plot as plot_profile
    plot_profile(pd.read_csv(ROOT/'uniform/curves.csv'),ROOT/'uniform')
    folder=ROOT/'moshpit-full'
    assert (folder/'COMPLETE').exists()
    manifest=json.loads((folder/'manifest.json').read_text())
    baseline_manifest=json.loads((ROOT/'heterogeneous/manifest.json').read_text())
    excluded={'output','variants','profiles','averaging_steps','cuda_visible_devices'}
    for key in baseline_manifest.keys()-excluded:
        assert baseline_manifest[key]==manifest[key],key
    assert manifest['averaging_steps']==2 and manifest['group_size']==8
    summary=pd.read_csv(folder/'summary.csv');curve=pd.read_csv(folder/'curves.csv')
    assert len(summary)==12 and summary.variant.eq('moshpit').all()
    audits=[];networks=[]
    for row in summary.to_dict('records'):
        run=folder/'runs'/f'moshpit-{row["profile"]}-{row["partitioner"]}-s{row["seed"]}'
        nodes=pd.read_csv(run/'accuracies.csv');network=pd.read_csv(run/'network-snapshots.csv')
        counts=nodes.groupby('time').peer.nunique()
        assert list(counts.index)==[150,300,450,600,750,900] and counts.eq(64).all()
        assert len(nodes)==384 and not nodes.duplicated(['time','peer']).any()
        assert np.isfinite(nodes[['accuracy','loss']]).all().all()
        assert network.bytes_sent.is_monotonic_increasing
        assert int(network.bytes_sent.iloc[-1])==int(row['bytes_sent'])
        state=json.loads((run/'settings.json').read_text())
        assert state['averaging_steps']==2 and state['learning']['momentum']==0
        assert state['learning']==common_learning
        assert state['model_size_bytes']==346251
        statistics=pd.read_csv(run/'client_statistics.csv')
        assert int(statistics.bytes_sent.sum())==int(statistics.bytes_received.sum())==int(row['bytes_sent'])
        actual=statistics[['client','training_speed','bandwidth']]
        pd.testing.assert_frame_equal(actual,capacities[row['profile'],row['seed']],check_exact=True)
        for key in ['variant','profile','partitioner','seed']:network[key]=row[key]
        networks.append(network)
        spreads=[]
        for t,g in nodes.groupby('time'):
            latest=g[g['round']==g['round'].max()]
            spreads.append({'time':t,'accuracy_spread':float(latest.accuracy.max()-latest.accuracy.min()),
                            'loss_spread':float(latest.loss.max()-latest.loss.min())})
        audits.append({k:row[k] for k in ['profile','partitioner','seed']} | {
            'per_node_rows':384,'finite_metrics':True,'all_nodes_per_checkpoint':True,
            'traffic_exact':True,'full_consensus_metric_spreads':spreads})
    curve=curve.merge(pd.concat(networks,ignore_index=True),
        on=['variant','profile','partitioner','seed','time'],validate='one_to_one')
    summary['variant']='moshpit-full-T2';curve['variant']='moshpit-full-T2'
    all_summary=pd.concat([pd.read_csv(ROOT/'summary.csv'),summary],ignore_index=True)
    all_curves=pd.concat([pd.read_csv(ROOT/'curves.csv'),curve],ignore_index=True)
    dest=ROOT/'with-ablation';dest.mkdir(exist_ok=True)
    all_summary.to_csv(dest/'summary.csv',index=False);all_curves.to_csv(dest/'curves.csv',index=False)
    comparison=all_summary.groupby(['profile','partitioner','variant']).agg(
        seeds=('seed','nunique'),accuracy_mean=('final_accuracy','mean'),accuracy_sd=('final_accuracy','std'),
        loss_mean=('final_loss','mean'),loss_sd=('final_loss','std'),bytes_mean=('bytes_sent','mean'),
        training_tasks_mean=('training_tasks','mean'),execution_s_mean=('actual_execution_s','mean')).reset_index()
    assert comparison.seeds.eq(3).all()
    comparison.to_csv(dest/'comparison.csv',index=False)
    keys=['profile','partitioner','seed']
    v2=all_summary[all_summary.variant.eq('conflux-v2')]
    pairs=all_summary[~all_summary.variant.eq('conflux-v2')].merge(v2,
        on=keys,suffixes=('_baseline','_v2'),validate='many_to_one')
    pairs['accuracy_delta_vs_v2']=pairs.final_accuracy_baseline-pairs.final_accuracy_v2
    pairs['traffic_ratio_vs_v2']=pairs.bytes_sent_baseline/pairs.bytes_sent_v2
    paired=pairs.groupby(['profile','partitioner','variant_baseline']).agg(
        accuracy_delta_mean=('accuracy_delta_vs_v2','mean'),accuracy_delta_sd=('accuracy_delta_vs_v2','std'),
        baseline_wins=('accuracy_delta_vs_v2',lambda a:int((a>0).sum())),
        traffic_ratio_mean=('traffic_ratio_vs_v2','mean')).reset_index()
    paired.to_csv(dest/'paired-v2-comparison.csv',index=False)
    targets=[]
    for (profile,partitioner,variant,seed),g in all_curves.groupby(['profile','partitioner','variant','seed']):
        for target in [20,30,40,50,60]:
            reached=g[g.accuracy>=target].sort_values('time')
            targets.append({'profile':profile,'partitioner':partitioner,'variant':variant,'seed':seed,
                'target_accuracy':target,'reached':bool(len(reached)),
                'time_s':None if reached.empty else float(reached.time.iloc[0]),
                'bytes_at_checkpoint':None if reached.empty else int(reached.bytes_sent.iloc[0]),
                'censor_time_s':900})
    pd.DataFrame(targets).to_csv(dest/'time-to-accuracy.csv',index=False)
    (dest/'audit.json').write_text(json.dumps({'passed':True,'execution_commit':'bd80909',
        'source_sha256':SHA,'main_runs':72,'exploratory_ablation_runs':12,
        'total_per_node_rows':84*384,'all_learning_and_capacity_assignments_exact':True,
        'all_84_runs_sent_received_and_summary_bytes_exact':True,'ablation_audits':audits,
        'ablation_note':'Two-phase Moshpit is an exploratory configuration check; both configurations are reported. No test-set winner is selected.'},indent=2)+'\n')
    plot(all_curves,dest,64,3)
    columns=[('uniform','iid'),('uniform','dirichlet'),('heterogeneous','iid'),('heterogeneous','dirichlet')]
    lines=['| Method | Uniform IID | Uniform Dirichlet | Heterogeneous IID | Heterogeneous Dirichlet |',
           '| --- | ---: | ---: | ---: | ---: |']
    indexed=comparison.set_index(['profile','partitioner','variant'])
    for variant in all_summary.variant.unique():
        cells=[variant]
        for profile,partitioner in columns:
            row=indexed.loc[profile,partitioner,variant]
            cells.append(f'{row.accuracy_mean:.2f} ± {row.accuracy_sd:.2f}')
        lines.append('| '+' | '.join(cells)+' |')
    (dest/'table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print(paired.to_string(index=False))


if __name__=='__main__':main()
