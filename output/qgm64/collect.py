"""Audit frozen qgm64 jobs; export only compact, per-seed/curve summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def collect(root,out,partial=False):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'plan.json').read_text());selection=json.loads((root/'selection.json').read_text())
    rows=[];curves=[];aggregates=[];capacities={};audited_curve_rows=audited_test_rows=0
    for profile in ['uniform','heterogeneous']:
        for partitioner in ['iid','dirichlet']:
            for seed in plan['seeds']:
                for variant in plan['variants']:
                    rate=selection['selected'][variant]
                    job=root/'evaluate'/f'{variant}-{profile}-{partitioner}-s{seed}-lr{rate:g}'
                    if not (job/'COMPLETE').exists() and partial:continue
                    assert (job/'COMPLETE').exists(),job
                    row=json.loads((job/'summary.json').read_text());assert row['source_sha256']==plan['source_sha256']
                    run=job/'runs'/f'{variant}-{profile}-{partitioner}-s{seed}'
                    cfg=json.loads((run/'settings.json').read_text())
                    assert cfg['participants']==plan['peers'] and cfg['learning']['local_steps']==1
                    assert cfg['evaluation_split']=='validation' and cfg['evaluation_examples']==1000
                    frame=pd.read_csv(run/'accuracies.csv');heldout=pd.read_csv(run/'heldout.csv')
                    times=np.arange(1,7)*plan['duration']/6
                    assert set(frame.time)==set(times) and len(frame)==6*plan['peers']
                    assert not frame.duplicated(['time','peer']).any()
                    assert frame.groupby('time').peer.nunique().eq(plan['peers']).all()
                    assert len(heldout)==plan['peers'] and set(heldout.peer)==set(range(plan['peers']))
                    assert heldout.time.eq(plan['duration']).all()
                    for data in [frame,heldout]:
                        assert np.isfinite(data[['accuracy','loss']]).all().all(),job
                        assert data['round'].ge(0).all()
                        for _,group in data[data['round']==0].groupby('time'):
                            assert group.accuracy.nunique()==1 and group.loss.nunique()==1
                    assert abs(row['final_test_accuracy']-float(heldout.accuracy.mean()))<1e-10
                    assert abs(row['final_test_loss']-float(heldout.loss.mean()))<1e-10
                    counts=pd.read_csv(run/'client_statistics.csv').sort_values('client').reset_index(drop=True)
                    assert counts.bytes_sent.sum()==counts.bytes_received.sum()
                    key=profile,seed
                    capacity=counts[['client','training_speed','bandwidth']]
                    if key in capacities:pd.testing.assert_frame_equal(capacity,capacities[key],check_exact=True)
                    else:capacities[key]=capacity
                    curve=pd.read_csv(job/'curves.csv');assert curve.nodes.eq(plan['peers']).all()
                    network=pd.read_csv(run/'network-snapshots.csv')
                    assert set(network.time)==set(times) and network.bytes_sent.is_monotonic_increasing
                    row['bytes_sent_including_inflight']=int(network.bytes_sent.iloc[-1])
                    curve=curve.merge(network,on='time',validate='one_to_one')
                    curves.append(curve);rows.append(row)
                    if variant=='conflux-v2':
                        latest=pd.read_csv(run/'latest-aggregate.csv')
                        for _,r in latest.iterrows():
                            group=frame[frame.time==r.time];round_nr=int(group['round'].max())
                            recipients=group[group['round']==round_nr]
                            assert recipients.accuracy.nunique()==1 and recipients.loss.nunique()==1
                            assert abs(r.accuracy-recipients.accuracy.iloc[0])<1e-10 and abs(r.loss-recipients.loss.iloc[0])<1e-10
                            assert abs(r.mean_staleness_rounds-(round_nr-group['round']).mean())<1e-10
                            assert r.max_staleness_rounds==(round_nr-group['round']).max()
                        r=int(heldout['round'].max());last=heldout[heldout['round']==r]
                        assert last.accuracy.nunique()==1 and last.loss.nunique()==1
                        assert abs(row['final_test_latest_aggregate_accuracy']-float(last.accuracy.iloc[0]))<1e-10
                        latest['variant']=variant;latest['profile']=profile;latest['partitioner']=partitioner;latest['seed']=seed
                        aggregates.append(latest)
                    audited_curve_rows+=len(frame);audited_test_rows+=len(heldout)
    assert rows,'No completed evaluation jobs'
    if not partial:assert len(rows)==48
    summary=pd.DataFrame(rows);all_curves=pd.concat(curves,ignore_index=True)
    controls=summary[summary.variant.isin(['qgm','dsgdm','dsgd'])]
    for _,group in controls.groupby(['profile','partitioner','seed']):
        for col in ['bytes_sent','bytes_sent_including_inflight','training_tasks','simulated_compute_s']:
            assert group[col].nunique()==1,(col,group)
    for _,group in summary.groupby(['variant','profile','seed']):
        for col in ['bytes_sent','bytes_sent_including_inflight','training_tasks','simulated_compute_s']:
            assert group[col].nunique()==1,('Data partition changed protocol schedule',col,group)
    summary.to_csv(out/'summary.csv',index=False);all_curves.to_csv(out/'curves.csv',index=False)
    if aggregates:pd.concat(aggregates,ignore_index=True).to_csv(out/'latest-aggregate.csv',index=False)
    groups=summary.groupby(['profile','partitioner','variant'],as_index=False)
    table=groups.agg(test_accuracy_mean=('final_test_accuracy','mean'),test_accuracy_sd=('final_test_accuracy','std'),
        test_loss_mean=('final_test_loss','mean'),traffic_bytes_mean=('bytes_sent_including_inflight','mean'),
        training_tasks_mean=('training_tasks','mean'),execution_seconds_mean=('actual_execution_s','mean'))
    table.to_csv(out/'comparison.csv',index=False)
    seed_tta=[]
    for (profile,part,variant,seed),g in all_curves.groupby(['profile','partitioner','variant','seed']):
        for target in [30,40,50,60]:
            reached=g[g.accuracy>=target].sort_values('time')
            seed_tta.append({'profile':profile,'partitioner':part,'variant':variant,'seed':seed,
                'validation_target':target,'observed_time_s':float(reached.time.iloc[0]) if len(reached) else None,
                'observed_traffic_bytes':int(reached.bytes_sent.iloc[0]) if len(reached) else None,
                'censored_after_s':plan['duration']})
    seed_tta=pd.DataFrame(seed_tta);seed_tta.to_csv(out/'time-to-accuracy-seeds.csv',index=False)
    tta=[]
    for (profile,part,variant),g in summary.groupby(['profile','partitioner','variant']):
        for target in [30,40,50,60]:
            values=seed_tta[(seed_tta.profile==profile)&(seed_tta.partitioner==part)&
                (seed_tta.variant==variant)&(seed_tta.validation_target==target)].observed_time_s.dropna()
            tta.append({'profile':profile,'partitioner':part,'variant':variant,'validation_target':target,
                'reached':len(values),'seeds':len(g),'observed_mean_s':float(values.mean()) if len(values) else None,
                'observed_sd_s':float(values.std()) if len(values)>1 else None,'censored_after_s':plan['duration']})
    pd.DataFrame(tta).to_csv(out/'time-to-accuracy.csv',index=False)
    (out/'audit.json').write_text(json.dumps({'passed':True,'partial':partial,'jobs':len(rows),'curve_node_rows':audited_curve_rows,
        'full_test_node_rows':audited_test_rows,'curve_metric':'1000 balanced validation images',
        'final_metric':'10000 heldout test images','plan':plan,'selection':selection,
        'checks':['all nodes, six snapshots, nonnegative round IDs and finite metrics; round-0 initial recipients agree','shared capacities by profile/seed',
            'balanced traffic counters','identical traffic/work counts among the three momentum controls',
            'data partition does not change the protocol schedule within a profile/seed',
            'latest V2 aggregate consistency, separately for validation and full test']},indent=2)+'\n')
    print(table.to_string(index=False));print('audited rows',audited_curve_rows,audited_test_rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--partial',action='store_true')
    args=p.parse_args();collect(args.input,args.output,args.partial)
