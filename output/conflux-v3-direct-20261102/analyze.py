"""Post-run protocol, geometry and timing diagnostics for the frozen study."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=[];paths={};round_curves=[];rates=[]
for record in sorted(root.glob('runs/*/record.json')):
    r=json.loads(record.read_text())
    if r['status']!='completed':continue
    data=next(record.parent.glob('data/*/accuracies.csv')).parent
    paths[(r['algorithm'],r['seed'])]=data
    rounds=pd.read_csv(data/'conflux_v2_rounds.csv')
    grouped=rounds.groupby('round').agg(time_us=('time_us','max'),recipients=('client','nunique'))
    complete=grouped[grouped.recipients==16].copy()
    complete['interval_seconds']=complete.time_us.diff()/1e6
    row=dict(algorithm=r['algorithm'],seed=r['seed'],first_full_cohort_seconds=complete.time_us.iloc[0]/1e6,
             completed_full_cohorts=len(complete),median_interval_seconds=complete.interval_seconds.median(),
             mean_interval_seconds=complete.interval_seconds.mean())
    for step,g in complete.iterrows():
        round_curves.append(dict(algorithm=r['algorithm'],seed=r['seed'],round=step,time_seconds=g.time_us/1e6))
    if r['algorithm']=='conflux-v3':
        s=json.loads((data/'scheduler.json').read_text())
        row.update(chunks=s['chunks'],reducers=s['reducers'],startup_seconds=s['startup_last_plan_us']/1e6,
                   planner_wall_seconds=s['planner_wall_seconds'],scheduler_cpu_seconds=s['scheduler_cpu_us']/1e6)
        candidates=pd.read_csv(data/'scheduler-candidates.csv')
        selected=candidates[(candidates.chunks==s['chunks'])&(candidates.reducers==s['reducers'])].dropna(subset=['forecast_seconds'])
        row['selected_one_round_forecast_seconds']=selected.forecast_seconds.min()
        stats=pd.read_csv(data/'client_statistics.csv').set_index('client')
        for peer,profile in s['profiles'].items():
            true=float(stats.loc[int(peer),'bandwidth'])
            rates.append(dict(seed=r['seed'],peer=int(peer),true_bandwidth=true,estimated_upload=profile[0],estimated_download=profile[1],
                              upload_ratio=profile[0]/true,download_ratio=profile[1]/true,estimated_latency_us=profile[3]))
    rows.append(row)
pd.DataFrame(rows).to_csv(root/'timing-diagnostics.csv',index=False)
pd.DataFrame(round_curves).to_csv(root/'completed-cohorts.csv',index=False)
pd.DataFrame(rates).to_csv(root/'profile-diagnostics.csv',index=False)
checks=[]
for seed in [0,1,2]:
    if ('conflux-v2',seed) not in paths or ('sharded-ps',seed) not in paths:continue
    a=paths[('conflux-v2',seed)];b=paths[('sharded-ps',seed)]
    fields=['client','training_speed','bandwidth','bytes_sent','bytes_received']
    equal_stats=pd.read_csv(a/'client_statistics.csv')[fields].equals(pd.read_csv(b/'client_statistics.csv')[fields])
    equal_rounds=pd.read_csv(a/'conflux_v2_rounds.csv').equals(pd.read_csv(b/'conflux_v2_rounds.csv'))
    fields=['peer','round','time','accuracy','loss']
    heldout=lambda p:pd.concat([pd.read_csv(f) for f in sorted(p.glob('heldout_[0-9]*.csv'))]).sort_values('peer')[fields].reset_index(drop=True)
    equal_heldout=heldout(a).equals(heldout(b))
    assert equal_stats and equal_rounds and equal_heldout
    checks.append(dict(seed=seed,all_round_completions_and_contributors_equal=equal_rounds,
                       per_node_capacities_and_wire_bytes_equal=equal_stats,all_final_test_rows_equal=equal_heldout))
(root/'matched-protocol-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
round_values=[]
for seed in [0,1,2]:
    tables=[]
    for alg in ['conflux-v2','conflux-v3']:
        data=pd.read_csv(paths[(alg,seed)]/'accuracies.csv')
        assert data.groupby('round').accuracy.nunique().max()==1
        assert data.groupby('round').loss.nunique().max()==1
        tables.append(data.groupby('round')[['accuracy','loss']].first())
    a,b=tables[0].align(tables[1],join='inner')
    difference=(a-b).abs().max()
    assert difference.max()==0
    round_values.append(dict(seed=seed,common_observed_rounds=len(a),
        max_validation_accuracy_difference=float(difference.accuracy),max_validation_loss_difference=float(difference.loss)))
(root/'round-value-audit.json').write_text(json.dumps(round_values,indent=2)+'\n')
frame=pd.DataFrame(rows)
colors={'conflux-v2':'#0072B2','sharded-ps':'#D55E00','conflux-v3':'#009E73'}
labels={'conflux-v2':'Conflux V2','sharded-ps':'Matched sharded PS','conflux-v3':'Automatic direct-only'}
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
curves=pd.DataFrame(round_curves)
for (alg,seed),g in curves.groupby(['algorithm','seed']):
    axes[0].step(g.time_seconds,g['round'],where='post',label=labels[alg] if seed==0 else None,
                 color=colors[alg],alpha=.65,linestyle='--' if alg=='sharded-ps' else '-')
axes[0].set(xlabel='Simulated seconds (startup included)',ylabel='Completed full-cohort rounds')
axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
stats=frame.groupby('algorithm').median_interval_seconds.agg(['mean','std'])
axes[1].barh([labels[a] for a in stats.index],stats['mean'],xerr=stats['std'].fillna(0),capsize=4,color=[colors[a] for a in stats.index])
axes[1].set(xlabel='Median interval between complete cohorts (seconds)',title='Excludes startup / first cohort · lower is faster')
axes[1].grid(axis='x',alpha=.2)
fig.suptitle('Protocol throughput · 64 nodes / 16 changing trainers · three paired seeds')
fig.savefig(root/'throughput.png',dpi=170);fig.savefig(root/'throughput.pdf');plt.close(fig)
print(frame.to_string(index=False))
