"""Rebuild paired threshold comparisons and compact traffic summaries."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
s=pd.read_csv(root/'all-runs.csv');t=pd.read_csv(root/'time-to-accuracy.csv')
confirm=s[(s.stage=='confirm')&(s.status=='completed')]
assert len(confirm)==27 and len(s)==63 and s.status.eq('completed').all()
assert confirm.groupby('variant').seed.nunique().eq(3).all()
rows=[]
variants=['server-momentum-fixed','server-adam-fixed','server-adam-rotate-fp32',
          'server-adam-rotate-int8','server-adam-epoch8-int8']
for baseline in ['original-fedavg-fixed','server-sgd-fixed']:
 for variant in variants:
  for target in [25,30,35,40,45]:
   g=t[(t.variant==variant)&(t.target==target)].merge(t[(t.variant==baseline)&(t.target==target)],on='seed',suffixes=('_opt','_baseline'))
   for _,r in g.iterrows():
    both=bool(r.reached_opt and r.reached_baseline)
    rows.append(dict(baseline=baseline,variant=variant,target=target,seed=int(r.seed),both_reached=both,
     observed_speedup=r.upper_s_baseline/r.upper_s_opt if both else None,
     interval_separated_win=bool(r.reached_opt and r.upper_s_opt<r.lower_s_baseline),
     opt_upper_s=r.upper_s_opt if r.reached_opt else None,baseline_upper_s=r.upper_s_baseline if r.reached_baseline else None))
paired=pd.DataFrame(rows);paired.to_csv(root/'paired-speedups.csv',index=False)
summary=[]
for key,g in paired.groupby(['baseline','variant','target']):
 summary.append(dict(zip(['baseline','variant','target'],key))|dict(
  reached_both=int(g.both_reached.sum()),interval_separated_wins=int(g.interval_separated_win.sum()),
  median_observed_speedup=float(g.observed_speedup.median()) if g.both_reached.all() else None))
pd.DataFrame(summary).to_csv(root/'speedup-summary.csv',index=False)
comparison=pd.read_csv(root/'comparison.csv').set_index('variant')
fp=comparison.loc['server-adam-rotate-fp32'];q=comparison.loc['server-adam-rotate-int8'];epoch=comparison.loc['server-adam-epoch8-int8']
traffic=dict(rotating_fp32_handoff_payload_mb=float(fp.handoff_payload_mean/1e6),
 rotating_int8_handoff_payload_mb=float(q.handoff_payload_mean/1e6),
 epoch8_int8_handoff_payload_mb=float(epoch.handoff_payload_mean/1e6),
 rotating_handoff_payload_reduction_pct=float(100*(1-q.handoff_payload_mean/fp.handoff_payload_mean)),
 rotating_total_traffic_reduction_pct=float(100*(1-q.bytes_mean/fp.bytes_mean)),
 note='Scheduled handoff payload includes FP32 base weights; total traffic includes actual sent bytes, headers and unfinished-transfer progress.')
(root/'traffic-summary.json').write_text(json.dumps(traffic,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
shown=[('original-fedavg-fixed','Original Conflux'),('server-sgd-fixed','Tuned server SGD'),
       ('server-momentum-fixed','FedAvgM'),('server-adam-fixed','FedAdam'),('server-adam-rotate-int8','FedAdam + 8-bit handoffs')]
x=np.arange(3);width=.15
for i,(variant,label) in enumerate(shown):
 g=t[(t.variant==variant)&(t.target==35)].sort_values('seed')
 assert len(g)==3 and g.reached.all()
 ax.bar(x+(i-2)*width,g.upper_s,width,label=label)
ax.set(xticks=x,xticklabels=['Seed 0','Seed 1','Seed 2'],ylabel='First observed simulated seconds to 35% validation accuracy',
 title='Reducer FedOpt: time to a common accuracy target\nNon-IID CIFAR-10/GNLeNet · heterogeneous hardware')
ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.supxlabel('50-second checkpoint resolution. Fixed 1,000-image validation subset; all 64 stored node checkpoints.',fontsize=9)
fig.savefig(root/'time-to-target.png',dpi=170);fig.savefig(root/'time-to-target.pdf');plt.close(fig)
print(json.dumps(traffic,indent=2))
