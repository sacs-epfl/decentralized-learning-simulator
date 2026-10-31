"""Read-only audit/collection of the already launched long experiment."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT=Path('/mnt/data/mdevos/conflux')
BASE=ROOT/'results/tiny128-long18000'
OUT=BASE/'collected'
EXPECTED='cc1c2670f8cbfcbc1c5011a0a065f7540b614de7b770e149488098367646d58e'


def main():
    digest=hashlib.sha256();source=ROOT/'tiny-memoized-code'
    for p in sorted([*(source/'dasklearn').rglob('*.py'),*(source/'scripts').glob('*.py')]):
        digest.update(str(p.relative_to(source)).encode());digest.update(p.read_bytes())
    assert digest.hexdigest()==EXPECTED
    OUT.mkdir(exist_ok=True)
    curves=[];summaries=[];manifests=[];settings=[];audits={};final_aggregate=None
    for variant in ['shatter-r2','conflux-v2']:
        run=BASE/variant
        assert (run/'COMPLETE').exists(),variant
        rows=pd.read_csv(run/'runs'/variant/'accuracies.csv')
        counts=rows.groupby('time').peer.nunique()
        assert list(counts.index)==list(range(900,18001,900)) and counts.eq(128).all()
        assert len(rows)==2560 and not rows.duplicated(['time','peer']).any()
        assert rows.peer.between(0,127).all() and np.isfinite(rows[['accuracy','loss']]).all().all()
        final=pd.read_csv(run/'final.csv')
        assert len(final)==128 and final.peer.nunique()==128 and final.peer.between(0,127).all()
        assert np.isfinite(final[['accuracy','loss']]).all().all()
        old=pd.read_csv(ROOT/'results/tiny128'/variant/'runs'/variant/'accuracies.csv')
        prefix=rows[rows.time.isin([900,1800])].sort_values(['time','peer']).reset_index(drop=True)
        prior=old[old.time.isin([900,1800])].sort_values(['time','peer']).reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix,prior,check_exact=True)
        manifest=json.loads((run/'manifest.json').read_text());assert manifest['source_sha256']==EXPECTED
        manifests.append(manifest)
        settings.append(json.loads((run/'runs'/variant/'settings.json').read_text()))
        stats=pd.read_csv(run/'runs'/variant/'client_statistics.csv')
        summary=pd.read_csv(run/'summary.csv')
        assert int(stats.bytes_sent.sum())==int(summary.bytes_sent.iloc[0])
        assert int(stats.bytes_received.sum())==int(summary.bytes_sent.iloc[0])
        summaries.append(summary)
        curve=rows.groupby('time',as_index=False).agg(accuracy=('accuracy','mean'),loss=('loss','mean'),nodes=('peer','nunique'))
        curve['variant']=variant;curve['readout']='all-node stored checkpoints';curves.append(curve)
        dest=OUT/variant;dest.mkdir(exist_ok=True)
        for name in ['manifest.json','summary.csv','curves.csv','final.csv','COMPLETE']:
            shutil.copy2(run/name,dest/name)
        if variant=='conflux-v2':
            fresh=[]
            for time,group in rows.groupby('time'):
                r=int(group['round'].max());latest=group[group['round']==r]
                assert latest.accuracy.nunique()==latest.loss.nunique()==1
                fresh.append({'time':time,'round':r,'accuracy':float(latest.accuracy.iloc[0]),
                    'loss':float(latest.loss.iloc[0]),'recipients':len(latest),
                    'mean_staleness_rounds':float((r-group['round']).mean()),
                    'max_staleness_rounds':int((r-group['round']).max())})
            latest=pd.DataFrame(fresh);latest.to_csv(OUT/'latest-v2-aggregate.csv',index=False)
            r=int(final['round'].max());f=final[final['round']==r]
            assert f.accuracy.nunique()==f.loss.nunique()==1
            final_aggregate={'round':r,'accuracy_10000':float(f.accuracy.iloc[0]),'loss_10000':float(f.loss.iloc[0]),
                'recipients':len(f),'mean_staleness_rounds':float((r-final['round']).mean()),
                'max_staleness_rounds':int((r-final['round']).max())}
            (OUT/'final-v2-aggregate.json').write_text(json.dumps(final_aggregate,indent=2)+'\n')
            extra=latest[['time','accuracy','loss']].copy();extra['variant']='conflux-v2'
            extra['readout']='latest completed sample aggregate';curves.append(extra)
        audits[variant]={'curve_rows':len(rows),'checkpoints':len(counts),'nodes_per_snapshot':128,
            'final_full_rows':len(final),'prefix_900_1800_exact_rows':len(prefix),'finite_metrics':True,
            'traffic_balanced':True,'source_sha256':EXPECTED}
    for key in manifests[0].keys()-{'variant','output','cuda_visible_devices'}:
        assert manifests[0][key]==manifests[1][key],key
    for key in ['learning','participants','dataset','model','alpha','partitioner','duration','test_period',
                'model_size_bytes','network_latency_ms','evaluation_examples','torch_threads','seed']:
        assert settings[0][key]==settings[1][key],key
    combined=pd.concat(curves,ignore_index=True);summary=pd.concat(summaries,ignore_index=True)
    combined.to_csv(OUT/'curves.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False)
    targets=[]
    for (variant,readout),frame in combined.groupby(['variant','readout']):
        for target in [2,5,10,15,20,25,30]:
            reached=frame[frame.accuracy>=target].sort_values('time')
            targets.append({'variant':variant,'readout':readout,'target_accuracy_2000':target,
                'reached':bool(len(reached)),'first_checkpoint_s':None if reached.empty else float(reached.time.iloc[0]),
                'censor_time_s':18000})
    pd.DataFrame(targets).to_csv(OUT/'time-to-accuracy.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps({'passed':True,'variants':audits,'common_configuration_exact':True,
        'latest_v2_recipients_identical_at_every_checkpoint_and_final':True,
        'source_sha256':EXPECTED,'one_seed':True,'curve_examples':2000,'final_examples':10000,
        'traffic':'Measured endpoint totals at the common 18000-second budget; intermediate cumulative traffic was not recorded.',
        'execution_note':'GPU execution uses the original frozen executor; separate optimization validation uses newer code.',
        'model_note':'Stored completed post-aggregation checkpoints, not intermediate SGD weights. No free global parameter averaging.'},indent=2)+'\n')
    plot(combined,summary,pd.read_csv(OUT/'latest-v2-aggregate.csv'))
    print(summary.to_string(index=False));print('FINAL V2 AGGREGATE',json.dumps(final_aggregate))


def plot(curves,summary,fresh):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for (variant,readout),g in curves.groupby(['variant','readout']):
        label=('Shatter: all-node mean' if variant=='shatter-r2' else 'V2: all-node mean' if readout.startswith('all-node') else 'V2: latest sample aggregate')
        color='tab:orange' if variant=='shatter-r2' else 'tab:blue'
        style='--' if readout.startswith('latest') else '-'
        axes[0,0].plot(g.time/3600,g.accuracy,style,color=color,label=label)
        axes[0,1].plot(g.time/3600,g.loss,style,color=color,label=label)
    axes[0,0].set(xlabel='Simulated protocol time [h]',ylabel='Accuracy, fixed 2000-image subset [%]')
    axes[0,1].set(xlabel='Simulated protocol time [h]',ylabel='Loss, fixed 2000-image subset')
    axes[0,0].legend(fontsize=8)
    axes[1,0].plot(fresh.time/3600,fresh.mean_staleness_rounds,label='Mean age',color='tab:blue')
    axes[1,0].plot(fresh.time/3600,fresh.max_staleness_rounds,label='Maximum age',color='tab:gray',linestyle=':')
    axes[1,0].set(xlabel='Simulated protocol time [h]',ylabel='V2 node age behind latest round')
    axes[1,0].legend()
    names=['Shatter','Conflux V2'];by_name=summary.set_index('variant')
    values=[by_name.loc[v,'bytes_sent']/1e12 for v in ['shatter-r2','conflux-v2']]
    axes[1,1].bar(names,values,color=['tab:orange','tab:blue'])
    axes[1,1].set(ylabel='Total traffic at 5 simulated hours [TB]')
    for i,value in enumerate(values):axes[1,1].text(i,value,f'{value:.2f}',ha='center',va='bottom')
    for ax in axes.flat:ax.grid(alpha=.2,axis='y')
    fig.suptitle('Tiny ImageNet / ResNet18, 128 nodes, heterogeneous network, Dirichlet α=0.1; seed 0')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'comparison.png',dpi=180)
    fig.savefig(OUT/'comparison.svg');plt.close(fig)


if __name__=='__main__':main()
