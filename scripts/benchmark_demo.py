"""Validation-only DeMo LR selection, followed by the fixed CIFAR test matrix.

Each H has its own LR selected on mean uniform IID/Dirichlet validation accuracy
at 900 simulated seconds, seed 0. Both H=1 and H=5 are reported on heldout test.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd


def command(output,dataset,profile,steps,seed,rate,split,device,variants=('demo',)):
    return [sys.executable,str(Path(__file__).with_name('run_cifar.py')),
        '--output',str(output),'--dataset-base-path',dataset,'--variants',*variants,
        '--profiles',profile,'--partitioners','iid','dirichlet','--seeds',str(seed),
        '--peers','64','--sample-size','16','--chunks','16','--relay-count','16','--immediate-uploads',
        '--duration','900','--test-period','150','--bandwidth','100000','--step-ms','20','--latency-ms','20',
        '--learning-rate',str(rate),'--local-steps',str(steps),'--batch-size','32','--alpha','.1',
        '--momentum','0','--weight-decay','0','--model-bytes','346251','--evaluation-split',split,
        '--torch-threads','4','--device',device,'--demo-decay','.999','--demo-chunk','64','--demo-topk','32']


def run(cmd):
    print(json.dumps({'starting':cmd}),flush=True);subprocess.run(cmd,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset-base-path',required=True)
    p.add_argument('--mode',choices=['tune','evaluate'],required=True)
    p.add_argument('--profile',choices=['uniform','heterogeneous'],default='uniform')
    p.add_argument('--device',default='cuda:0');args=p.parse_args();root=args.output.resolve()
    if args.mode=='tune':
        scores=[]
        for steps in [5,1]:
            for rate in [.0003,.001,.003,.01]:
                folder=root/'validation'/f'h{steps}-lr{rate:g}'
                run(command(folder,args.dataset_base_path,'uniform',steps,0,rate,'validation',args.device))
                frame=pd.read_csv(folder/'summary.csv')
                if len(frame)!=2:raise ValueError('Missing tuning partition')
                scores.append({'local_steps':steps,'learning_rate':rate,
                    'mean_validation_accuracy':float(frame.final_accuracy.mean()),
                    'iid_validation_accuracy':float(frame[frame.partitioner=='iid'].final_accuracy.iloc[0]),
                    'dirichlet_validation_accuracy':float(frame[frame.partitioner=='dirichlet'].final_accuracy.iloc[0])})
        scores=pd.DataFrame(scores);scores.to_csv(root/'validation-scores.csv',index=False)
        selected={str(h):float(g.sort_values(['mean_validation_accuracy','learning_rate'],ascending=[False,True]).iloc[0].learning_rate)
                  for h,g in scores.groupby('local_steps')}
        (root/'selection.json').write_text(json.dumps({'selected':selected,
            'rule':'Maximize mean IID/Dirichlet validation accuracy at 900s; uniform, seed 0; lower LR breaks ties.',
            'candidates':[.0003,.001,.003,.01],'split':'5000 heldout tuning images; no test-set selection'},indent=2)+'\n')
    else:
        selected=json.loads((root/'selection.json').read_text())['selected']
        # First verify the optimized V2 cache under the new execution source.
        run(command(root/args.profile/'v2-replay',args.dataset_base_path,args.profile,5,0,.01,'test',args.device,('conflux-v2',)))
        for seed in [0,1,2]:
            for steps in [5,1]:
                run(command(root/args.profile/f'h{steps}-s{seed}',args.dataset_base_path,args.profile,steps,seed,selected[str(steps)],'test',args.device))
        (root/args.profile/'COMPLETE').write_text('All DeMo test runs and V2 cache replay completed\n')


if __name__=='__main__':main()
