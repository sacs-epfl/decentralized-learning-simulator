"""Fixed CIFAR matrix, prioritizing a seed-0 preview and reusing audited V2 runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset-base-path',required=True)
    parser.add_argument('--profile',choices=['uniform','heterogeneous'],required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    folder=args.output.resolve()/args.profile;folder.mkdir(parents=True,exist_ok=True)
    commands=[]
    for steps in [5,1]:
        for seed in [0,1,2]:
            case=f'h{steps}-s{seed}'
            command=[sys.executable,str(Path(__file__).with_name('run_cifar.py')),
                '--output',str(folder/case),'--dataset-base-path',args.dataset_base_path,
                '--variants','dspodfl','--profiles',args.profile,'--partitioners','iid','dirichlet',
                '--seeds',str(seed),'--peers','64','--sample-size','16','--chunks','16','--relay-count','16',
                '--duration','900','--test-period','150','--bandwidth','100000','--step-ms','20',
                '--latency-ms','20','--learning-rate','.01','--local-steps',str(steps),'--batch-size','32',
                '--alpha','.1','--momentum','0','--weight-decay','0','--model-bytes','346251',
                '--evaluation-split','test','--torch-threads','4','--device',args.device,
                '--dspod-degree','4','--dspod-sgd-probability','.5',
                '--dspod-communication-probability','.5','--dspod-probability-mode','capacity']
            commands.append(command)
    dispatch={'profile':args.profile,'commands':commands,
        'comparison':'Original and immediate-wide V2 cached from audited v2-uploads64; no duplicate V2 runs.',
        'semantics':'H=1 is Eq. (2); H=5 uses the same local-SGD block length as the comparison matrix.',
        'policy':'Connected degree-4 regular graph, Metropolis physical-degree weights; nominal probabilities 0.5 scaled down by relative capacity.',
        'selection':'Predeclared two semantic configurations; report both, no heldout-test hyperparameter selection.'}
    path=folder/'dispatch.json'
    if path.exists() and json.loads(path.read_text())!=dispatch:raise ValueError('Existing dispatch differs')
    path.write_text(json.dumps(dispatch,indent=2)+'\n')
    for command in commands:
        print(json.dumps({'starting':command}),flush=True)
        subprocess.run(command,check=True)
    (folder/'COMPLETE').write_text('All 12 profile runs complete\n')


if __name__=='__main__':main()
