import argparse
import importlib.util
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--steps',type=int,required=True);args=p.parse_args()
root=Path('/mnt/data/mdevos/conflux/results/demo64');code=Path('/mnt/data/mdevos/conflux/demo-code')
spec=importlib.util.spec_from_file_location('benchmark',code/'scripts/benchmark_demo.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
scores=[]
for rate in [.0003,.001,.003,.01]:
    folder=root/'validation'/f'h{args.steps}-lr{rate:g}'
    b.run(b.command(folder,'/mnt/data/mdevos/conflux/datasets','uniform',args.steps,0,rate,'validation','cuda:0'))
    frame=pd.read_csv(folder/'summary.csv');assert len(frame)==2
    scores.append({'local_steps':args.steps,'learning_rate':rate,'mean_validation_accuracy':float(frame.final_accuracy.mean()),
        'iid_validation_accuracy':float(frame[frame.partitioner=='iid'].final_accuracy.iloc[0]),
        'dirichlet_validation_accuracy':float(frame[frame.partitioner=='dirichlet'].final_accuracy.iloc[0])})
pd.DataFrame(scores).to_csv(root/f'validation-h{args.steps}.csv',index=False)
