"""Measure model payloads and actual GPU costs independently of protocol time."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from dasklearn.datasets import create_dataset
    from dasklearn.model_trainer import ModelTrainer
    from dasklearn.model_evaluator import ModelEvaluator
    from dasklearn.models import create_model,serialize_model
    from dasklearn.session_settings import SessionSettings,LearningSettings
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    study=json.loads(args.study.read_text());rows=[]
    for workload in study['workloads']:
        settings=SessionSettings(algorithm='calibration',work_dir=str(args.output.parent),seed=0,
            dataset=workload['dataset'],model=workload['model'],participants=study['peers'],
            dataset_base_path=None if workload['dataset']=='digits' else str(args.data/workload['dataset']),
            evaluation_split='validation',evaluation_examples=workload['evaluation_examples'],
            learning=LearningSettings(workload['learning_rates'][0],0,0,study['batch_size'],study['local_steps']))
        data=create_dataset(settings);torch.manual_seed(0)
        model=create_model(settings.dataset,settings.model);payload=serialize_model(model)
        initial_hash=hashlib.sha256(b''.join(key.encode()+value.detach().cpu().numpy().tobytes() for key,value in sorted(model.state_dict().items()))).hexdigest()
        trainer=ModelTrainer(data,settings,0)
        trainer.train(model,study['local_steps'],device_name='cuda:0',seed=0)
        times=[]
        for seed in range(1,4):
            torch.cuda.synchronize();start=time.perf_counter()
            trainer.train(model,study['local_steps'],device_name='cuda:0',seed=seed)
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
        start=time.perf_counter();accuracy,loss=ModelEvaluator(data,settings).evaluate_accuracy(model,'cuda:0')
        torch.cuda.synchronize()
        rows.append(dict(workload=workload['name'],parameters=sum(p.numel() for p in model.parameters()),
            model_payload_bytes=len(payload),initial_payload_sha256=hashlib.sha256(payload).hexdigest(),
            local_block_median_seconds=statistics.median(times),validation_seconds=time.perf_counter()-start,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),validation_accuracy=accuracy,
            modeled_compute_step_ms=workload['compute_step_ms'],gpu=torch.cuda.get_device_name(),torch=torch.__version__))
        args.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows[-1]),flush=True)
        del data,model,trainer


if __name__=='__main__':
    main()
