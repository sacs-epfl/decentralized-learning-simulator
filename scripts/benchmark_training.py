"""Compare complete independent SGD task cost and state accuracy on one device.

Dataset loading and warmup are reported separately. Forced batched trials
measure candidate widths; production execution additionally applies startup
numerical/performance checks and can fall back to scalar training.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from dasklearn import functions
from dasklearn.datasets import create_dataset
from dasklearn.models import copy_model_state
from dasklearn.session_settings import SessionSettings,LearningSettings
from dasklearn.tasks.training import TrainingExecutor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tasks",type=int,default=32)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--widths",type=int,nargs="+",default=[2,4,8])
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--shared-inputs",action="store_true")
    args=p.parse_args()
    if args.tasks<2 or args.tasks>128 or args.repeats<1 or any(w<2 for w in args.widths):
        p.error("Use 2..128 tasks, positive repeats, and widths >=2")
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    s=SessionSettings(algorithm="benchmark",seed=0,work_dir=str(args.output),data_dir=str(args.output),
        dataset="tiny_imagenet",model="resnet18",dataset_base_path=args.data,participants=128,
        partitioner="dirichlet",alpha=.1,torch_device_name=args.device,torch_threads=4,
        learning=LearningSettings(.01,.9,0,32,5))
    start=time.perf_counter();dataset=create_dataset(s);loading=time.perf_counter()-start
    base=functions.initialize(s,{})[0]
    models=[]
    for peer in range(args.tasks):
        model=base if args.shared_inputs else copy_model_state(base,s.dataset,s.model)
        if not args.shared_inputs:
            with torch.no_grad():next(model.parameters()).view(-1)[0].add_(peer*.001)
        models.append(model)
    def sync():
        if torch.device(args.device).type=="cuda":torch.cuda.synchronize()
    expected=[];records=[]
    modes=[("reference",False,False,1),("resident",True,False,1)]+[(f"batch-{w}",True,True,w) for w in args.widths]
    for label,resident,batching,width in modes:
        settings=replace(s,gpu_residency=resident,training_batching=batching,training_batch_size=width)
        functions.dataset=dataset;functions.model_trainers=None
        backend=None if label=="reference" else TrainingExecutor(settings,autotune=False)
        def run(round_nr):
            data=[{"model":m,"round":round_nr,"peer":i,"time":0} for i,m in enumerate(models)]
            if backend is None:return [functions.train(settings,item)[0] for item in data]
            outputs=[]
            for offset in range(0,len(data),width):
                group=data[offset:offset+width]
                outputs.extend(backend.train_many(group,["shared" if args.shared_inputs else None]*len(group)))
            return outputs
        start=time.perf_counter();warm=run(3);del warm;sync();warmup=time.perf_counter()-start
        if torch.device(args.device).type=="cuda":torch.cuda.reset_peak_memory_stats()
        durations=[];max_error=0.
        try:
            for repeat in range(args.repeats):
                sync();start=time.perf_counter();actual=run(17+repeat);sync()
                durations.append(time.perf_counter()-start)
                if label=="reference":expected.append(actual)
                else:
                    for a,b in zip(actual,expected[repeat]):
                        for name,value in a.state_dict().items():
                            other=b.state_dict()[name]
                            if label=="resident" or backend.strategy=="cuda-streams":
                                if not torch.equal(value,other):raise AssertionError(f"Resident mismatch: {name}")
                            else:torch.testing.assert_close(value,other,rtol=1e-4,atol=1e-5,msg=name)
                            if value.is_floating_point():max_error=max(max_error,float((value-other).abs().max()))
                    del actual
            record={"mode":label,"tasks":args.tasks,"repeats_s":durations,
                "median_s":statistics.median(durations),"warmup_s":warmup,"max_state_abs_error":max_error,
                "peak_gpu_bytes":torch.cuda.max_memory_allocated() if torch.device(args.device).type=="cuda" else None,
                "backend":None if backend is None else backend.status()}
        finally:
            if backend:backend.close()
        records.append(record)
        record["speedup_vs_reference"]=records[0]["median_s"]/record["median_s"]
        print(json.dumps(record),flush=True)
    root=Path(__file__).resolve().parents[1];digest=hashlib.sha256()
    for path in sorted([*(root/"dasklearn").rglob("*.py"),*(root/"scripts").glob("*.py")]):
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    report={"source_sha256":digest.hexdigest(),"torch":torch.__version__,"device":args.device,
            "shared_inputs":args.shared_inputs,"dataset_loading_s":loading,"records":records,
            "tolerance":{"rtol":1e-4,"atol":1e-5},"optimizer":"Fresh SGD per task, lr .01, momentum .9, 5 steps, batch 32"}
    (args.output/"benchmark.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
