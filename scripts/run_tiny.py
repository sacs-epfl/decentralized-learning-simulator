"""One-seed, heterogeneous Tiny ImageNet/ResNet-18 protocol comparison."""
import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.conflux_v2.settings import ConfluxV2Settings
from dasklearn.simulation.conflux_v2.simulation import ConfluxV2Simulation
from dasklearn.simulation.shatter.settings import ShatterSettings
from dasklearn.simulation.shatter.simulation import ShatterSimulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from dasklearn import AUGMENTATION_FACTOR_SIM
from validate_conflux import plot


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--variant", choices=["shatter-r2", "conflux-v2"], required=True)
    p.add_argument("--peers", type=int, default=128)
    p.add_argument("--sample-size", type=int, default=32)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--duration", type=int, default=1800)
    p.add_argument("--test-period", type=int, default=300)
    p.add_argument("--evaluation-examples", type=int, default=2000)
    p.add_argument("--bandwidth", type=int, default=10000000)
    p.add_argument("--step-ms", type=float, default=50)
    p.add_argument("--learning-rate", type=float, default=.01)
    p.add_argument("--momentum", type=float, default=.9)
    p.add_argument("--local-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda[:index], or mps")
    p.add_argument("--release-cuda-cache", action="store_true")
    p.add_argument("--model-bytes", type=int, default=45195718,
                   help="Fixed simulated payload size, pinned to the original ResNet18 run")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--no-final-full", action="store_true")
    args = p.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    logging.basicConfig(level=logging.WARNING)
    Task.COUNTER.clear()
    cls, config = (ShatterSimulation, ShatterSettings) if args.variant == "shatter-r2" else (ConfluxV2Simulation, ConfluxV2Settings)
    specific = {"k": args.chunks, "r": 2} if args.variant == "shatter-r2" else {"sample_size": args.sample_size, "chunks_in_sample": args.chunks, "success_fraction": 1}
    settings = config(algorithm="shatter" if args.variant == "shatter-r2" else "conflux-v2", seed=args.seed,
        work_dir=str(args.output), dataset="tiny_imagenet", model="resnet18", participants=args.peers,
        dataset_base_path=args.data, partitioner="dirichlet", alpha=.1,
        torch_device_name=str(device), torch_threads=args.torch_threads,
        release_cuda_cache=args.release_cuda_cache, model_size_bytes=args.model_bytes,
        learning=LearningSettings(args.learning_rate, args.momentum, 0, 32, args.local_steps),
        rounds=0, duration=args.duration*MICROSECONDS, stop="duration", test_interval=0,
        test_method="snapshot", test_period=args.test_period*MICROSECONDS,
        evaluation_examples=args.evaluation_examples, network_latency_ms=20, dry_run=True, unit_testing=True, **specific)
    root = Path(__file__).resolve().parents[1]; digest = hashlib.sha256()
    for path in sorted(list((root/"dasklearn").rglob("*.py"))+list((root/"scripts").glob("*.py"))):
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    manifest = {**vars(args), "output":str(args.output), "dataset":"tiny_imagenet", "model":"resnet18",
        "alpha":.1, "torch":torch.__version__, "python":platform.python_version(), "numpy":np.__version__,
        "source_sha256":digest.hexdigest(), "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        "device":str(device),
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"), "input_size":64,
        "pretrained":False,"train_images":95000,"tuning_images":5000,"heldout_images":10000,
        "curve_examples":args.evaluation_examples,"final_examples":0 if args.no_final_full else 10000,
        "network":"quarter fast, 8x spread, 10 MB/s mean by default; independent compute ranking",
        "provenance":json.loads((Path(args.data)/"tiny-manifest.json").read_text())}
    manifest=json.loads(json.dumps(manifest,default=str))
    path=args.output/"manifest.json"
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError("Existing configuration/source differs")
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    if (args.output/"COMPLETE").exists():return

    if args.calibrate:
        from dasklearn import functions
        from dasklearn.datasets import create_dataset
        from dasklearn.model_evaluator import ModelEvaluator
        functions.dataset=create_dataset(settings); functions.model_trainers=None
        params={"model":(None,0),"round":1,"peer":0,"time":0}
        functions.train(settings,params)
        started=time.perf_counter()
        for i in range(3):
            params["round"]=i+2
            result=functions.train(settings,params)
        training=(time.perf_counter()-started)/3
        evaluator=ModelEvaluator(functions.dataset,settings)
        started=time.perf_counter();accuracy,loss=evaluator.evaluate_accuracy(result[0],str(device))
        report={"training_task_s":training,"evaluation_2000_s":time.perf_counter()-started,
                "accuracy_after_one_local_update":accuracy,"loss":loss,
                "peak_gpu_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None}
        (args.output/"calibration.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)
        return

    random.seed(args.seed)
    sim=cls(settings);sim.data_dir=str(args.output/"runs"/args.variant);settings.data_dir=sim.data_dir;sim.setup_directories()
    def capacities():
        rng=random.Random(args.seed+100);order=list(range(args.peers));rng.shuffle(order)
        fast=set(order[:max(1,args.peers//4)]);mean=(8*len(fast)+args.peers-len(fast))/args.peers
        compute=[rng.choice([.5,1,2,4]) for _ in range(args.peers)];cm=np.mean(compute)
        for client in sim.clients:
            client.bw_scheduler.bw_limit=int(args.bandwidth*(8 if client.index in fast else 1)/mean)
            client.simulated_speed=args.step_ms/AUGMENTATION_FACTOR_SIM*compute[client.index]/cm
    sim.apply_traces=capacities
    snapshot=sim.test_snapshot
    def final_snapshot(event):
        snapshot(event)
        if event.time==settings.duration and not args.no_final_full:
            for peer,(round_nr,model) in sorted(sim.latest_models.items()):
                sim.clients[peer].add_compute_task(Task(Task.generate_name("test_final"),"test",{
                    "model":model,"round":round_nr,"time":event.time,"peer":peer,
                    "evaluation_examples":0,"output_prefix":"final_accuracies"}))
    sim.test_snapshot=final_snapshot
    started=time.perf_counter();sim.simulate();sim.workflow_dag.build_task_indices();sim.workflow_dag.check_validity()
    print(json.dumps({"event_simulation_s":time.perf_counter()-started,"tasks":len(sim.workflow_dag.tasks),
          "model_bytes":sim.model_size,"training_tasks":sum(t.func=="train" for t in sim.workflow_dag.tasks.values())}),flush=True)
    def progress(status):
        print(json.dumps(status),flush=True)
    execute_local(sim,retain_results=False,progress=progress)
    sim.merge_accuracies_files()
    frame=pd.read_csv(Path(sim.data_dir)/"accuracies.csv")
    curve=frame.groupby("time",as_index=False).agg(accuracy=("accuracy","mean"),loss=("loss","mean"),nodes=("peer","nunique"))
    assert curve.nodes.eq(args.peers).all()
    curve["variant"]=args.variant;curve["profile"]="heterogeneous";curve["partitioner"]="dirichlet";curve["seed"]=args.seed
    curve.to_csv(args.output/"curves.csv",index=False);plot(curve,args.output)
    final_paths=list(Path(sim.data_dir).glob("final_accuracies_*.csv"))
    final=pd.concat([pd.read_csv(p) for p in final_paths],ignore_index=True) if final_paths else None
    if final is not None:
        assert len(final)==args.peers and final.peer.nunique()==args.peers
        final.to_csv(args.output/"final.csv",index=False)
    row={"variant":args.variant,"seed":args.seed,"peers":args.peers,"model_bytes":sim.model_size,
        "final_accuracy_2000":float(curve.accuracy.iloc[-1]),"final_loss_2000":float(curve.loss.iloc[-1]),
        "final_accuracy_10000":None if final is None else float(final.accuracy.mean()),
        "final_loss_10000":None if final is None else float(final.loss.mean()),
        "bytes_sent":sum(c.bw_scheduler.total_bytes_sent for c in sim.clients),
        "training_tasks":sum(t.func=="train" for t in sim.workflow_dag.tasks.values()),
        "actual_execution_s":time.perf_counter()-started}
    pd.DataFrame([row]).to_csv(args.output/"summary.csv",index=False)
    print(json.dumps(row),flush=True);(args.output/"COMPLETE").write_text("Completed one-seed Tiny ImageNet run\n")


if __name__=="__main__":asyncio.run(main())
