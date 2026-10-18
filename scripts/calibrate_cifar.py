"""Measure actual GPU training/evaluation cost before choosing a sweep budget."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dasklearn.datasets.cifar10 import Cifar10Dataset
from dasklearn.model_trainer import ModelTrainer
from dasklearn.model_evaluator import ModelEvaluator
from dasklearn.models import create_model, serialize_model
from dasklearn.session_settings import SessionSettings, LearningSettings

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(0)
s = SessionSettings(algorithm="calibration", seed=0, work_dir=".", dataset="cifar10", model="gnlenet",
    learning=LearningSettings(.03, 0, 0, 32, 5), participants=128, dataset_base_path=args.data)
d = Cifar10Dataset(s)
m = create_model("cifar10", "gnlenet")
t = ModelTrainer(d, s, 0)
t.train(m, 5, device_name="cuda:0", seed=0)
torch.cuda.synchronize()
start = time.perf_counter()
for i in range(5):
    t.train(m, 5, device_name="cuda:0", seed=i)
torch.cuda.synchronize()
train_s = (time.perf_counter() - start) / 5
start = time.perf_counter()
accuracy, loss = ModelEvaluator(d, s).evaluate_accuracy(m, device_name="cuda:0")
torch.cuda.synchronize()
report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "five_steps_s": train_s,
    "full_test_s": time.perf_counter() - start, "model_bytes": len(serialize_model(m)),
    "accuracy_after_30_steps": accuracy, "loss": loss,
    "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "train_examples": sum(len(x) for x in d.partitions)}
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
